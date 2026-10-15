/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_nve_spin.h"

#include "atom.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "modify.h"
#include "spin_force_convention.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVESpin::FixNVESpin(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg)
{
  if ((narg - 3) % 2 != 0) error->all(FLERR, "Illegal fix nve/spin command");
  if (!atom->sp_flag)
    error->all(FLERR, "Fix nve/spin requires atom style spin");

  dynamic_group_allow = 1;
  time_integrate = 1;
  lattice_flag = 1;
  spin_flag = 1;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "lattice_flag") == 0) {
      lattice_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
    } else if (strcmp(arg[iarg], "spin_flag") == 0) {
      spin_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
    } else {
      error->all(FLERR, iarg, "Unknown fix nve/spin keyword: {}", arg[iarg]);
    }
    iarg += 2;
  }

  // Under KOKKOS the integrator moves per-atom data between host and device
  // from these masks. The default claims every array, which would mark the
  // magnetic force as modified on the host and collide with the device
  // scatter of an accelerated pair style. Only the arrays this fix touches
  // are declared, and only the coordinates it advances are declared modified.
  datamask_read = X_MASK | V_MASK | F_MASK | SP_MASK | FM_MASK | TYPE_MASK |
      RMASS_MASK | MASK_MASK;
  datamask_modify = X_MASK | V_MASK | SP_MASK;
}

/* ---------------------------------------------------------------------- */

int FixNVESpin::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVESpin::init()
{
  reset_dt();

  if (utils::strmatch(update->integrate_style, "^respa"))
    error->all(FLERR, "Fix nve/spin Newtonian dynamics does not support RESPA");
  if (strcmp(update->unit_style, "metal") != 0)
    error->all(FLERR, "Fix nve/spin Newtonian dynamics requires metal units");
  require_cartesian_magnetic_force(lmp, "Fix nve/spin Newtonian dynamics");
  if (!modify->get_fix_by_style("^precession/spin").empty() ||
      !modify->get_fix_by_style("^setforce/spin").empty())
    error->all(FLERR,
               "Fix nve/spin Newtonian dynamics cannot mix Cartesian "
               "magnetic forces with precession/spin or setforce/spin");

  if (!spin_flag) return;

  int local_active = 0;
  int *mask = atom->mask;
  double *spin_mass = atom->spin_mass;
  for (int i = 0; i < atom->nlocal; i++)
    if ((mask[i] & groupbit) && spin_mass[i] > 0.0) local_active++;

  int global_active = 0;
  MPI_Allreduce(&local_active, &global_active, 1, MPI_INT, MPI_SUM, world);
  if (global_active == 0)
    error->all(FLERR,
               "Fix nve/spin requires initialized spin masses; use "
               "'velocity ... create ... spin yes spmass VALUE'");
}

/* ---------------------------------------------------------------------- */

void FixNVESpin::initial_integrate(int /*vflag*/)
{
  if (lattice_flag) update_lattice(true);
  if (spin_flag) update_spins(true);
}

/* ---------------------------------------------------------------------- */

void FixNVESpin::final_integrate()
{
  if (lattice_flag) update_lattice(false);
  if (spin_flag) update_spins(false);
}

/* ----------------------------------------------------------------------
   half-step kick of the atomic velocities, followed by the full-step drift
   of the positions when drift is set; one pass over the atoms applies both
------------------------------------------------------------------------- */

void FixNVESpin::update_lattice(bool drift)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const double atom_mass = rmass ? rmass[i] : mass[type[i]];
    const double scale = dtf / atom_mass;
    v[i][0] += scale * f[i][0];
    v[i][1] += scale * f[i][1];
    v[i][2] += scale * f[i][2];
    if (drift) {
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
    }
  }
}

/* ----------------------------------------------------------------------
   half-step kick of the Cartesian spin velocities, followed by the full-step
   drift of the spin vectors when drift is set; the drifted vector is split
   back into unit direction and unconstrained magnitude in the same pass
------------------------------------------------------------------------- */

void FixNVESpin::update_spins(bool drift)
{
  double **sp = atom->sp;
  double **fm = atom->fm;
  double **spin_velocity = atom->spin_velocity;
  double *spin_mass = atom->spin_mass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit) || spin_mass[i] <= 0.0) continue;
    const double scale = dtf / spin_mass[i];
    spin_velocity[i][0] += scale * fm[i][0];
    spin_velocity[i][1] += scale * fm[i][1];
    spin_velocity[i][2] += scale * fm[i][2];
    if (!drift) continue;

    double spin[3] = {
      sp[i][0] * sp[i][3] + dtv * spin_velocity[i][0],
      sp[i][1] * sp[i][3] + dtv * spin_velocity[i][1],
      sp[i][2] * sp[i][3] + dtv * spin_velocity[i][2]
    };
    const double magnitude =
      std::sqrt(spin[0] * spin[0] + spin[1] * spin[1] + spin[2] * spin[2]);
    if (!std::isfinite(magnitude))
      error->one(FLERR, "Non-numeric spin magnitude - simulation unstable");

    sp[i][3] = magnitude;
    if (magnitude > 0.0) {
      const double inverse_magnitude = 1.0 / magnitude;
      sp[i][0] = spin[0] * inverse_magnitude;
      sp[i][1] = spin[1] * inverse_magnitude;
      sp[i][2] = spin[2] * inverse_magnitude;
    }
  }
}

void FixNVESpin::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}
