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

/* ------------------------------------------------------------------------
   Contributing author: Zhengtao Huang (The University of Hong Kong)
                        hzt990224@gmail.com

   The spin velocities are coupled to a second Nose-Hoover chain.  That
   chain is independent of the lattice chain of fix nh but is driven by the
   same target temperature and the same damping parameter, so that a
   spin-lattice simulation is thermostatted as a whole.
------------------------------------------------------------------------- */

#include "tspin_nh.h"

#include "atom.h"
#include "error.h"
#include "lammps.h"
#include "memory.h"
#include "tspin.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <string>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

TSpinNH::TSpinNH(LAMMPS *lmp) :
    lattice_flag(1), spin_flag(1), spinmass_flag(0), spinmass(1.0), index_vs(-1), index_sm(-1),
    hbar(0.0), t_current_spin(0.0), ke_target_spin(0.0), tdof_spin(0.0), mtchain_spin(0),
    etas(nullptr), etas_dot(nullptr), etas_dotdot(nullptr), etas_mass(nullptr), tspin_lmp(lmp)
{
}

/* ----------------------------------------------------------------------
   parse the keywords that fix nh skips for the tspin styles
------------------------------------------------------------------------- */

void TSpinNH::parse_keywords(const char *style, int narg, char **arg, int pstat_flag)
{
  Error *error = tspin_lmp->error;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "spinmass") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, std::string("fix ") + style + " spinmass", error);
      spinmass = utils::numeric(FLERR, arg[iarg + 1], false, tspin_lmp);
      if (spinmass <= 0.0) error->all(FLERR, iarg + 1, "Fix {} spinmass must be > 0", style);
      spinmass_flag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "lattice") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, std::string("fix ") + style + " lattice", error);
      const std::string value = arg[iarg + 1];
      if (value == "frozen")
        lattice_flag = 0;
      else if ((value == "moving") || (value == "mobile"))
        lattice_flag = 1;
      else
        lattice_flag = utils::logical(FLERR, arg[iarg + 1], false, tspin_lmp) ? 1 : 0;
      iarg += 2;
    } else if (strcmp(arg[iarg], "spin") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, std::string("fix ") + style + " spin", error);
      const std::string svalue = arg[iarg + 1];
      if (svalue == "frozen")
        spin_flag = 0;
      else if ((svalue == "moving") || (svalue == "mobile"))
        spin_flag = 1;
      else
        spin_flag = utils::logical(FLERR, arg[iarg + 1], false, tspin_lmp) ? 1 : 0;
      iarg += 2;

      // the value of the dilate keyword of the parent class is a group ID and
      // could be spelled like one of the keywords above, so skip it explicitly

    } else if (strcmp(arg[iarg], "dilate") == 0) {
      iarg += 2;
    } else {
      iarg++;    // every other keyword belongs to the parent class
    }
  }

  if (!lattice_flag && !spin_flag)
    error->all(FLERR, "Fix {} has nothing to integrate with both lattice and spin frozen", style);
  if (pstat_flag && !lattice_flag)
    error->all(FLERR, "Fix {} cannot use a barostat with lattice frozen", style);
}

/* ----------------------------------------------------------------------
   the spin chain has the same length as the lattice chain
   etas_dot has one extra element so the update can read etas_dot[ich+1]
------------------------------------------------------------------------- */

void TSpinNH::create_chain(int mtchain)
{
  Memory *memory = tspin_lmp->memory;
  mtchain_spin = mtchain;
  memory->create(etas, mtchain, "nh/tspin:etas");
  memory->create(etas_dot, mtchain + 1, "nh/tspin:etas_dot");
  memory->create(etas_dotdot, mtchain, "nh/tspin:etas_dotdot");
  memory->create(etas_mass, mtchain, "nh/tspin:etas_mass");

  for (int ich = 0; ich < mtchain; ich++) etas[ich] = etas_dot[ich] = etas_dotdot[ich] = 0.0;
  etas_dot[mtchain] = 0.0;
}

/* ---------------------------------------------------------------------- */

void TSpinNH::destroy_chain()
{
  Memory *memory = tspin_lmp->memory;
  memory->destroy(etas);
  memory->destroy(etas_dot);
  memory->destroy(etas_dotdot);
  memory->destroy(etas_mass);
}

/* ----------------------------------------------------------------------
   assign the spin mass as a multiple of the atomic mass of the atom type
------------------------------------------------------------------------- */

void TSpinNH::set_spin_mass(int groupbit)
{
  Atom *atom = tspin_lmp->atom;
  double **sp = atom->sp;
  double *s_mass = atom->dvector[index_sm];
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  const int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if ((sp[i][3] <= TSPIN_EPS) && (s_mass[i] <= 0.0)) continue;
    s_mass[i] = spinmass * (rmass ? rmass[i] : mass[type[i]]);
  }
}

/* ----------------------------------------------------------------------
   count the thermostatted spin degrees of freedom and check the spin masses
------------------------------------------------------------------------- */

void TSpinNH::count_spin_dof(const char *style, int groupbit)
{
  Atom *atom = tspin_lmp->atom;
  double **sp = atom->sp;
  double *s_mass = atom->dvector[index_sm];
  int *mask = atom->mask;
  const int nlocal = atom->nlocal;

  int ndof = 0, flag = 0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (s_mass[i] > 0.0)
      ndof += 3;
    else if (sp[i][3] > TSPIN_EPS)
      flag = 1;
  }

  int ndof_all = 0, flagall = 0;
  MPI_Allreduce(&ndof, &ndof_all, 1, MPI_INT, MPI_SUM, tspin_lmp->world);
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, tspin_lmp->world);
  tdof_spin = ndof_all;

  if (flagall)
    tspin_lmp->error->all(FLERR,
                          "Fix {} requires a spin mass for every magnetic atom in the group; "
                          "use the spinmass keyword or the velocity/tspin command",
                          style);
  if (tdof_spin == 0.0)
    tspin_lmp->error->all(FLERR, "Fix {} has no spin degrees of freedom to thermostat", style);
}

/* ----------------------------------------------------------------------
   initialize the spin chain masses from the target temperature that the
   parent class has just computed
------------------------------------------------------------------------- */

void TSpinNH::init_chain_masses(const TSpinNHStep &step)
{
  const double boltz = step.boltz, t_target = step.t_target, t_freq = step.t_freq;
  etas_mass[0] = tdof_spin * boltz * t_target / (t_freq * t_freq);
  for (int ich = 1; ich < mtchain_spin; ich++)
    etas_mass[ich] = boltz * t_target / (t_freq * t_freq);
  for (int ich = 1; ich < mtchain_spin; ich++)
    etas_dotdot[ich] =
        (etas_mass[ich - 1] * etas_dot[ich - 1] * etas_dot[ich - 1] - boltz * t_target) /
        etas_mass[ich];
}

/* ----------------------------------------------------------------------
   propagate the spin Nose-Hoover chain by dthalf
   same scheme as FixNH::nhc_temp_integrate, acting on the spin velocities,
   from the instantaneous spin temperature of the spin kinetic energy
------------------------------------------------------------------------- */

void TSpinNH::integrate_chain(const TSpinNHStep &step)
{
  const int mtchain = mtchain_spin;
  const double boltz = step.boltz, t_target = step.t_target, t_freq = step.t_freq;
  const double dt4 = step.dt4, dt8 = step.dt8, dthalf = step.dthalf;
  const double tdrag_factor = step.tdrag_factor;

  ke_target_spin = tdof_spin * boltz * t_target;
  t_current_spin = compute_spin_temp();

  int ich;
  double expfac;
  double kecurrent = tdof_spin * boltz * t_current_spin;

  etas_mass[0] = tdof_spin * boltz * t_target / (t_freq * t_freq);
  for (ich = 1; ich < mtchain; ich++) etas_mass[ich] = boltz * t_target / (t_freq * t_freq);

  if (etas_mass[0] > 0.0)
    etas_dotdot[0] = (kecurrent - ke_target_spin) / etas_mass[0];
  else
    etas_dotdot[0] = 0.0;

  const double ncfac = 1.0 / step.nc_tchain;
  for (int iloop = 0; iloop < step.nc_tchain; iloop++) {

    for (ich = mtchain - 1; ich > 0; ich--) {
      expfac = exp(-ncfac * dt8 * etas_dot[ich + 1]);
      etas_dot[ich] *= expfac;
      etas_dot[ich] += etas_dotdot[ich] * ncfac * dt4;
      etas_dot[ich] *= tdrag_factor;
      etas_dot[ich] *= expfac;
    }

    expfac = exp(-ncfac * dt8 * etas_dot[1]);
    etas_dot[0] *= expfac;
    etas_dot[0] += etas_dotdot[0] * ncfac * dt4;
    etas_dot[0] *= tdrag_factor;
    etas_dot[0] *= expfac;

    const double factor_etas = exp(-ncfac * dthalf * etas_dot[0]);
    nh_vs_scale(factor_etas);

    // rescale the temperature to account for the velocity scaling

    t_current_spin *= factor_etas * factor_etas;
    kecurrent = tdof_spin * boltz * t_current_spin;

    if (etas_mass[0] > 0.0)
      etas_dotdot[0] = (kecurrent - ke_target_spin) / etas_mass[0];
    else
      etas_dotdot[0] = 0.0;

    for (ich = 0; ich < mtchain; ich++) etas[ich] += ncfac * dthalf * etas_dot[ich];

    etas_dot[0] *= expfac;
    etas_dot[0] += etas_dotdot[0] * ncfac * dt4;
    etas_dot[0] *= expfac;

    for (ich = 1; ich < mtchain; ich++) {
      expfac = exp(-ncfac * dt8 * etas_dot[ich + 1]);
      etas_dot[ich] *= expfac;
      etas_dotdot[ich] =
          (etas_mass[ich - 1] * etas_dot[ich - 1] * etas_dot[ich - 1] - boltz * t_target) /
          etas_mass[ich];
      etas_dot[ich] += etas_dotdot[ich] * ncfac * dt4;
      etas_dot[ich] *= expfac;
    }
  }
}

/* ----------------------------------------------------------------------
   energy of the spin chain, part of the conserved quantity
------------------------------------------------------------------------- */

double TSpinNH::chain_energy(const TSpinNHStep &step) const
{
  const double boltz = step.boltz, t_target = step.t_target;
  double energy = tdof_spin * boltz * t_target * etas[0];
  for (int ich = 1; ich < mtchain_spin; ich++) energy += boltz * t_target * etas[ich];
  for (int ich = 0; ich < mtchain_spin; ich++)
    energy += 0.5 * etas_mass[ich] * etas_dot[ich] * etas_dot[ich];
  return energy;
}

/* ----------------------------------------------------------------------
   the spin chain state follows the restart data of the parent class
------------------------------------------------------------------------- */

int TSpinNH::pack_chain(double *list, int n) const
{
  for (int ich = 0; ich < mtchain_spin; ich++) list[n++] = etas[ich];
  for (int ich = 0; ich < mtchain_spin; ich++) list[n++] = etas_dot[ich];
  return n;
}

/* ---------------------------------------------------------------------- */

void TSpinNH::unpack_chain(const double *list, int n)
{
  for (int ich = 0; ich < mtchain_spin; ich++) etas[ich] = list[n++];
  for (int ich = 0; ich < mtchain_spin; ich++) etas_dot[ich] = list[n++];
}
