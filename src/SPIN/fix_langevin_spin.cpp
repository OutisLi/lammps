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

#include "fix_langevin_spin.h"

#include "atom.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "spin_counter_rng.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixLangevinSpin::FixLangevinSpin(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), zero_flag(0)
{
  if (narg < 7) error->all(FLERR, "Illegal fix langevin/spin command");

  dynamic_group_allow = 1;
  t_start = utils::numeric(FLERR, arg[3], false, lmp);
  t_stop = utils::numeric(FLERR, arg[4], false, lmp);
  t_period = utils::numeric(FLERR, arg[5], false, lmp);
  seed = utils::inumeric(FLERR, arg[6], false, lmp);

  if (t_start < 0.0 || t_stop < 0.0)
    error->all(FLERR, "Fix langevin/spin temperature must be >= 0.0");
  if (t_period <= 0.0)
    error->all(FLERR, "Fix langevin/spin damping period must be > 0.0");
  if (seed <= 0) error->all(FLERR, "Fix langevin/spin seed must be > 0");

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "zero") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "fix langevin/spin zero", error);
      zero_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, iarg, "Unknown fix langevin/spin keyword: {}", arg[iarg]);
    }
  }

  // Under KOKKOS the integrator moves per-atom data between host and device
  // from these masks. The default claims every array; only the arrays this
  // fix touches are declared, and the magnetic force it adds to is the one
  // array it declares modified.
  datamask_read = SP_MASK | FM_MASK | MASK_MASK | SPV_MASK | SPM_MASK;
  datamask_modify = FM_MASK;
}

/* ---------------------------------------------------------------------- */

int FixLangevinSpin::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpin::init()
{
  if (!atom->sp_flag || !atom->spin_velocity || !atom->spin_mass)
    error->all(FLERR, "Fix langevin/spin requires atom style spin");
  if (strcmp(update->unit_style, "metal") != 0)
    error->all(FLERR, "Fix langevin/spin Newtonian dynamics requires metal units");
  if (utils::strmatch(update->integrate_style, "^respa"))
    error->all(FLERR, "Fix langevin/spin Newtonian dynamics does not support RESPA");
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpin::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixLangevinSpin::post_force(int /*vflag*/)
{
  double **spin_velocity = atom->spin_velocity;
  double **fm = atom->fm;
  double *spin_mass = atom->spin_mass;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  const double temperature = target_temperature();
  const double noise_scale =
    std::sqrt(2.0 * force->boltz * temperature /
              (t_period * update->dt * force->mvv2e)) / force->ftm2v;

  double local_random_sum[3] = {0.0, 0.0, 0.0};
  int local_active = 0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit) || spin_mass[i] <= 0.0) continue;

    const double drag = -spin_mass[i] / (t_period * force->ftm2v);
    const double atom_noise = std::sqrt(spin_mass[i]) * noise_scale;
    double random_force[3] = {
      atom_noise *
        SpinCounterRNG::gaussian(seed, update->ntimestep, tag[i], 0,
                                 SpinCounterRNG::Stream::LANGEVIN_FORCE),
      atom_noise *
        SpinCounterRNG::gaussian(seed, update->ntimestep, tag[i], 1,
                                 SpinCounterRNG::Stream::LANGEVIN_FORCE),
      atom_noise *
        SpinCounterRNG::gaussian(seed, update->ntimestep, tag[i], 2,
                                 SpinCounterRNG::Stream::LANGEVIN_FORCE)
    };

    fm[i][0] += drag * spin_velocity[i][0] + random_force[0];
    fm[i][1] += drag * spin_velocity[i][1] + random_force[1];
    fm[i][2] += drag * spin_velocity[i][2] + random_force[2];

    local_random_sum[0] += random_force[0];
    local_random_sum[1] += random_force[1];
    local_random_sum[2] += random_force[2];
    local_active++;
  }

  if (!zero_flag) return;

  double global_random_sum[3];
  int global_active = 0;
  MPI_Allreduce(local_random_sum, global_random_sum, 3, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&local_active, &global_active, 1, MPI_INT, MPI_SUM, world);
  if (global_active == 0)
    error->all(FLERR, "Cannot zero Langevin spin force for zero active spins");

  const double inverse_count = 1.0 / global_active;
  global_random_sum[0] *= inverse_count;
  global_random_sum[1] *= inverse_count;
  global_random_sum[2] *= inverse_count;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit) || spin_mass[i] <= 0.0) continue;
    fm[i][0] -= global_random_sum[0];
    fm[i][1] -= global_random_sum[1];
    fm[i][2] -= global_random_sum[2];
  }
}

double FixLangevinSpin::target_temperature() const
{
  double fraction = update->ntimestep - update->beginstep;
  const double run_steps = update->endstep - update->beginstep;
  if (run_steps > 0.0) fraction /= run_steps;
  else fraction = 0.0;
  return t_start + fraction * (t_stop - t_start);
}
