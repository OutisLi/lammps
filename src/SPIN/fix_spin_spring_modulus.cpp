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

#include "fix_spin_spring_modulus.h"

#include "atom.h"
#include "error.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSpinSpringModulus::FixSpinSpringModulus(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), spring_energy(0.0)
{
  if (narg != 5) error->all(FLERR, "Illegal fix spring/spin/mod command");
  if (!atom->sp_flag)
    error->all(FLERR, "Fix spring/spin/mod requires atom style spin");

  spring_constant = utils::numeric(FLERR, arg[3], false, lmp);
  target_magnitude = utils::numeric(FLERR, arg[4], false, lmp);
  if (spring_constant <= 0.0)
    error->all(FLERR, "Fix spring/spin/mod spring constant must be > 0.0");
  if (target_magnitude < 0.0)
    error->all(FLERR, "Fix spring/spin/mod target magnitude must be >= 0.0");

  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  energy_global_flag = 1;
}

/* ---------------------------------------------------------------------- */

int FixSpinSpringModulus::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSpinSpringModulus::init()
{
  if (utils::strmatch(update->integrate_style, "^respa"))
    error->all(FLERR, "Fix spring/spin/mod does not support RESPA");
}

/* ---------------------------------------------------------------------- */

void FixSpinSpringModulus::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixSpinSpringModulus::post_force(int /*vflag*/)
{
  double **sp = atom->sp;
  double **fm = atom->fm;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  spring_energy = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    const double displacement = sp[i][3] - target_magnitude;
    spring_energy += 0.5 * spring_constant * displacement * displacement;

    const double radial_force = -spring_constant * displacement;
    fm[i][0] += radial_force * sp[i][0];
    fm[i][1] += radial_force * sp[i][1];
    fm[i][2] += radial_force * sp[i][2];
  }
}

double FixSpinSpringModulus::compute_scalar()
{
  double global_energy = 0.0;
  MPI_Allreduce(&spring_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, world);
  return global_energy;
}
