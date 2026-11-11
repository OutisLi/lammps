// clang-format off
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

/* ----------------------------------------------------------------------
   Polak-Ribiere conjugate gradients over the atom coordinates x and the
   Cartesian spin vectors S = sp[0:3] * sp[3] in R^3, so that the moment
   magnitudes relax together with their directions and the positions.

   The magnetic force is read in the encoding of the tspin styles
   (tspin.h): a compatible interaction stores fm = |S|/hbar * (-dE/dS) in
   atom->fm, so the force on S is F = hbar/|S| * fm with all three
   components. The encoding carries no force at exactly zero modulus, so a
   moment of zero magnitude takes no force and stays zero.

   The spin vectors join the line search as extra per-atom variables: every
   step of the coordinates moves them along their search direction, dmax
   limits their change per step in moment units, and their forces enter the
   force norm of the stopping criterion.
------------------------------------------------------------------------- */

#include "min_spin_cg_r3.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"

#include <cmath>

using namespace LAMMPS_NS;
using MathConst::MY_2PI;

/* ---------------------------------------------------------------------- */

MinSpinCGR3::MinSpinCGR3(LAMMPS *lmp) :
    MinCG(lmp), nmax(0), spin(nullptr), spin_force(nullptr)
{
}

/* ---------------------------------------------------------------------- */

MinSpinCGR3::~MinSpinCGR3()
{
  memory->destroy(spin);
  memory->destroy(spin_force);
}

/* ---------------------------------------------------------------------- */

void MinSpinCGR3::setup_style()
{
  if (!atom->sp_flag) error->all(FLERR, "Min style spin/cg/r3 requires atom style spin");

  // the request must precede the line-search storage it sizes

  request(this, 3, dmax);
  MinLineSearch::setup_style();
}

/* ----------------------------------------------------------------------
   size the buffers for atom->nmax rows; after a reneighbor the forces are
   gathered before the vectors are reset, so both callbacks size them
------------------------------------------------------------------------- */

void MinSpinCGR3::grow()
{
  if (atom->nmax <= nmax) return;
  nmax = atom->nmax;
  memory->destroy(spin);
  memory->destroy(spin_force);
  memory->create(spin, nmax, 3, "min/spin/cg/r3:spin");
  memory->create(spin_force, nmax, 3, "min/spin/cg/r3:spin_force");
}

/* ----------------------------------------------------------------------
   atoms carry sp when they migrate, and min_x_set keeps sp equal to the
   minimizer state, so the spin buffer is rebuilt from sp for the local atoms
------------------------------------------------------------------------- */

void MinSpinCGR3::min_xf_pointers(int /*index*/, double **xextra, double **fextra)
{
  grow();

  double **sp = atom->sp;
  for (int i = 0; i < atom->nlocal; i++) {
    spin[i][0] = sp[i][0] * sp[i][3];
    spin[i][1] = sp[i][1] * sp[i][3];
    spin[i][2] = sp[i][2] * sp[i][3];
  }

  *xextra = nmax ? spin[0] : nullptr;
  *fextra = nmax ? spin_force[0] : nullptr;
}

/* ---------------------------------------------------------------------- */

void MinSpinCGR3::min_xf_get(int /*index*/)
{
  grow();

  const double hbar = force->hplanck / MY_2PI;
  double **sp = atom->sp;
  double **fm = atom->fm;
  for (int i = 0; i < atom->nlocal; i++) {
    const double scale = (sp[i][3] > 0.0) ? hbar / sp[i][3] : 0.0;
    spin_force[i][0] = scale * fm[i][0];
    spin_force[i][1] = scale * fm[i][1];
    spin_force[i][2] = scale * fm[i][2];
  }
}

/* ---------------------------------------------------------------------- */

void MinSpinCGR3::min_x_set(int /*index*/)
{
  double **sp = atom->sp;
  for (int i = 0; i < atom->nlocal; i++) {
    const double magnitude =
        sqrt(spin[i][0] * spin[i][0] + spin[i][1] * spin[i][1] + spin[i][2] * spin[i][2]);
    sp[i][3] = magnitude;
    if (magnitude > 0.0) {
      const double inverse = 1.0 / magnitude;
      sp[i][0] = spin[i][0] * inverse;
      sp[i][1] = spin[i][1] * inverse;
      sp[i][2] = spin[i][2] * inverse;
    }
  }
}

/* ---------------------------------------------------------------------- */

double MinSpinCGR3::memory_usage()
{
  return MinCG::memory_usage() + 6.0 * nmax * sizeof(double);
}
