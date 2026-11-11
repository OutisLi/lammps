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
   Cartesian spin vectors S = sp[0:3] * sp[3] on the device, the
   minimization of MinSpinCGR3 with the line search of min style cg/kk.

   The spin vectors are extra per-atom variables of three values per atom,
   kept in device buffers of this style.  The force on S is hbar/|S| * fm
   with all three components, fm being the magnetic force in the encoding
   of the tspin styles, and zero at zero modulus, where the encoding
   carries no force.
------------------------------------------------------------------------- */

#include "min_spin_cg_r3_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using MathConst::MY_2PI;

/* ---------------------------------------------------------------------- */

MinSpinCGR3Kokkos::MinSpinCGR3Kokkos(LAMMPS *lmp) : MinCGKokkos(lmp), nmax(0)
{
}

/* ---------------------------------------------------------------------- */

void MinSpinCGR3Kokkos::setup_style()
{
  if (!atom->sp_flag) error->all(FLERR, "Min style spin/cg/r3/kk requires atom style spin");

  // the request must precede the line-search storage it sizes

  request(this, 3, dmax);
  MinLineSearchKokkos::setup_style();
}

/* ----------------------------------------------------------------------
   size the buffers for atom->nmax atoms; after a reneighbor the forces are
   gathered before the vectors are reset, so both callbacks size them
------------------------------------------------------------------------- */

void MinSpinCGR3Kokkos::grow()
{
  if (atom->nmax <= nmax) return;
  nmax = atom->nmax;
  d_spin = DAT::t_kkfloat_1d("min/spin/cg/r3/kk:spin", 3 * nmax);
  d_spin_force = DAT::t_kkacc_1d("min/spin/cg/r3/kk:spin_force", 3 * nmax);
}

/* ----------------------------------------------------------------------
   atoms carry sp when they migrate, and min_x_set keeps sp equal to the
   minimizer state, so the spin buffer is rebuilt from sp for the local atoms
------------------------------------------------------------------------- */

void MinSpinCGR3Kokkos::min_xf_views(int /*index*/, DAT::t_kkfloat_1d &xextra,
                                     DAT::t_kkacc_1d &fextra)
{
  grow();
  atomKK->sync(Device, SP_MASK);

  auto l_sp = atomKK->k_sp.view_device();
  auto l_spin = d_spin;
  Kokkos::parallel_for(atom->nlocal, LAMMPS_LAMBDA(const int &i) {
    for (int k = 0; k < 3; k++) l_spin[3*i + k] = l_sp(i,k) * l_sp(i,3);
  });

  xextra = d_spin;
  fextra = d_spin_force;
}

/* ---------------------------------------------------------------------- */

void MinSpinCGR3Kokkos::min_xf_get(int /*index*/)
{
  grow();
  atomKK->sync(Device, SP_MASK | FM_MASK);

  const double hbar = force->hplanck / MY_2PI;
  auto l_sp = atomKK->k_sp.view_device();
  auto l_fm = atomKK->k_fm.view_device();
  auto l_spin_force = d_spin_force;
  Kokkos::parallel_for(atom->nlocal, LAMMPS_LAMBDA(const int &i) {
    const double modulus = l_sp(i,3);
    const double scale = (modulus > 0.0) ? hbar / modulus : 0.0;
    for (int k = 0; k < 3; k++)
      l_spin_force[3*i + k] = static_cast<KK_ACC_FLOAT>(scale * static_cast<double>(l_fm(i,k)));
  });
}

/* ---------------------------------------------------------------------- */

void MinSpinCGR3Kokkos::min_x_set(int /*index*/)
{
  atomKK->sync(Device, SP_MASK);

  auto l_sp = atomKK->k_sp.view_device();
  auto l_spin = d_spin;
  Kokkos::parallel_for(atom->nlocal, LAMMPS_LAMBDA(const int &i) {
    const double s0 = l_spin[3*i];
    const double s1 = l_spin[3*i + 1];
    const double s2 = l_spin[3*i + 2];
    const double magnitude = sqrt(s0 * s0 + s1 * s1 + s2 * s2);
    l_sp(i,3) = magnitude;
    if (magnitude > 0.0) {
      const double inverse = 1.0 / magnitude;
      l_sp(i,0) = s0 * inverse;
      l_sp(i,1) = s1 * inverse;
      l_sp(i,2) = s2 * inverse;
    }
  });

  atomKK->modified(Device, SP_MASK);
}

/* ---------------------------------------------------------------------- */

double MinSpinCGR3Kokkos::memory_usage()
{
  return MinCGKokkos::memory_usage() + 3.0 * nmax * (sizeof(KK_FLOAT) + sizeof(KK_ACC_FLOAT));
}
