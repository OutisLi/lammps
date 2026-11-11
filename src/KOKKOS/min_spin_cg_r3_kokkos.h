/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef MINIMIZE_CLASS
// clang-format off
MinimizeStyle(spin/cg/r3/kk,MinSpinCGR3Kokkos);
MinimizeStyle(spin/cg/r3/kk/device,MinSpinCGR3Kokkos);
MinimizeStyle(spin/cg/r3/kk/host,MinSpinCGR3Kokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_MIN_SPIN_CG_R3_KOKKOS_H
#define LMP_MIN_SPIN_CG_R3_KOKKOS_H

#include "min_cg_kokkos.h"

namespace LAMMPS_NS {

// Conjugate gradients over the atom coordinates and the Cartesian spin
// vectors on the device, the minimization of min style spin/cg/r3

class MinSpinCGR3Kokkos : public MinCGKokkos, public MinExtraDOFKokkos {
 public:
  MinSpinCGR3Kokkos(class LAMMPS *);
  void setup_style() override;
  double memory_usage() override;

  void min_xf_views(int, DAT::t_kkfloat_1d &, DAT::t_kkacc_1d &) override;
  void min_xf_get(int) override;
  void min_x_set(int) override;

 //protected: // won't compile with CUDA
  int nmax;                        // atoms the spin buffers hold
  DAT::t_kkfloat_1d d_spin;        // Cartesian spin vector of each local atom
  DAT::t_kkacc_1d d_spin_force;    // force -dE/dS on it, zero for a zero moment

  void grow();
};

}    // namespace LAMMPS_NS

#endif
#endif
