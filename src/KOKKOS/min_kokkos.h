// clang-format off
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

#ifndef LMP_MIN_KOKKOS_H
#define LMP_MIN_KOKKOS_H

#include "min.h"
#include "kokkos_type.h"

#include <vector>

namespace LAMMPS_NS {

// Owner of an extra per-atom minimization variable that keeps it, with three
// values per atom, and the force on it in device views, such as min style
// spin/cg/r3/kk.  The KOKKOS minimizers move it on the device together with
// the atom coordinates.

class MinExtraDOFKokkos : public MinExtraDOF {
 public:
  // point to the variable and to the force on it, after atoms migrated
  virtual void min_xf_views(int, DAT::t_kkfloat_1d &, DAT::t_kkacc_1d &) = 0;

 protected:
  ~MinExtraDOFKokkos() = default;
};

class MinKokkos : public Min {
 public:
  MinKokkos(class LAMMPS *);

  static constexpr int F_LAYOUTRIGHT = std::is_same<LMPDeviceLayout,Kokkos::LayoutRight>::value;

  void init() override;
  void setup(int flag=1) override;
  void setup_minimal(int) override;
  void run(int) override;
  double fnorm_sqr() override;
  double fnorm_inf() override;
  double fnorm_max() override;

  void init_style() override {}
  void setup_style() override = 0;
  void reset_vectors() override = 0;
  int iterate(int) override = 0;

  // possible return values of iterate() method
  enum{MAXITER,MAXEVAL,ETOL,FTOL,DOWNHILL,ZEROALPHA,ZEROFORCE,
       ZEROQUAD,TRSMALL,INTERROR,TIMEOUT};

 //protected: // won't compile with CUDA
  class FixMinimizeKokkos *fix_minimize_kk;  // fix that stores auxiliary data

  DAT::t_kkfloat_1d xvec;            // variables for atomic dof, as 1d vector
  DAT::t_kkacc_1d fvec;            // force vector for atomic dof, as 1d vector

  std::vector<DAT::t_kkfloat_1d> xextra_atom_kk;   // extra per-atom dof
  std::vector<DAT::t_kkacc_1d> fextra_atom_kk;     // forces on them

  double energy_force(int) override;
  void force_clear() override;
};

}

#endif

