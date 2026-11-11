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

#ifdef FIX_CLASS
// clang-format off
FixStyle(spring/tspin/kk,FixSpringTSpinKokkos<LMPDeviceType>);
FixStyle(spring/tspin/kk/device,FixSpringTSpinKokkos<LMPDeviceType>);
FixStyle(spring/tspin/kk/host,FixSpringTSpinKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_SPRING_TSPIN_KOKKOS_H
#define LMP_FIX_SPRING_TSPIN_KOKKOS_H

#include "fix_spring_tspin.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixSpringTSpin {};

template<class DeviceType>
class FixSpringTSpinKokkos : public FixSpringTSpin {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef double value_type;

  FixSpringTSpinKokkos(class LAMMPS *, int, char **);

  void post_force(int) override;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixSpringTSpin, const int &, double &) const;

 private:
  typename AT::t_kkfloat_1d_4 sp;
  typename AT::t_kkacc_1d_3 fm;
  typename AT::t_kkfloat_2d dvector;
  typename AT::t_int_1d_const mask;
  int dynamic_flag;    // 1 if the spin masses mark the dynamical spins
  double hbar;    // Planck constant over 2 pi, in eV/(rad.THz)
};

}    // namespace LAMMPS_NS

#endif
#endif
