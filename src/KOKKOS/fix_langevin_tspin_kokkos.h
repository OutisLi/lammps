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
FixStyle(langevin/tspin/kk,FixLangevinTSpinKokkos<LMPDeviceType>);
FixStyle(langevin/tspin/kk/device,FixLangevinTSpinKokkos<LMPDeviceType>);
FixStyle(langevin/tspin/kk/host,FixLangevinTSpinKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_LANGEVIN_TSPIN_KOKKOS_H
#define LMP_FIX_LANGEVIN_TSPIN_KOKKOS_H

#include "fix_langevin_tspin.h"
#include "kokkos_type.h"

#include "Kokkos_Random.hpp"

namespace LAMMPS_NS {

struct TagFixLangevinTSpinBath {};
struct TagFixLangevinTSpinZero {};

// sum of the random spin forces and number of bath atoms of a rank

struct s_TSpinNoiseSum {
  double f[3];
  int count;
  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION s_TSpinNoiseSum() : f{0.0, 0.0, 0.0}, count(0) {}
  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION s_TSpinNoiseSum &operator+=(const s_TSpinNoiseSum &rhs)
  {
    f[0] += rhs.f[0];
    f[1] += rhs.f[1];
    f[2] += rhs.f[2];
    count += rhs.count;
    return *this;
  }
};

template<class DeviceType>
class FixLangevinTSpinKokkos : public FixLangevinTSpin {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef s_TSpinNoiseSum value_type;

  FixLangevinTSpinKokkos(class LAMMPS *, int, char **);

  void post_force(int) override;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixLangevinTSpinBath, const int &, s_TSpinNoiseSum &) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixLangevinTSpinZero, const int &) const;

 private:
  typename AT::t_kkfloat_1d_4 sp;
  typename AT::t_kkacc_1d_3 fm;
  typename AT::t_double_2d_lr s_dot;
  typename AT::t_kkfloat_2d dvector;
  typename AT::t_int_1d_const mask;

  // prefactors of the current step: friction, noise and the conversion of a
  // bath force into the fm encoding, and the mean random force to remove
  double gamma1_scale, gamma2_scale, inv_hbar;
  double fmean[3];

  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
};

}    // namespace LAMMPS_NS

#endif
#endif
