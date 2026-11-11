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

// clang-format off
#ifndef LMP_FIX_NH_TSPIN_KOKKOS_H
#define LMP_FIX_NH_TSPIN_KOKKOS_H

#include "fix_nh_kokkos.h"
#include "fix_nh_tspin.h"

namespace LAMMPS_NS {

struct TagFixNHTSpinKick {};
struct TagFixNHTSpinDrift {};
struct TagFixNHTSpinScale {};
struct TagFixNHTSpinKinetic {};

// Nose-Hoover base of fix nvt/tspin/kk, npt/tspin/kk and nph/tspin/kk: the
// integration of FixNHTSpinT over fix nh/kk, with the kick, drift and chain
// scaling of the spin velocities on the same device as the lattice

template<class DeviceType>
class FixNHTSpinKokkos : public FixNHTSpinT<FixNHKokkos<DeviceType>> {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixNHTSpinKokkos(class LAMMPS *, int, char **);

  void setup(int) override;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNHTSpinKick, const int &) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNHTSpinDrift, const int &) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNHTSpinScale, const int &) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNHTSpinKinetic, const int &, double &) const;

 protected:
  double compute_spin_temp() override;
  void nh_vs_scale(double) override;
  void spin_kick() override;
  void spin_drift() override;

 private:
  double spin_factor;     // velocity scaling of the chain update in progress

  typename AT::t_kkfloat_1d_4 sp;
  typename AT::t_kkacc_1d_3_const fm;
  typename AT::t_double_2d_lr s_dot;
  typename AT::t_kkfloat_2d dvector;

  int set_spin_views(uint64_t);
};

}    // namespace LAMMPS_NS

#endif
