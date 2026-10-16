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
FixStyle(nve/spin/kk,FixNVESpinKokkos<LMPDeviceType>);
FixStyle(nve/spin/kk/device,FixNVESpinKokkos<LMPDeviceType>);
FixStyle(nve/spin/kk/host,FixNVESpinKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_NVE_SPIN_KOKKOS_H
#define LMP_FIX_NVE_SPIN_KOKKOS_H

#include "fix_nve_spin.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixNVESpinInitial {};
struct TagFixNVESpinFinal {};

template<class DeviceType>
class FixNVESpinKokkos : public FixNVESpin {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixNVESpinKokkos(class LAMMPS *, int, char **);

  void init() override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void post_run() override;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVESpinInitial, const int &) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVESpinFinal, const int &) const;

 private:
  int rmass_flag;
  uint64_t datamask_final;    // arrays the final half step writes

  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkfloat_1d_3 v;
  typename AT::t_kkacc_1d_3_const f;
  typename AT::t_kkfloat_1d_4 sp;
  typename AT::t_kkacc_1d_3_const fm;
  typename AT::t_kkfloat_1d_3 spin_velocity;
  typename AT::t_kkfloat_1d_const spin_mass;
  typename AT::t_kkfloat_1d_const rmass;
  typename AT::t_kkfloat_1d_const mass;
  typename AT::t_int_1d_const type;
  typename AT::t_int_1d_const mask;

  // set when a drifted spin vector has a non-finite magnitude
  typename AT::t_int_scalar d_nonfinite;

  void set_views();
  void check_spin_magnitude();

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void update_lattice_item(int, bool) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void update_spin_item(int, bool) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
