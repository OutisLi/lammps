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
FixStyle(nve/tspin/kk,FixNVETSpinKokkos<LMPDeviceType>);
FixStyle(nve/tspin/kk/device,FixNVETSpinKokkos<LMPDeviceType>);
FixStyle(nve/tspin/kk/host,FixNVETSpinKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_NVE_TSPIN_KOKKOS_H
#define LMP_FIX_NVE_TSPIN_KOKKOS_H

#include "fix_nve_tspin.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixNVETSpinInitial {};
struct TagFixNVETSpinFinal {};

template<class DeviceType>
class FixNVETSpinKokkos : public FixNVETSpin {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixNVETSpinKokkos(class LAMMPS *, int, char **);

  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVETSpinInitial, const int &) const;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVETSpinFinal, const int &) const;

 private:
  int rmass_flag;
  uint64_t datamask_final;    // arrays the final half step writes

  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkfloat_1d_3 v;
  typename AT::t_kkacc_1d_3_const f;
  typename AT::t_kkfloat_1d_4 sp;
  typename AT::t_kkacc_1d_3_const fm;
  typename AT::t_kkfloat_1d_const rmass;
  typename AT::t_kkfloat_1d_const mass;
  typename AT::t_int_1d_const type;
  typename AT::t_int_1d_const mask;

  // the TSPIN state of fix property/atom/kk: the spin velocities, row
  // index_sm of the custom vectors for the spin masses

  typename AT::t_double_2d_lr s_dot;
  typename AT::t_kkfloat_2d dvector;

  void set_views();

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
