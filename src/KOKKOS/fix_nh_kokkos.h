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

#ifndef LMP_FIX_NH_KOKKOS_H
#define LMP_FIX_NH_KOKKOS_H

#include "compute_temp_kokkos.h"
#include "fix_nh.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<int TRICLINIC_FLAG>
struct TagFixNH_nh_v_press{};

template<int RMASS>
struct TagFixNH_nve_v{};

struct TagFixNH_nve_x{};

struct TagFixNH_nh_v_temp{};

struct TagFixNH_nhc_temp{};

struct TagFixNH_nh_v_temp_chain{};

template<class DeviceType>
class FixNHKokkos : public FixNH {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixNHKokkos(class LAMMPS *, int, char **);

  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void pre_exchange() override;
  void post_run() override;
  void write_restart(FILE *) override;
  double compute_scalar() override;
  double compute_vector(int) override;
  void *extract(const char *, int &) override;

  template<int TRICLINIC_FLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nh_v_press<TRICLINIC_FLAG>, const int&) const;

  template<int RMASS>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nve_v<RMASS>, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nve_x, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nh_v_temp, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nhc_temp, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nh_v_temp_chain, const int&) const;

 protected:
  void remap() override;

  void nve_x() override;            // may be overwritten by child classes
  void nve_v() override;
  void nh_v_press() override;
  void nh_v_temp() override;

  KK_FLOAT factor[3];

  class DomainKokkos *domainKK;

  // A thermostat-only run on one rank keeps its Nose-Hoover chain on the
  // device: the kinetic energy, the chain update and the velocity scaling
  // factors stay in device memory, so the integrator never waits for the
  // force evaluation to finish. The host chain state is brought up to date
  // only when something on the host reads it.

  void nhc_temp_integrate_device(bool);
  void upload_chain();
  void sync_chain_host();

  static constexpr int CHAIN_MAX = 16;    // longest chain integrated on the device

  bool chain_device_style;   // the style scales velocities with nh_v_temp() of this class
  bool chain_on_device;      // chain integrated by TagFixNH_nhc_temp
  bool chain_device_newer;   // device chain state is newer than the host copy
  bool chain_refresh;        // the chain update starts from d_kinetic
  ComputeTempKokkos<DeviceType> *temperatureKK;

  // quantities of a chain update that depend on host data only, evaluated
  // on the host as FixNH::nhc_temp_integrate() evaluates them

  double chain_tfactor;          // converts d_kinetic().t0 into the temperature
  double chain_kecoef;           // tdof*boltz
  double chain_kt;               // boltz*t_target
  double chain_ncfac;            // 1/nc_tchain
  double chain_dt8_arg;          // -ncfac*dt8
  double chain_dthalf_arg;       // -ncfac*dthalf
  double chain_eta_step;         // ncfac*dthalf
  double chain_mass[CHAIN_MAX];  // eta_mass

  Kokkos::View<typename ComputeTempKokkos<DeviceType>::CTEMP, DeviceType> d_kinetic;
  Kokkos::View<double, DeviceType> d_t_current;
  Kokkos::View<double *, DeviceType> d_eta, d_eta_dot, d_eta_dotdot;
  Kokkos::View<double *, DeviceType> d_factor_eta;    // one factor per chain loop

  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkfloat_1d_3 v;
  typename AT::t_kkacc_1d_3_const f;
  typename AT::t_kkfloat_1d rmass;
  typename AT::t_kkfloat_1d mass;
  typename AT::t_int_1d type;
  typename AT::t_int_1d mask;
};

}

#endif

