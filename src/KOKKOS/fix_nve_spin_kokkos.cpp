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

#include "fix_nve_spin_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "math_rounded_kokkos.h"
#include "output.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace MathRoundedKokkos;

/* ----------------------------------------------------------------------
   Newtonian Cartesian velocity-Verlet integration of the lattice and the
   spin vectors on the device, step for step the update of FixNVESpin

   Every multiplication, addition and division is rounded on its own
   (math_rounded_kokkos.h) and evaluated in the order of the host fix, so a
   double-precision device trajectory agrees with the host fix bit for bit
   where the host compiler keeps its multiplications and additions apart.
   A drifted spin vector with a non-finite magnitude is recorded on the
   device and reported at the next thermo output or at the end of the run,
   so that a step needs no reply from the device.
------------------------------------------------------------------------- */

template<class DeviceType>
FixNVESpinKokkos<DeviceType>::FixNVESpinKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixNVESpin(lmp, narg, arg), rmass_flag(0)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = MASK_MASK;
  datamask_modify = datamask_final = EMPTY_MASK;
  if (lattice_flag) {
    datamask_read |= X_MASK | V_MASK | F_MASK | TYPE_MASK | RMASS_MASK;
    datamask_modify |= X_MASK | V_MASK;
    datamask_final |= V_MASK;
  }
  if (spin_flag) {
    datamask_read |= SP_MASK | FM_MASK | SPV_MASK | SPM_MASK;
    datamask_modify |= SP_MASK | SPV_MASK;
    datamask_final |= SPV_MASK;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVESpinKokkos<DeviceType>::init()
{
  // FixNVESpin::init() counts the atoms with spin inertia on the host

  atomKK->sync(Host, MASK_MASK | SPM_MASK);
  FixNVESpin::init();

  atomKK->k_mass.modify_host();
  atomKK->k_mass.sync<DeviceType>();

  if (!d_nonfinite.data()) d_nonfinite = typename AT::t_int_scalar("nve/spin/kk:nonfinite");
  Kokkos::deep_copy(d_nonfinite, 0);
}

/* ----------------------------------------------------------------------
   half-step kicks of the atomic and spin velocities, each followed by the
   full-step drift of the positions and spin vectors
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNVESpinKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  atomKK->sync(execution_space, datamask_read);
  set_views();

  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVESpinInitial>(0, nlocal), *this);
  copymode = 0;

  atomKK->modified(execution_space, datamask_modify);
}

/* ----------------------------------------------------------------------
   half-step kicks of the atomic and spin velocities
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNVESpinKokkos<DeviceType>::final_integrate()
{
  atomKK->sync(execution_space, datamask_read);
  set_views();

  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVESpinFinal>(0, nlocal), *this);
  copymode = 0;

  atomKK->modified(execution_space, datamask_final);

  if (spin_flag && (update->ntimestep == output->next_thermo)) check_spin_magnitude();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVESpinKokkos<DeviceType>::post_run()
{
  if (spin_flag) check_spin_magnitude();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVESpinKokkos<DeviceType>::set_views()
{
  rmass_flag = atomKK->rmass_flag;
  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  sp = atomKK->k_sp.view<DeviceType>();
  fm = atomKK->k_fm.view<DeviceType>();
  spin_velocity = atomKK->k_spin_velocity.view<DeviceType>();
  spin_mass = atomKK->k_spin_mass.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVESpinKokkos<DeviceType>::check_spin_magnitude()
{
  int nonfinite = 0;
  Kokkos::deep_copy(nonfinite, d_nonfinite);
  if (nonfinite) error->one(FLERR, "Non-numeric spin magnitude - simulation unstable");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNVESpinKokkos<DeviceType>::operator()(TagFixNVESpinInitial, const int &i) const
{
  if (!(mask[i] & groupbit)) return;
  if (lattice_flag) update_lattice_item(i, true);
  if (spin_flag && !(spin_mass[i] <= 0.0)) update_spin_item(i, true);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNVESpinKokkos<DeviceType>::operator()(TagFixNVESpinFinal, const int &i) const
{
  if (!(mask[i] & groupbit)) return;
  if (lattice_flag) update_lattice_item(i, false);
  if (spin_flag && !(spin_mass[i] <= 0.0)) update_spin_item(i, false);
}

/* ----------------------------------------------------------------------
   v += dtf/m f, followed by x += dtv v when drift is set
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNVESpinKokkos<DeviceType>::update_lattice_item(int i, bool drift) const
{
  const KK_FLOAT atom_mass = rmass_flag ? rmass[i] : mass[type[i]];
  const KK_FLOAT scale = div_rn(static_cast<KK_FLOAT>(dtf), atom_mass);
  for (int k = 0; k < 3; k++)
    v(i,k) = add_rn(v(i,k), mul_rn(scale, static_cast<KK_FLOAT>(f(i,k))));
  if (!drift) return;
  const KK_FLOAT dtv_kk = static_cast<KK_FLOAT>(dtv);
  for (int k = 0; k < 3; k++)
    x(i,k) = add_rn(x(i,k), mul_rn(dtv_kk, v(i,k)));
}

/* ----------------------------------------------------------------------
   V_S += dtf/I F_S, followed by S += dtv V_S when drift is set; the drifted
   vector S = |S| s is split back into the unit direction s, stored in sp[0..2],
   and the unconstrained magnitude |S|, stored in sp[3]
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNVESpinKokkos<DeviceType>::update_spin_item(int i, bool drift) const
{
  const KK_FLOAT scale = div_rn(static_cast<KK_FLOAT>(dtf), spin_mass[i]);
  for (int k = 0; k < 3; k++)
    spin_velocity(i,k) =
      add_rn(spin_velocity(i,k), mul_rn(scale, static_cast<KK_FLOAT>(fm(i,k))));
  if (!drift) return;

  const KK_FLOAT dtv_kk = static_cast<KK_FLOAT>(dtv);
  KK_FLOAT spin[3];
  for (int k = 0; k < 3; k++)
    spin[k] = add_rn(mul_rn(sp(i,k), sp(i,3)), mul_rn(dtv_kk, spin_velocity(i,k)));
  const KK_FLOAT magnitude = Kokkos::sqrt(
    add_rn(add_rn(mul_rn(spin[0], spin[0]), mul_rn(spin[1], spin[1])), mul_rn(spin[2], spin[2])));
  if (!Kokkos::isfinite(magnitude)) d_nonfinite() = 1;

  sp(i,3) = magnitude;
  if (magnitude > 0.0) {
    const KK_FLOAT inverse_magnitude = div_rn(static_cast<KK_FLOAT>(1.0), magnitude);
    for (int k = 0; k < 3; k++) sp(i,k) = mul_rn(spin[k], inverse_magnitude);
  }
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixNVESpinKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixNVESpinKokkos<LMPHostType>;
#endif
}
