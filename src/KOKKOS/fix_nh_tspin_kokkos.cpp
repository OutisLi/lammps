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

#include "fix_nh_tspin_kokkos.h"
#include "fix_nh_tspin_impl.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "force.h"
#include "math_rounded_kokkos.h"

using namespace LAMMPS_NS;
using namespace MathRoundedKokkos;

/* ----------------------------------------------------------------------
   Nose-Hoover integration of the lattice and of the inertial spin dynamics
   on the device, step for step the update of FixNHTSpin

   The lattice follows fix nh/kk with its host thermostat chain; the spin
   chain of FixNHTSpinT is propagated on the host from a device reduction of
   the spin kinetic energy and scales the spin velocities on the device. The
   spin velocities and spin masses are the TSPIN state of fix
   property/atom/kk. The kick and the drift of the spins are rounded
   operation by operation as in fix nve/tspin/kk.
------------------------------------------------------------------------- */

template<class DeviceType>
FixNHTSpinKokkos<DeviceType>::FixNHTSpinKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixNHTSpinT<FixNHKokkos<DeviceType>>(lmp, narg, arg), spin_factor(1.0)
{
}

/* ----------------------------------------------------------------------
   the spin masses are assigned and the spin degrees of freedom counted on
   the host
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHTSpinKokkos<DeviceType>::setup(int vflag)
{
  this->atomKK->sync(Host, SP_MASK | MASK_MASK | TYPE_MASK | RMASS_MASK | DVECTOR_MASK);
  FixNHTSpinT<FixNHKokkos<DeviceType>>::setup(vflag);
  if (this->spinmass_flag) this->atomKK->modified(Host, DVECTOR_MASK);
}

/* ----------------------------------------------------------------------
   synchronize the spin arrays named by mask in the execution space, bind
   their views and return the number of atoms the kernels visit
------------------------------------------------------------------------- */

template<class DeviceType>
int FixNHTSpinKokkos<DeviceType>::set_spin_views(uint64_t datamask)
{
  this->atomKK->sync(this->execution_space, datamask);
  sp = this->atomKK->k_sp.template view<DeviceType>();
  fm = this->atomKK->k_fm.template view<DeviceType>();
  s_dot = this->atomKK->k_darray.view_host()[this->index_vs].k_view.template view<DeviceType>();
  dvector = this->atomKK->k_dvector.template view<DeviceType>();
  this->mask = this->atomKK->k_mask.template view<DeviceType>();

  int nlocal = this->atomKK->nlocal;
  if (this->igroup == this->atomKK->firstgroup) nlocal = this->atomKK->nfirst;
  return nlocal;
}

/* ----------------------------------------------------------------------
   instantaneous spin temperature from the spin kinetic energy, reduced on
   the device
------------------------------------------------------------------------- */

template<class DeviceType>
double FixNHTSpinKokkos<DeviceType>::compute_spin_temp()
{
  const int nlocal = set_spin_views(MASK_MASK | DVECTOR_MASK | DARRAY_MASK);
  double ke = 0.0;
  this->copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagFixNHTSpinKinetic>(0, nlocal),
                          *this, ke);
  this->copymode = 0;

  double keall = 0.0;
  MPI_Allreduce(&ke, &keall, 1, MPI_DOUBLE, MPI_SUM, this->world);

  if (this->tdof_spin == 0.0) return 0.0;
  return keall * this->force->mvv2e / (this->tdof_spin * this->boltz);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNHTSpinKokkos<DeviceType>::nh_vs_scale(double factor)
{
  const int nlocal = set_spin_views(MASK_MASK | DVECTOR_MASK | DARRAY_MASK);
  spin_factor = factor;
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNHTSpinScale>(0, nlocal), *this);
  this->copymode = 0;
  this->atomKK->modified(this->execution_space, DARRAY_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNHTSpinKokkos<DeviceType>::spin_kick()
{
  const int nlocal = set_spin_views(SP_MASK | FM_MASK | MASK_MASK | DVECTOR_MASK | DARRAY_MASK);
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNHTSpinKick>(0, nlocal), *this);
  this->copymode = 0;
  this->atomKK->modified(this->execution_space, DARRAY_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNHTSpinKokkos<DeviceType>::spin_drift()
{
  const int nlocal = set_spin_views(SP_MASK | MASK_MASK | DVECTOR_MASK | DARRAY_MASK);
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNHTSpinDrift>(0, nlocal), *this);
  this->copymode = 0;
  this->atomKK->modified(this->execution_space, SP_MASK);
}

/* ----------------------------------------------------------------------
   dS/dt += dtf hbar/(m_s |S|) fm, skipped at exactly zero modulus where the
   fm encoding carries no force; the spin velocities are stored and kicked in
   double precision, as on the host, in every KOKKOS precision
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNHTSpinKokkos<DeviceType>::operator()(TagFixNHTSpinKick, const int &i) const
{
  if (!(this->mask[i] & this->groupbit)) return;
  const KK_FLOAT s_mass = dvector(this->index_sm, i);
  if (s_mass <= 0.0) return;
  const KK_FLOAT modulus = sp(i, 3);
  if (modulus == 0.0) return;
  const double scale = div_rn(mul_rn(this->dtf, this->hbar),
                              mul_rn(static_cast<double>(s_mass), static_cast<double>(modulus)));
  for (int k = 0; k < 3; k++)
    s_dot(i, k) = add_rn(s_dot(i, k), mul_rn(scale, static_cast<double>(fm(i, k))));
}

/* ----------------------------------------------------------------------
   S += dtv dS/dt, split back into the direction sp[0..2] and the modulus
   sp[3]
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNHTSpinKokkos<DeviceType>::operator()(TagFixNHTSpinDrift, const int &i) const
{
  if (!(this->mask[i] & this->groupbit)) return;
  if (dvector(this->index_sm, i) <= 0.0) return;

  const KK_FLOAT dtv_kk = static_cast<KK_FLOAT>(this->dtv);
  const KK_FLOAT modulus = sp(i, 3);
  KK_FLOAT s[3];
  for (int k = 0; k < 3; k++)
    s[k] = add_rn(mul_rn(modulus, sp(i, k)), mul_rn(dtv_kk, static_cast<KK_FLOAT>(s_dot(i, k))));
  const KK_FLOAT smag =
    Kokkos::sqrt(add_rn(add_rn(mul_rn(s[0], s[0]), mul_rn(s[1], s[1])), mul_rn(s[2], s[2])));

  sp(i, 3) = smag;
  if (smag > 0.0)
    for (int k = 0; k < 3; k++) sp(i, k) = div_rn(s[k], smag);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNHTSpinKokkos<DeviceType>::operator()(TagFixNHTSpinScale, const int &i) const
{
  if (!(this->mask[i] & this->groupbit)) return;
  if (dvector(this->index_sm, i) <= 0.0) return;
  for (int k = 0; k < 3; k++) s_dot(i, k) *= spin_factor;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNHTSpinKokkos<DeviceType>::operator()(TagFixNHTSpinKinetic, const int &i,
                                              double &ke) const
{
  if (!(this->mask[i] & this->groupbit)) return;
  const double s_mass = dvector(this->index_sm, i);
  if (s_mass <= 0.0) return;
  ke += s_mass * (s_dot(i, 0) * s_dot(i, 0) + s_dot(i, 1) * s_dot(i, 1) + s_dot(i, 2) * s_dot(i, 2));
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixNHTSpinT<FixNHKokkos<LMPDeviceType>>;
template class FixNHTSpinKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixNHTSpinT<FixNHKokkos<LMPHostType>>;
template class FixNHTSpinKokkos<LMPHostType>;
#endif
}
