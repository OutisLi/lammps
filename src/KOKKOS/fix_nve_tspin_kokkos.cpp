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

#include "fix_nve_tspin_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "math_rounded_kokkos.h"

using namespace LAMMPS_NS;
using namespace MathRoundedKokkos;

/* ----------------------------------------------------------------------
   velocity-Verlet integration of the lattice and of the inertial spin
   dynamics on the device, step for step the update of FixNVETSpin

   The spin velocities and spin masses are the TSPIN state of fix
   property/atom/kk, whose dual views AtomKokkos keeps synchronized and
   whose device sort and exchange move them with the atoms.  Every
   multiplication, addition and division is rounded on its own
   (math_rounded_kokkos.h) and evaluated in the order of the host fix, so a
   double-precision device trajectory agrees with the host fix bit for bit
   where the host compiler keeps its multiplications and additions apart.
------------------------------------------------------------------------- */

template<class DeviceType>
FixNVETSpinKokkos<DeviceType>::FixNVETSpinKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixNVETSpin(lmp, narg, arg), rmass_flag(0)
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
    datamask_read |= SP_MASK | FM_MASK | DVECTOR_MASK | DARRAY_MASK;
    datamask_modify |= SP_MASK | DARRAY_MASK;
    datamask_final |= DARRAY_MASK;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVETSpinKokkos<DeviceType>::init()
{
  FixNVETSpin::init();

  atomKK->k_mass.modify_host();
  atomKK->k_mass.sync<DeviceType>();
}

/* ----------------------------------------------------------------------
   the spin masses are assigned and checked on the host
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNVETSpinKokkos<DeviceType>::setup(int vflag)
{
  atomKK->sync(Host, SP_MASK | MASK_MASK | TYPE_MASK | RMASS_MASK | DVECTOR_MASK);
  FixNVETSpin::setup(vflag);
  if (spinmass_flag) atomKK->modified(Host, DVECTOR_MASK);
}

/* ----------------------------------------------------------------------
   half-step kicks of the atomic and spin velocities, each followed by the
   full-step drift of the positions and spin vectors
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNVETSpinKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  atomKK->sync(execution_space, datamask_read);
  set_views();

  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVETSpinInitial>(0, nlocal), *this);
  copymode = 0;

  atomKK->modified(execution_space, datamask_modify);
}

/* ----------------------------------------------------------------------
   half-step kicks of the atomic and spin velocities
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNVETSpinKokkos<DeviceType>::final_integrate()
{
  atomKK->sync(execution_space, datamask_read);
  set_views();

  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVETSpinFinal>(0, nlocal), *this);
  copymode = 0;

  atomKK->modified(execution_space, datamask_final);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVETSpinKokkos<DeviceType>::set_views()
{
  rmass_flag = atomKK->rmass_flag;
  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  sp = atomKK->k_sp.view<DeviceType>();
  fm = atomKK->k_fm.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  if (spin_flag) {
    s_dot = atomKK->k_darray.view_host()[index_vs].k_view.view<DeviceType>();
    dvector = atomKK->k_dvector.view<DeviceType>();
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNVETSpinKokkos<DeviceType>::operator()(TagFixNVETSpinInitial, const int &i) const
{
  if (!(mask[i] & groupbit)) return;
  if (lattice_flag) update_lattice_item(i, true);
  if (spin_flag && !(dvector(index_sm,i) <= 0.0)) update_spin_item(i, true);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNVETSpinKokkos<DeviceType>::operator()(TagFixNVETSpinFinal, const int &i) const
{
  if (!(mask[i] & groupbit)) return;
  if (lattice_flag) update_lattice_item(i, false);
  if (spin_flag && !(dvector(index_sm,i) <= 0.0)) update_spin_item(i, false);
}

/* ----------------------------------------------------------------------
   v += dtf/m f, followed by x += dtv v when drift is set
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNVETSpinKokkos<DeviceType>::update_lattice_item(int i, bool drift) const
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
   dS/dt += dtf hbar/(m_s |S|) fm, skipped at exactly zero modulus where the
   fm encoding carries no force, followed by S += dtv dS/dt when drift is
   set; the drifted vector S is split back into the direction, stored in
   sp[0..2], and the modulus |S|, stored in sp[3]. The spin velocities are
   stored and kicked in double precision, as on the host, in every KOKKOS
   precision.
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNVETSpinKokkos<DeviceType>::update_spin_item(int i, bool drift) const
{
  const KK_FLOAT modulus = sp(i,3);
  if (modulus != 0.0) {
    const double scale = div_rn(mul_rn(dtf, hbar),
                                mul_rn(static_cast<double>(dvector(index_sm,i)),
                                       static_cast<double>(modulus)));
    for (int k = 0; k < 3; k++)
      s_dot(i,k) = add_rn(s_dot(i,k), mul_rn(scale, static_cast<double>(fm(i,k))));
  }
  if (!drift) return;

  const KK_FLOAT dtv_kk = static_cast<KK_FLOAT>(dtv);
  KK_FLOAT s[3];
  for (int k = 0; k < 3; k++)
    s[k] = add_rn(mul_rn(modulus, sp(i,k)), mul_rn(dtv_kk, static_cast<KK_FLOAT>(s_dot(i,k))));
  const KK_FLOAT smag =
    Kokkos::sqrt(add_rn(add_rn(mul_rn(s[0], s[0]), mul_rn(s[1], s[1])), mul_rn(s[2], s[2])));

  sp(i,3) = smag;
  if (smag > 0.0)
    for (int k = 0; k < 3; k++) sp(i,k) = div_rn(s[k], smag);
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixNVETSpinKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixNVETSpinKokkos<LMPHostType>;
#endif
}
