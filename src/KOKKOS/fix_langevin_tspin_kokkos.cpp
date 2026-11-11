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

#include "fix_langevin_tspin_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "force.h"
#include "math_const.h"

#include <cmath>

using namespace LAMMPS_NS;
using MathConst::MY_2PI;

/* ----------------------------------------------------------------------
   inertial Langevin bath of the spin vectors on the device, the friction
   and noise of FixLangevinTSpin added to the magnetic force in the fm
   encoding of the tspin styles

   The noise comes from a device random-number pool seeded per rank, as in
   fix langevin/kk, so the device and the host fix sample the same bath with
   different random sequences.
------------------------------------------------------------------------- */

template<class DeviceType>
FixLangevinTSpinKokkos<DeviceType>::FixLangevinTSpinKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixLangevinTSpin(lmp, narg, arg), gamma1_scale(0.0), gamma2_scale(0.0), inv_hbar(0.0),
  fmean{0.0, 0.0, 0.0}, rand_pool(utils::inumeric(FLERR, arg[6], false, lmp) + comm->me)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = SP_MASK | FM_MASK | MASK_MASK | DVECTOR_MASK | DARRAY_MASK;
  datamask_modify = FM_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixLangevinTSpinKokkos<DeviceType>::post_force(int /*vflag*/)
{
  compute_target();

  atomKK->sync(execution_space, datamask_read);
  sp = atomKK->k_sp.view<DeviceType>();
  fm = atomKK->k_fm.view<DeviceType>();
  s_dot = atomKK->k_darray.view_host()[index_vs].k_view.view<DeviceType>();
  dvector = atomKK->k_dvector.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  gamma1_scale = -gamma_drag;
  gamma2_scale = gamma_random * sqrt(t_target);
  inv_hbar = MY_2PI / force->hplanck;
  const int nlocal = atomKK->nlocal;

  copymode = 1;

  s_TSpinNoiseSum sum;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagFixLangevinTSpinBath>(0, nlocal),
                          *this, sum);

  // optionally remove the net random force on the spin system

  if (zeroflag) {
    s_TSpinNoiseSum sumall;
    MPI_Allreduce(sum.f, sumall.f, 3, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&sum.count, &sumall.count, 1, MPI_INT, MPI_SUM, world);
    if (sumall.count > 0) {
      for (int k = 0; k < 3; k++) fmean[k] = sumall.f[k] / sumall.count;
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixLangevinTSpinZero>(0, nlocal),
                           *this);
    }
  }

  copymode = 0;

  atomKK->modified(execution_space, datamask_modify);
}

/* ----------------------------------------------------------------------
   friction -m_s/t_period v_s plus uniform noise of the variance of the
   host fix, converted to the fm encoding by |S|/hbar
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixLangevinTSpinKokkos<DeviceType>::operator()(TagFixLangevinTSpinBath, const int &i,
                                                    s_TSpinNoiseSum &sum) const
{
  if (!(mask[i] & groupbit)) return;
  const double s_mass = dvector(index_sm, i);
  if (s_mass <= 0.0) return;

  const double gamma1 = s_mass * gamma1_scale;
  const double gamma2 = Kokkos::sqrt(s_mass) * gamma2_scale;

  rand_type rand_gen = rand_pool.get_state();
  double fran[3];
  for (int k = 0; k < 3; k++) fran[k] = gamma2 * (rand_gen.drand() - 0.5);
  rand_pool.free_state(rand_gen);

  const double tofm = sp(i, 3) * inv_hbar;
  for (int k = 0; k < 3; k++) {
    fm(i, k) += tofm * (gamma1 * s_dot(i, k) + fran[k]);
    sum.f[k] += fran[k];
  }
  sum.count++;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixLangevinTSpinKokkos<DeviceType>::operator()(TagFixLangevinTSpinZero, const int &i) const
{
  if (!(mask[i] & groupbit)) return;
  if (dvector(index_sm, i) <= 0.0) return;
  const double tofm = sp(i, 3) * inv_hbar;
  for (int k = 0; k < 3; k++) fm(i, k) -= tofm * fmean[k];
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixLangevinTSpinKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixLangevinTSpinKokkos<LMPHostType>;
#endif
}
