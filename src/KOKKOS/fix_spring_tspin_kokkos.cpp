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

#include "fix_spring_tspin_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "force.h"
#include "math_const.h"
#include "math_rounded_kokkos.h"
#include "tspin.h"

using namespace LAMMPS_NS;
using namespace MathRoundedKokkos;
using MathConst::MY_2PI;

/* ----------------------------------------------------------------------
   harmonic restoring force on the spin moduli on the device, the force of
   FixSpringTSpin added to the magnetic force in the fm encoding of the
   tspin styles, with the spring energy reduced on the device

   The force is rounded operation by operation in the order of the host
   fix (math_rounded_kokkos.h), so it agrees with the host fix bit for bit
   where the host compiler keeps its multiplications and additions apart.
------------------------------------------------------------------------- */

template<class DeviceType>
FixSpringTSpinKokkos<DeviceType>::FixSpringTSpinKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixSpringTSpin(lmp, narg, arg), dynamic_flag(0), hbar(0.0)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = SP_MASK | FM_MASK | MASK_MASK | DVECTOR_MASK;
  datamask_modify = FM_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixSpringTSpinKokkos<DeviceType>::post_force(int /*vflag*/)
{
  atomKK->sync(execution_space, datamask_read);
  sp = atomKK->k_sp.view<DeviceType>();
  fm = atomKK->k_fm.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  dynamic_flag = index_sm >= 0;
  if (dynamic_flag) dvector = atomKK->k_dvector.view<DeviceType>();
  hbar = force->hplanck / MY_2PI;

  copymode = 1;
  espring = 0.0;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagFixSpringTSpin>(0, atomKK->nlocal),
                          *this, espring);
  copymode = 0;

  atomKK->modified(execution_space, datamask_modify);
}

/* ----------------------------------------------------------------------
   F = -k (|S|-S0) s^, stored scaled by |S|/hbar; a dynamical spin keeps its
   energy at zero modulus, where the radial direction is undefined
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixSpringTSpinKokkos<DeviceType>::operator()(TagFixSpringTSpin, const int &i,
                                                  double &energy) const
{
  if (!(mask[i] & groupbit)) return;

  const double smag = sp(i, 3);
  const bool dynamic = dynamic_flag && (dvector(index_sm, i) > 0.0);
  if ((smag <= TSPIN_EPS) && !dynamic) return;

  const double dmod = smag - s0;
  energy += 0.5 * k * dmod * dmod;
  if (smag == 0.0) return;

  const double pref = div_rn(mul_rn(mul_rn(-k, dmod), smag), hbar);
  for (int d = 0; d < 3; d++)
    fm(i, d) = add_rn(static_cast<double>(fm(i, d)), mul_rn(pref, static_cast<double>(sp(i, d))));
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixSpringTSpinKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixSpringTSpinKokkos<LMPHostType>;
#endif
}
