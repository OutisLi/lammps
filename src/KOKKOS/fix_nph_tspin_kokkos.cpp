// clang-format off
/* ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_nph_tspin_kokkos.h"

#include "error.h"
#include "group.h"
#include "modify.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixNPHTSpinKokkos<DeviceType>::FixNPHTSpinKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixNHTSpinKokkos<DeviceType>(lmp, narg, arg)
{
  if (this->tstat_flag)
    this->error->all(FLERR, "Temperature control can not be used with fix {}", this->style);
  if (!this->pstat_flag)
    this->error->all(FLERR, "Pressure control must be used with fix {}", this->style);

  // create a new compute temp style
  // id = fix-ID + temp
  // compute group = all since pressure is always global (group all)
  // and thus its KE/temperature contribution should use group all

  this->id_temp = utils::strdup(std::string(this->id) + "_temp");
  this->modify->add_compute(fmt::format("{} all temp/kk", this->id_temp));
  this->tcomputeflag = 1;

  // create a new compute pressure style
  // id = fix-ID + press, compute group = all
  // pass id_temp as 4th arg to pressure constructor

  this->id_press = utils::strdup(std::string(this->id) + "_press");
  this->modify->add_compute(fmt::format("{} all pressure {}", this->id_press, this->id_temp));
  this->pcomputeflag = 1;
}

namespace LAMMPS_NS {
template class FixNPHTSpinKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixNPHTSpinKokkos<LMPHostType>;
#endif
}
