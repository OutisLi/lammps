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

#include "fix_property_atom_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "memory_kokkos.h"

#include <cstring>
#include <type_traits>

using namespace LAMMPS_NS;
using namespace FixConst;

namespace {

// value of atom i in column k of a per-atom vector (rank 1) or array (rank 2)

template <class ViewType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION decltype(auto) custom_value(const ViewType &values, int i, int k)
{
  if constexpr (ViewType::rank == 1) return values(i);
  else return values(i, k);
}

// integer values travel bit for bit in the double buffer, as in the host
// pack_exchange() of FixPropertyAtom

template <class T>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION double to_buffer(T value)
{
  if constexpr (std::is_integral_v<T>) return d_ubuf(value).d;
  else return static_cast<double>(value);
}

template <class T>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION T from_buffer(double value)
{
  if constexpr (std::is_integral_v<T>) return static_cast<T>(d_ubuf(value).i);
  else return static_cast<T>(value);
}

/* ----------------------------------------------------------------------
   exchange buffer layout: slot m holds the offset of the record of the m-th
   sent atom, and the records of values_peratom values follow the nsend slots
------------------------------------------------------------------------- */

template <class DeviceType>
struct PropertyAtomExchangeOffsets {
  typedef DeviceType device_type;
  typename ArrayTypes<DeviceType>::t_double_1d_um buf;
  int nsend, width;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION void operator()(const int m) const { buf(m) = nsend + m * width; }
};

// pack the values of the sent atom, then refill its slot from the atom of
// the copy list: each sent slot is written by one thread only, and the atoms
// of the copy list are not sent, so no thread reads a slot another one writes

template <class DeviceType, class ViewType>
struct PropertyAtomPackExchange {
  typedef DeviceType device_type;
  typename ArrayTypes<DeviceType>::t_double_1d_um buf;
  typename ArrayTypes<DeviceType>::t_int_1d sendlist, copylist;
  ViewType values;
  int nsend, width, offset, ncol;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION void operator()(const int m) const
  {
    const int i = sendlist(m);
    const int first = nsend + m * width + offset;
    for (int k = 0; k < ncol; k++) buf(first + k) = to_buffer(custom_value(values, i, k));
    const int j = copylist(m);
    if (j > -1)
      for (int k = 0; k < ncol; k++) custom_value(values, i, k) = custom_value(values, j, k);
  }
};

// the records of the first neighbor come first; the slots and offsets of the
// second neighbor are relative to the start of its part of the buffer

template <class DeviceType, class ViewType>
struct PropertyAtomUnpackExchange {
  typedef DeviceType device_type;
  typename ArrayTypes<DeviceType>::t_double_1d_um buf;
  typename ArrayTypes<DeviceType>::t_int_1d indices;
  ViewType values;
  int nrecv1, nextrarecv1, offset, ncol;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const
  {
    const int index = indices(i);
    if (index < 0) return;
    int m = static_cast<int>(buf(i));
    if (i >= nrecv1) m = nextrarecv1 + static_cast<int>(buf(nextrarecv1 + i - nrecv1));
    m += offset;
    using value_type = std::remove_cv_t<std::remove_reference_t<decltype(custom_value(values, 0, 0))>>;
    for (int k = 0; k < ncol; k++) custom_value(values, index, k) = from_buffer<value_type>(buf(m + k));
  }
};

}    // namespace

/* ---------------------------------------------------------------------- */

FixPropertyAtomKokkos::FixPropertyAtomKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixPropertyAtom(lmp, narg, arg)
{
  atomKK = (AtomKokkos *) atom;
  kokkosable = 1;

  // this fix syncs the atom data it owns itself, through AtomKokkos::sync()
  // and AtomKokkos::modified(); Modify must not sync or invalidate any of it

  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  dvector_flag = 0;
  ivector_flag = 0;
  iarray_flag = 0;
  darray_flag = 0;
  for (int nv = 0; nv < nvalue; nv++) {
    if (styles[nv] == IVEC) ivector_flag = 1;
    if (styles[nv] == DVEC) dvector_flag = 1;
    if (styles[nv] == IARRAY) iarray_flag = 1;
    if (styles[nv] == DARRAY) darray_flag = 1;
  }
  custom_datamask = (ivector_flag ? IVECTOR_MASK : EMPTY_MASK) |
      (dvector_flag ? DVECTOR_MASK : EMPTY_MASK) | (iarray_flag ? IARRAY_MASK : EMPTY_MASK) |
      (darray_flag ? DARRAY_MASK : EMPTY_MASK);

  // the custom vectors and arrays live in dual views, which the device sort
  // and exchange permute and move in place; molecule IDs, charges and masses
  // belong to the atom style and the other values are host arrays, so a fix
  // holding any of them keeps the host sort and exchange, as does a fix with
  // ghost values, whose border communication runs on the host

  int custom_only = !border;
  for (int nv = 0; nv < nvalue; nv++)
    if ((styles[nv] != IVEC) && (styles[nv] != DVEC) && (styles[nv] != IARRAY) &&
        (styles[nv] != DARRAY))
      custom_only = 0;
  sort_device = exchange_comm_device = custom_only;

  // an exchanged atom takes its offset slot and its values in the buffer

  if (custom_only) maxexchange = values_peratom + 1;
}

/* ---------------------------------------------------------------------- */

void FixPropertyAtomKokkos::post_constructor()
{
  atomKK->update_property_atom();

  FixPropertyAtom::post_constructor();
}

/* ----------------------------------------------------------------------
   the commands between runs, such as read_restart, set or velocity/tspin,
   write the custom values on the host, so at the start of a run the host
   copy is the current one and the next device access copies it over
------------------------------------------------------------------------- */

void FixPropertyAtomKokkos::init()
{
  FixPropertyAtom::init();
  sync(Host, custom_datamask);
  modified(Host, custom_datamask);
}

/* ---------------------------------------------------------------------- */

FixPropertyAtomKokkos::~FixPropertyAtomKokkos()
{
  // deallocate per-atom vectors in Atom class
  // set ptrs to a null pointer, so they no longer exist for Atom class

  for (int nv = 0; nv < nvalue; nv++) {
    if (styles[nv] == MOLECULE) {
      atom->molecule_flag = 0;
      memoryKK->destroy_kokkos(atomKK->k_molecule,atom->molecule);
      atom->molecule = nullptr;
    } else if (styles[nv] == CHARGE) {
      atom->q_flag = 0;
      memoryKK->destroy_kokkos(atomKK->k_q,atom->q);
      atom->q = nullptr;
    } else if (styles[nv] == RMASS) {
      atom->rmass_flag = 0;
      memoryKK->destroy_kokkos(atomKK->k_rmass,atom->rmass);
      atom->rmass = nullptr;
    }
  }

  // drop this fix from the list of property/atom fixes in AtomKokkos

  int n = 0;
  for (int i = 0; i < atomKK->nprop_atom; i++)
    if (atomKK->fix_prop_atom[i] != this)
      atomKK->fix_prop_atom[n++] = atomKK->fix_prop_atom[i];
  atomKK->nprop_atom = n;
}

/* ----------------------------------------------------------------------
   allocate atom-based arrays
   initialize new values to 0,
   since AtomVec class won't do it as atoms are added,
   e.g. in create_atom() or data_atom()
------------------------------------------------------------------------- */

void FixPropertyAtomKokkos::grow_arrays(int nmax)
{
  // ivector is one shared contiguous 2D view holding all custom int vectors,
  // so grow it once here rather than re-growing the whole view once per IVEC
  // value inside the loop below
  if (ivector_flag) {
    atomKK->sync(Device,IVECTOR_MASK);
    atomKK->modified(Device,IVECTOR_MASK);
    memoryKK->grow_kokkos(atomKK->k_ivector,atom->ivector,atomKK->k_ivector.extent(0),nmax,
                        "atom:ivector");
    atomKK->sync(Host,IVECTOR_MASK);
  }

  for (int nv = 0; nv < nvalue; nv++) {
    if (styles[nv] == MOLECULE) {
      atomKK->sync(Device,MOLECULE_MASK);
      atomKK->modified(Device,MOLECULE_MASK);
      memoryKK->grow_kokkos(atomKK->k_molecule,atom->molecule,nmax,"atom:molecule");
      atomKK->sync(Host,MOLECULE_MASK);
    } else if (styles[nv] == CHARGE) {
      atomKK->sync(Device,Q_MASK);
      atomKK->modified(Device,Q_MASK);
      memoryKK->grow_kokkos(atomKK->k_q,atom->q,nmax,"atom:q");
      atomKK->sync(Host,Q_MASK);
    } else if (styles[nv] == RMASS) {
      atomKK->sync(Device,RMASS_MASK);
      atomKK->modified(Device,RMASS_MASK);
      memoryKK->grow_kokkos(atomKK->k_rmass,atom->rmass,nmax,"atom:rmass");
      atomKK->sync(Host,RMASS_MASK);
    } else if (styles[nv] == TEMPERATURE) {
      memory->grow(atom->temperature, nmax, "atom:temperature");
      size_t nbytes = (nmax - nmax_old) * sizeof(double);
      memset(&atom->temperature[nmax_old], 0, nbytes);
    } else if (styles[nv] == HEATFLOW) {
      memory->grow(atom->heatflow, nmax, "atom:heatflow");
      size_t nbytes = (nmax - nmax_old) * sizeof(double);
      memset(&atom->heatflow[nmax_old], 0, nbytes);
    } else if (styles[nv] == IVEC) {
      // storage was grown once above (shared contiguous k_ivector); just zero
      // this vector's newly added tail
      size_t nbytes = (nmax-nmax_old) * sizeof(int);
      memset(&atom->ivector[index[nv]][nmax_old],0,nbytes);
    } else if (styles[nv] == DVEC) {
      atomKK->sync(Device,DVECTOR_MASK);
      atomKK->modified(Device,DVECTOR_MASK);
      memoryKK->grow_kokkos(atomKK->k_dvector,atom->dvector,atomKK->k_dvector.extent(0),nmax,
                          "atom:dvector");
      atomKK->sync(Host,DVECTOR_MASK);
    } else if (styles[nv] == IARRAY) {
      // ragged cols: grow this property's inner DualView in the view-of-views,
      // using the same device/host sync handshake as the DVEC branch so a grow
      // that follows a device-side modification does not lose data
      int idx = index[nv];
      auto& inner = atomKK->k_iarray.view_host()[idx].k_view;
      inner.sync_device();
      inner.modify_device();
      memoryKK->grow_kokkos(inner, atom->iarray[idx], nmax, cols[nv], "atom:iarray");
      inner.sync_host();
      size_t nbytes = (size_t) (nmax - nmax_old) * cols[nv] * sizeof(int);
      if (nbytes) {
        memset(&atom->iarray[idx][nmax_old][0], 0, nbytes);
        inner.modify_host();   // the memset wrote host; mark it so it reaches device
      }
      // re-sync the outer struct array: growing the inner view changed its
      // device pointer, which the device-side view-of-views must see
      atomKK->k_iarray.modify_host();
      atomKK->k_iarray.sync_device();
    } else if (styles[nv] == DARRAY) {
      int idx = index[nv];
      auto& inner = atomKK->k_darray.view_host()[idx].k_view;
      inner.sync_device();
      inner.modify_device();
      memoryKK->grow_kokkos(inner, atom->darray[idx], nmax, cols[nv], "atom:darray");
      inner.sync_host();
      size_t nbytes = (size_t) (nmax - nmax_old) * cols[nv] * sizeof(double);
      if (nbytes) {
        memset(&atom->darray[idx][nmax_old][0], 0, nbytes);
        inner.modify_host();
      }
      atomKK->k_darray.modify_host();
      atomKK->k_darray.sync_device();
    }
  }
  nmax_old = nmax;
}

/* ----------------------------------------------------------------------
   the host callbacks of the legacy sort and exchange read and write the
   host copies of the values of this fix, which device kernels may have
   changed: the legacy exchange synchronizes only the arrays of the atom style
------------------------------------------------------------------------- */

static constexpr uint64_t PROPERTY_ATOM_MASK = MOLECULE_MASK | Q_MASK | RMASS_MASK |
    IVECTOR_MASK | DVECTOR_MASK | IARRAY_MASK | DARRAY_MASK;

void FixPropertyAtomKokkos::copy_arrays(int i, int j, int delflag)
{
  sync(Host, PROPERTY_ATOM_MASK);
  FixPropertyAtom::copy_arrays(i, j, delflag);
  modified(Host, PROPERTY_ATOM_MASK);
}

/* ---------------------------------------------------------------------- */

int FixPropertyAtomKokkos::pack_exchange(int i, double *buf)
{
  sync(Host, PROPERTY_ATOM_MASK);
  return FixPropertyAtom::pack_exchange(i, buf);
}

/* ---------------------------------------------------------------------- */

int FixPropertyAtomKokkos::unpack_exchange(int nlocal, double *buf)
{
  sync(Host, PROPERTY_ATOM_MASK);
  const int m = FixPropertyAtom::unpack_exchange(nlocal, buf);
  modified(Host, PROPERTY_ATOM_MASK);
  return m;
}

/* ---------------------------------------------------------------------- */

void FixPropertyAtomKokkos::sync(ExecutionSpace space, uint64_t mask)
{
  if (space == Device) {
    if (molecule_flag && (mask & MOLECULE_MASK)) atomKK->k_molecule.sync_device();
    if (q_flag && (mask & Q_MASK)) atomKK->k_q.sync_device();
    if (rmass_flag && (mask & RMASS_MASK)) {atomKK->k_rmass.sync_device();}
    if (dvector_flag && (mask & DVECTOR_MASK)) atomKK->k_dvector.sync_device();
    if (ivector_flag && (mask & IVECTOR_MASK)) atomKK->k_ivector.sync_device();
    if (iarray_flag && (mask & IARRAY_MASK))
      for (int i = 0; i < atom->niarray; i++)
        atomKK->k_iarray.view_host()[i].k_view.sync_device();
    if (darray_flag && (mask & DARRAY_MASK))
      for (int i = 0; i < atom->ndarray; i++)
        atomKK->k_darray.view_host()[i].k_view.sync_device();
  } else if (space == Host) {
    if (molecule_flag && (mask & MOLECULE_MASK)) atomKK->k_molecule.sync_host();
    if (q_flag && (mask & Q_MASK)) atomKK->k_q.sync_host();
    if (rmass_flag && (mask & RMASS_MASK)) atomKK->k_rmass.sync_host();
    if (dvector_flag && (mask & DVECTOR_MASK)) atomKK->k_dvector.sync_host();
    if (ivector_flag && (mask & IVECTOR_MASK)) atomKK->k_ivector.sync_host();
    if (iarray_flag && (mask & IARRAY_MASK))
      for (int i = 0; i < atom->niarray; i++)
        atomKK->k_iarray.view_host()[i].k_view.sync_host();
    if (darray_flag && (mask & DARRAY_MASK))
      for (int i = 0; i < atom->ndarray; i++)
        atomKK->k_darray.view_host()[i].k_view.sync_host();
  } else if (space == HostKK) {
    if (molecule_flag && (mask & MOLECULE_MASK)) atomKK->k_molecule.sync_host();
    if (q_flag && (mask & Q_MASK)) atomKK->k_q.sync_hostkk();
    if (rmass_flag && (mask & RMASS_MASK)) atomKK->k_rmass.sync_hostkk();
    if (dvector_flag && (mask & DVECTOR_MASK)) atomKK->k_dvector.sync_hostkk();
    if (ivector_flag && (mask & IVECTOR_MASK)) atomKK->k_ivector.sync_host();
    if (iarray_flag && (mask & IARRAY_MASK))
      for (int i = 0; i < atom->niarray; i++)
        atomKK->k_iarray.view_host()[i].k_view.sync_host();
    if (darray_flag && (mask & DARRAY_MASK))
      for (int i = 0; i < atom->ndarray; i++)
        atomKK->k_darray.view_host()[i].k_view.sync_host();
  }
}

/* ---------------------------------------------------------------------- */

void FixPropertyAtomKokkos::sync_pinned(ExecutionSpace space, uint64_t mask, int async_flag)
{
  if (space == Device) {
    if ((mask & MOLECULE_MASK) && atomKK->k_molecule.need_sync_device())
      atomKK->avecKK->perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_molecule,space,async_flag);
    if ((mask & Q_MASK) && atomKK->k_q.need_sync_device())
      atomKK->avecKK->perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d>(atomKK->k_q,space,async_flag);
    if ((mask & RMASS_MASK) && atomKK->k_rmass.need_sync_device())
      atomKK->avecKK->perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d>(atomKK->k_rmass,space,async_flag);
    if ((mask & DVECTOR_MASK) && atomKK->k_dvector.need_sync_device())
      atomKK->avecKK->perform_pinned_copy_transform<DAT::ttransform_kkfloat_2d>(atomKK->k_dvector,space,async_flag);
    if ((mask & IVECTOR_MASK) && atomKK->k_ivector.need_sync_device())
      atomKK->avecKK->perform_pinned_copy<DAT::tdual_int_2d_lr>(atomKK->k_ivector,space,async_flag);
    if (mask & IARRAY_MASK)
      for (int i = 0; i < atom->niarray; i++)
        if (atomKK->k_iarray.view_host()[i].k_view.need_sync_device())
          atomKK->avecKK->perform_pinned_copy<DAT::tdual_int_2d_lr>(
              atomKK->k_iarray.view_host()[i].k_view, space, async_flag);
    if (mask & DARRAY_MASK)
      for (int i = 0; i < atom->ndarray; i++)
        if (atomKK->k_darray.view_host()[i].k_view.need_sync_device())
          atomKK->avecKK->perform_pinned_copy<DAT::tdual_double_2d_lr>(
              atomKK->k_darray.view_host()[i].k_view, space, async_flag);
  } else {
    if ((mask & MOLECULE_MASK) && atomKK->k_molecule.need_sync_host())
      atomKK->avecKK->perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_molecule,space,async_flag);
    if ((mask & Q_MASK) && atomKK->k_q.need_sync_host())
      atomKK->avecKK->perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d>(atomKK->k_q,space,async_flag);
    if ((mask & RMASS_MASK) && atomKK->k_rmass.need_sync_host())
      atomKK->avecKK->perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d>(atomKK->k_rmass,space,async_flag);
    if ((mask & DVECTOR_MASK) && atomKK->k_dvector.need_sync_host())
      atomKK->avecKK->perform_pinned_copy_transform<DAT::ttransform_kkfloat_2d>(atomKK->k_dvector,space,async_flag);
    if ((mask & IVECTOR_MASK) && atomKK->k_ivector.need_sync_host())
      atomKK->avecKK->perform_pinned_copy<DAT::tdual_int_2d_lr>(atomKK->k_ivector,space,async_flag);
    if (mask & IARRAY_MASK)
      for (int i = 0; i < atom->niarray; i++)
        if (atomKK->k_iarray.view_host()[i].k_view.need_sync_host())
          atomKK->avecKK->perform_pinned_copy<DAT::tdual_int_2d_lr>(
              atomKK->k_iarray.view_host()[i].k_view, space, async_flag);
    if (mask & DARRAY_MASK)
      for (int i = 0; i < atom->ndarray; i++)
        if (atomKK->k_darray.view_host()[i].k_view.need_sync_host())
          atomKK->avecKK->perform_pinned_copy<DAT::tdual_double_2d_lr>(
              atomKK->k_darray.view_host()[i].k_view, space, async_flag);
  }
}

/* ---------------------------------------------------------------------- */

void FixPropertyAtomKokkos::modified(ExecutionSpace space, uint64_t mask)
{
  if (space == Device) {
    if (molecule_flag && (mask & MOLECULE_MASK)) atomKK->k_molecule.modify_device();
    if (q_flag && (mask & Q_MASK)) atomKK->k_q.modify_device();
    if (rmass_flag && (mask & RMASS_MASK)) atomKK->k_rmass.modify_device();
    if (dvector_flag && (mask & DVECTOR_MASK)) atomKK->k_dvector.modify_device();
    if (ivector_flag && (mask & IVECTOR_MASK)) atomKK->k_ivector.modify_device();
    if (iarray_flag && (mask & IARRAY_MASK))
      for (int i = 0; i < atom->niarray; i++)
        atomKK->k_iarray.view_host()[i].k_view.modify_device();
    if (darray_flag && (mask & DARRAY_MASK))
      for (int i = 0; i < atom->ndarray; i++)
        atomKK->k_darray.view_host()[i].k_view.modify_device();
  } else if (space == Host) {
    if (molecule_flag && (mask & MOLECULE_MASK)) atomKK->k_molecule.modify_host();
    if (q_flag && (mask & Q_MASK)) atomKK->k_q.modify_host();
    if (rmass_flag && (mask & RMASS_MASK)) atomKK->k_rmass.modify_host();
    if (dvector_flag && (mask & DVECTOR_MASK)) atomKK->k_dvector.modify_host();
    if (ivector_flag && (mask & IVECTOR_MASK)) atomKK->k_ivector.modify_host();
    if (iarray_flag && (mask & IARRAY_MASK))
      for (int i = 0; i < atom->niarray; i++)
        atomKK->k_iarray.view_host()[i].k_view.modify_host();
    if (darray_flag && (mask & DARRAY_MASK))
      for (int i = 0; i < atom->ndarray; i++)
        atomKK->k_darray.view_host()[i].k_view.modify_host();
  } else if (space == HostKK) {
    if (molecule_flag && (mask & MOLECULE_MASK)) atomKK->k_molecule.modify_host();
    if (q_flag && (mask & Q_MASK)) atomKK->k_q.modify_hostkk();
    if (rmass_flag && (mask & RMASS_MASK)) atomKK->k_rmass.modify_hostkk();
    if (dvector_flag && (mask & DVECTOR_MASK)) atomKK->k_dvector.modify_hostkk();
    if (ivector_flag && (mask & IVECTOR_MASK)) atomKK->k_ivector.modify_host();
    if (iarray_flag && (mask & IARRAY_MASK))
      for (int i = 0; i < atom->niarray; i++)
        atomKK->k_iarray.view_host()[i].k_view.modify_host();
    if (darray_flag && (mask & DARRAY_MASK))
      for (int i = 0; i < atom->ndarray; i++)
        atomKK->k_darray.view_host()[i].k_view.modify_host();
  }
}

/* ----------------------------------------------------------------------
   call f(per-atom view, columns, offset in the exchange record) for every
   custom vector and array of this fix, with the view of the execution space
   of DeviceType

   The kernels below synchronize the values through AtomKokkos; init() has
   claimed the host copy at the start of the run.
------------------------------------------------------------------------- */

template <class DeviceType, class Function>
void FixPropertyAtomKokkos::for_each_custom_value(Function &&f)
{
  int offset = 0;
  for (int nv = 0; nv < nvalue; nv++) {
    const int idx = index[nv];
    if (styles[nv] == IVEC) {
      f(Kokkos::subview(atomKK->k_ivector.view<DeviceType>(), idx, Kokkos::ALL), 1, offset);
      offset += 1;
    } else if (styles[nv] == DVEC) {
      f(Kokkos::subview(atomKK->k_dvector.view<DeviceType>(), idx, Kokkos::ALL), 1, offset);
      offset += 1;
    } else if (styles[nv] == IARRAY) {
      f(atomKK->k_iarray.view_host()[idx].k_view.view<DeviceType>(), cols[nv], offset);
      offset += cols[nv];
    } else if (styles[nv] == DARRAY) {
      f(atomKK->k_darray.view_host()[idx].k_view.view<DeviceType>(), cols[nv], offset);
      offset += cols[nv];
    }
  }
}

/* ----------------------------------------------------------------------
   apply the permutation of the device atom sort to the custom values
------------------------------------------------------------------------- */

void FixPropertyAtomKokkos::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  atomKK->sync(Device, custom_datamask);
  for_each_custom_value<LMPDeviceType>(
      [&](auto values, int, int) { Sorter.sort(LMPDeviceType(), values); });
  atomKK->modified(Device, custom_datamask);
}

/* ---------------------------------------------------------------------- */

int FixPropertyAtomKokkos::pack_exchange_kokkos(const int &nsend, DAT::tdual_double_2d_lr &k_buf,
                                                DAT::tdual_int_1d k_sendlist,
                                                DAT::tdual_int_1d k_copylist,
                                                ExecutionSpace space)
{
  if (space == Device)
    return pack_exchange_space<LMPDeviceType>(nsend, k_buf, k_sendlist, k_copylist);
  return pack_exchange_space<LMPHostType>(nsend, k_buf, k_sendlist, k_copylist);
}

/* ---------------------------------------------------------------------- */

void FixPropertyAtomKokkos::unpack_exchange_kokkos(DAT::tdual_double_2d_lr &k_buf,
                                                   DAT::tdual_int_1d &k_indices, int nrecv,
                                                   int nrecv1, int nextrarecv1,
                                                   ExecutionSpace space)
{
  if (space == Device)
    unpack_exchange_space<LMPDeviceType>(k_buf, k_indices, nrecv, nrecv1, nextrarecv1);
  else
    unpack_exchange_space<LMPHostType>(k_buf, k_indices, nrecv, nrecv1, nextrarecv1);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
int FixPropertyAtomKokkos::pack_exchange_space(int nsend, DAT::tdual_double_2d_lr &k_buf,
                                               DAT::tdual_int_1d &k_sendlist,
                                               DAT::tdual_int_1d &k_copylist)
{
  const ExecutionSpace space = ExecutionSpaceFromDevice<DeviceType>::space;
  atomKK->sync(space, custom_datamask);
  k_buf.sync<DeviceType>();
  k_sendlist.sync<DeviceType>();
  k_copylist.sync<DeviceType>();

  const typename ArrayTypes<DeviceType>::t_double_1d_um buf(k_buf.view<DeviceType>().data(),
                                                            k_buf.extent(0) * k_buf.extent(1));
  const auto sendlist = k_sendlist.view<DeviceType>();
  const auto copylist = k_copylist.view<DeviceType>();
  const int width = values_peratom;

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nsend),
                       PropertyAtomExchangeOffsets<DeviceType>{buf, nsend, width});

  for_each_custom_value<DeviceType>([&](auto values, int ncol, int offset) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nsend),
                         PropertyAtomPackExchange<DeviceType, decltype(values)>{
                             buf, sendlist, copylist, values, nsend, width, offset, ncol});
  });

  k_buf.modify<DeviceType>();
  atomKK->modified(space, custom_datamask);
  return nsend * (width + 1);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixPropertyAtomKokkos::unpack_exchange_space(DAT::tdual_double_2d_lr &k_buf,
                                                  DAT::tdual_int_1d &k_indices, int nrecv,
                                                  int nrecv1, int nextrarecv1)
{
  const ExecutionSpace space = ExecutionSpaceFromDevice<DeviceType>::space;
  atomKK->sync(space, custom_datamask);
  k_buf.sync<DeviceType>();
  k_indices.sync<DeviceType>();

  const typename ArrayTypes<DeviceType>::t_double_1d_um buf(k_buf.view<DeviceType>().data(),
                                                            k_buf.extent(0) * k_buf.extent(1));
  const auto indices = k_indices.view<DeviceType>();

  for_each_custom_value<DeviceType>([&](auto values, int ncol, int offset) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nrecv),
                         PropertyAtomUnpackExchange<DeviceType, decltype(values)>{
                             buf, indices, values, nrecv1, nextrarecv1, offset, ncol});
  });
  atomKK->modified(space, custom_datamask);
}
