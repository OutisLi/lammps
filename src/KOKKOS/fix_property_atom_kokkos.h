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
FixStyle(property/atom/kk,FixPropertyAtomKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_PROPERTY_ATOM_KOKKOS_H
#define LMP_FIX_PROPERTY_ATOM_KOKKOS_H

#include "fix_property_atom.h"
#include "atom_vec_kokkos.h"
#include "kokkos_base.h"

namespace LAMMPS_NS {

class FixPropertyAtomKokkos : public FixPropertyAtom, public KokkosBase {
 public:
  FixPropertyAtomKokkos(class LAMMPS *, int, char **);
  void post_constructor() override;
  ~FixPropertyAtomKokkos() override;
  void init() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

  void sync(ExecutionSpace space, uint64_t mask);
  void modified(ExecutionSpace space, uint64_t mask);
  void sync_pinned(ExecutionSpace space, uint64_t mask, int async_flag = 0);

  // device sort and exchange of the custom per-atom vectors and arrays

  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;
  int pack_exchange_kokkos(const int &nsend, DAT::tdual_double_2d_lr &k_buf,
                           DAT::tdual_int_1d k_sendlist, DAT::tdual_int_1d k_copylist,
                           ExecutionSpace space) override;
  void unpack_exchange_kokkos(DAT::tdual_double_2d_lr &k_buf, DAT::tdual_int_1d &k_indices,
                              int nrecv, int nrecv1, int nextrarecv1,
                              ExecutionSpace space) override;

 private:
  int dvector_flag;
  int ivector_flag;
  int iarray_flag;
  int darray_flag;
  uint64_t custom_datamask;    // masks of the custom vectors and arrays

  template <class DeviceType, class Function> void for_each_custom_value(Function &&);
  template <class DeviceType>
  int pack_exchange_space(int, DAT::tdual_double_2d_lr &, DAT::tdual_int_1d &,
                          DAT::tdual_int_1d &);
  template <class DeviceType>
  void unpack_exchange_space(DAT::tdual_double_2d_lr &, DAT::tdual_int_1d &, int, int, int);
};

}

#endif
#endif

