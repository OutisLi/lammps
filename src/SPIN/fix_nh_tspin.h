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

/* ------------------------------------------------------------------------
   Contributing author: Zhengtao Huang (The University of Hong Kong)
                        hzt990224@gmail.com
------------------------------------------------------------------------- */

#ifndef LMP_FIX_NH_TSPIN_H
#define LMP_FIX_NH_TSPIN_H

#include "fix_nh.h"
#include "tspin_nh.h"

namespace LAMMPS_NS {

// Nose-Hoover integration of the lattice and of the inertial spin dynamics on
// top of a parent Nose-Hoover fix NH: fix nh for the host styles and fix nh/kk
// for their KOKKOS variants.  The deriving fix performs the per-atom work on
// the spin velocities where its atom data live: the spin temperature, the
// chain scaling, the kick and the drift.  The member definitions are in
// fix_nh_tspin_impl.h.

template <class NH> class FixNHTSpinT : public NH, public TSpinNH {
 public:
  FixNHTSpinT(class LAMMPS *, int, char **);
  ~FixNHTSpinT() override;

  void post_constructor() override;
  void init() override;
  void setup(int) override;
  double compute_scalar() override;
  void reset_dt() override;
  int size_restart_global() override;
  int pack_restart_data(double *) override;
  void restart(char *) override;

 protected:
  void nve_v() override;
  void nve_x() override;
  void nh_v_temp() override;
  void nhc_temp_integrate() override;

  // half-step update of the spin velocities
  virtual void spin_kick() = 0;
  // full-step update of the spins
  virtual void spin_drift() = 0;
  TSpinNHStep chain_step() const;
};

extern template class FixNHTSpinT<FixNH>;

class FixNHTSpin : public FixNHTSpinT<FixNH> {
 public:
  FixNHTSpin(class LAMMPS *, int, char **);

 protected:
  double compute_spin_temp() override;
  void nh_vs_scale(double) override;
  void spin_kick() override;
  void spin_drift() override;
};

}    // namespace LAMMPS_NS

#endif
