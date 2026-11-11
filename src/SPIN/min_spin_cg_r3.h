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

#ifdef MINIMIZE_CLASS
// clang-format off
MinimizeStyle(spin/cg/r3,MinSpinCGR3);
// clang-format on
#else

#ifndef LMP_MIN_SPIN_CG_R3_H
#define LMP_MIN_SPIN_CG_R3_H

#include "min_cg.h"

namespace LAMMPS_NS {

// Conjugate gradients over the atom coordinates and the Cartesian spin
// vectors, which the line search moves as extra per-atom variables.

class MinSpinCGR3 : public MinCG, public MinExtraDOF {
 public:
  MinSpinCGR3(class LAMMPS *);
  ~MinSpinCGR3() override;
  void setup_style() override;
  double memory_usage() override;

  void min_xf_pointers(int, double **, double **) override;
  void min_xf_get(int) override;
  void min_x_set(int) override;

 protected:
  int nmax;               // rows allocated in the spin buffers
  double **spin;          // Cartesian spin vector of each local atom
  double **spin_force;    // force -dE/dS on it, zero for a zero moment

  void grow();
};

}    // namespace LAMMPS_NS

#endif
#endif
