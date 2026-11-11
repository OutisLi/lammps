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

#ifndef LMP_MIN_EXTRA_DOF_H
#define LMP_MIN_EXTRA_DOF_H

namespace LAMMPS_NS {

// Owner of an extra per-atom minimization variable, which the line-search
// minimizers move together with the atom coordinates: a pair style or a min
// style.  The index names the variable among all requests.

class MinExtraDOF {
 public:
  // point to the variable and to the force on it, after atoms migrated
  virtual void min_xf_pointers(int, double **, double **) {}
  // refresh the force on the variable after a force evaluation
  virtual void min_xf_get(int) {}
  // apply a change of the variable made by the minimizer
  virtual void min_x_set(int) {}

 protected:
  // owners are deleted through their Pair or Min base, never through this one
  ~MinExtraDOF() = default;
};

}    // namespace LAMMPS_NS

#endif
