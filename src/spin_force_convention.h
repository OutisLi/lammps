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

#ifndef LMP_SPIN_FORCE_CONVENTION_H
#define LMP_SPIN_FORCE_CONVENTION_H

#include "error.h"
#include "force.h"
#include "lammps.h"
#include "pair.h"
#include "utils.h"

namespace LAMMPS_NS {

/**
 * Key a pair style answers through Pair::extract to declare that it writes
 * the Cartesian generalized force -dE/dS into atom->fm. The value is an int
 * equal to one and the reported dimension is zero.
 */
static constexpr const char *FM_CARTESIAN_FORCE_KEY = "fm_cartesian_force";

/**
 * Require that the active pair style writes the Cartesian magnetic force.
 *
 * Newtonian spin dynamics integrates atom->fm as a generalized force. A pair
 * style that instead writes a precession quantity differs from it by the
 * moment magnitude over hbar, which is neither dimensionally nor numerically
 * detectable downstream: the run would produce a plausible but wrong
 * trajectory. The convention is therefore declared by the producer and
 * verified here rather than inferred from the pair style name, so that a
 * style added later is covered by the same check.
 *
 * A hybrid pair style is refused even when one sub-style declares the
 * convention, because the remaining sub-styles accumulate into the same fm
 * array under a different one.
 *
 * \param lmp  the LAMMPS instance whose active pair style is inspected
 * \param who  the requesting command, named in any error message
 */
static inline void require_cartesian_magnetic_force(LAMMPS *lmp, const char *who)
{
  if (!lmp->force->pair)
    lmp->error->all(FLERR, "{} requires a pair style", who);

  if (utils::strmatch(lmp->force->pair_style, "^hybrid"))
    lmp->error->all(FLERR,
                    "{} does not support hybrid pair styles, whose sub-styles "
                    "may accumulate magnetic forces of different conventions "
                    "into the same array",
                    who);

  int dim = -1;
  auto *declared =
      static_cast<int *>(lmp->force->pair->extract(FM_CARTESIAN_FORCE_KEY, dim));
  if (!declared || dim != 0 || *declared != 1)
    lmp->error->all(FLERR,
                    "{} requires a pair style that writes the Cartesian "
                    "magnetic force -dE/dS, which pair style {} does not "
                    "declare; use dpa4spin/kk with magnetic_force "
                    "cartesian",
                    who, lmp->force->pair_style);
}

}    // namespace LAMMPS_NS

#endif
