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

#include "compute_msd_spin.h"

#include "atom.h"
#include "error.h"
#include "fix_store_atom.h"
#include "group.h"
#include "modify.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeMSDSpin::ComputeMSDSpin(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), average_flag(0), average_count(0), atom_count(0),
    id_fix(nullptr), fix(nullptr)
{
  if (!atom->sp_flag)
    error->all(FLERR, "Compute msd/spin requires atom style spin");
  if (group->dynamic[igroup])
    error->all(FLERR, "Compute msd/spin is not compatible with dynamic groups");

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "average") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "compute msd/spin average", error);
      average_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, iarg, "Unknown compute msd/spin keyword: {}", arg[iarg]);
    }
  }

  vector_flag = 1;
  size_vector = 4;
  extvector = 0;
  create_attribute = 1;
  dynamic_group_allow = 0;
  vector = new double[size_vector];

  id_fix = utils::strdup(id + std::string("_COMPUTE_STORE_SPIN"));
  fix = dynamic_cast<FixStoreAtom *>(modify->add_fix(
    fmt::format("{} {} STORE/ATOM 3 0 0 1", id_fix, group->names[igroup])));

  if (fix->restart_reset) {
    fix->restart_reset = 0;
  } else {
    double **reference = fix->astore;
    double **sp = atom->sp;
    int *mask = atom->mask;
    for (int i = 0; i < atom->nlocal; i++) {
      if (mask[i] & groupbit) {
        reference[i][0] = sp[i][0] * sp[i][3];
        reference[i][1] = sp[i][1] * sp[i][3];
        reference[i][2] = sp[i][2] * sp[i][3];
      } else {
        reference[i][0] = 0.0;
        reference[i][1] = 0.0;
        reference[i][2] = 0.0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

ComputeMSDSpin::~ComputeMSDSpin()
{
  if (modify->nfix) modify->delete_fix(id_fix);
  delete[] id_fix;
  delete[] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeMSDSpin::init()
{
  fix = dynamic_cast<FixStoreAtom *>(modify->get_fix_by_id(id_fix));
  if (!fix)
    error->all(FLERR, "Could not find compute msd/spin storage fix {}", id_fix);
  atom_count = group->count(igroup);
}

/* ---------------------------------------------------------------------- */

void ComputeMSDSpin::compute_vector()
{
  invoked_vector = update->ntimestep;

  double **reference = fix->astore;
  double **sp = atom->sp;
  int *mask = atom->mask;
  double local_msd[4] = {0.0, 0.0, 0.0, 0.0};

  double average_weight = 0.0;
  if (average_flag) {
    average_count++;
    average_weight = 1.0 / (average_count + 1);
  }

  for (int i = 0; i < atom->nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    const double current[3] = {
      sp[i][0] * sp[i][3],
      sp[i][1] * sp[i][3],
      sp[i][2] * sp[i][3]
    };
    if (average_flag) {
      for (int d = 0; d < 3; d++)
        reference[i][d] =
          (reference[i][d] * average_count + current[d]) * average_weight;
    }

    for (int d = 0; d < 3; d++) {
      const double displacement = current[d] - reference[i][d];
      local_msd[d] += displacement * displacement;
      local_msd[3] += displacement * displacement;
    }
  }

  MPI_Allreduce(local_msd, vector, 4, MPI_DOUBLE, MPI_SUM, world);
  if (atom_count > 0)
    for (int d = 0; d < 4; d++) vector[d] /= atom_count;
}

/* ---------------------------------------------------------------------- */

void ComputeMSDSpin::set_arrays(int i)
{
  double **reference = fix->astore;
  double **sp = atom->sp;
  reference[i][0] = sp[i][0] * sp[i][3];
  reference[i][1] = sp[i][1] * sp[i][3];
  reference[i][2] = sp[i][2] * sp[i][3];
}
