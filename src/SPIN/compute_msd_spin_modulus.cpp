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

#include "compute_msd_spin_modulus.h"

#include "atom.h"
#include "error.h"
#include "fix_store_atom.h"
#include "group.h"
#include "modify.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeMSDSpinModulus::ComputeMSDSpinModulus(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), average_flag(0), average_count(0), atom_count(0),
    id_fix(nullptr), fix(nullptr)
{
  if (!atom->sp_flag)
    error->all(FLERR, "Compute msd/spin/mod requires atom style spin");
  if (group->dynamic[igroup])
    error->all(FLERR, "Compute msd/spin/mod is not compatible with dynamic groups");

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "average") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "compute msd/spin/mod average", error);
      average_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, iarg, "Unknown compute msd/spin/mod keyword: {}", arg[iarg]);
    }
  }

  scalar_flag = 1;
  extscalar = 0;
  create_attribute = 1;
  dynamic_group_allow = 0;

  id_fix = utils::strdup(id + std::string("_COMPUTE_STORE_SPIN_MOD"));
  fix = dynamic_cast<FixStoreAtom *>(modify->add_fix(
    fmt::format("{} {} STORE/ATOM 1 0 0 1", id_fix, group->names[igroup])));

  if (fix->restart_reset) {
    fix->restart_reset = 0;
  } else {
    double *reference = fix->vstore;
    double **sp = atom->sp;
    int *mask = atom->mask;
    for (int i = 0; i < atom->nlocal; i++)
      reference[i] = (mask[i] & groupbit) ? sp[i][3] : 0.0;
  }
}

/* ---------------------------------------------------------------------- */

ComputeMSDSpinModulus::~ComputeMSDSpinModulus()
{
  if (modify->nfix) modify->delete_fix(id_fix);
  delete[] id_fix;
}

/* ---------------------------------------------------------------------- */

void ComputeMSDSpinModulus::init()
{
  fix = dynamic_cast<FixStoreAtom *>(modify->get_fix_by_id(id_fix));
  if (!fix)
    error->all(FLERR, "Could not find compute msd/spin/mod storage fix {}", id_fix);
  atom_count = group->count(igroup);
}

/* ---------------------------------------------------------------------- */

double ComputeMSDSpinModulus::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  double *reference = fix->vstore;
  double **sp = atom->sp;
  int *mask = atom->mask;
  double local_msd = 0.0;

  double average_weight = 0.0;
  if (average_flag) {
    average_count++;
    average_weight = 1.0 / (average_count + 1);
  }

  for (int i = 0; i < atom->nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (average_flag)
      reference[i] = (reference[i] * average_count + sp[i][3]) * average_weight;
    const double displacement = sp[i][3] - reference[i];
    local_msd += displacement * displacement;
  }

  MPI_Allreduce(&local_msd, &scalar, 1, MPI_DOUBLE, MPI_SUM, world);
  if (atom_count > 0) scalar /= atom_count;
  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeMSDSpinModulus::set_arrays(int i)
{
  fix->vstore[i] = atom->sp[i][3];
}
