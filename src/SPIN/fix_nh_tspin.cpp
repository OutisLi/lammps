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

/* ------------------------------------------------------------------------
   Contributing author: Zhengtao Huang (The University of Hong Kong)
                        hzt990224@gmail.com

   Nose-Hoover machinery for inertial spin dynamics.  This is a base class
   and registers no fix style of its own; fix nvt/tspin, fix npt/tspin and
   fix nph/tspin derive from it.  The integration is that of FixNHTSpinT
   over fix nh, with the per-atom work on the spin velocities done here on
   the host.
------------------------------------------------------------------------- */

#include "fix_nh_tspin.h"
#include "fix_nh_tspin_impl.h"

#include "atom.h"

#include <cmath>

using namespace LAMMPS_NS;

namespace LAMMPS_NS {
template class FixNHTSpinT<FixNH>;
}

/* ---------------------------------------------------------------------- */

FixNHTSpin::FixNHTSpin(LAMMPS *lmp, int narg, char **arg) : FixNHTSpinT<FixNH>(lmp, narg, arg) {}

/* ----------------------------------------------------------------------
   instantaneous spin temperature from the spin kinetic energy
------------------------------------------------------------------------- */

double FixNHTSpin::compute_spin_temp()
{
  double **sp = atom->sp;
  double **s_dot = atom->darray[index_vs];
  double *s_mass = atom->dvector[index_sm];
  int *mask = atom->mask;
  const int nlocal = atom->nlocal;

  double ke = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (s_mass[i] <= 0.0) continue;
    ke += s_mass[i] *
        (s_dot[i][0] * s_dot[i][0] + s_dot[i][1] * s_dot[i][1] + s_dot[i][2] * s_dot[i][2]);
  }

  double keall = 0.0;
  MPI_Allreduce(&ke, &keall, 1, MPI_DOUBLE, MPI_SUM, world);

  if (tdof_spin == 0.0) return 0.0;
  return keall * force->mvv2e / (tdof_spin * boltz);
}

/* ----------------------------------------------------------------------
   scale all thermostatted spin velocities by factor
------------------------------------------------------------------------- */

void FixNHTSpin::nh_vs_scale(double factor)
{
  double **sp = atom->sp;
  double **s_dot = atom->darray[index_vs];
  double *s_mass = atom->dvector[index_sm];
  int *mask = atom->mask;
  const int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (s_mass[i] <= 0.0) continue;
    s_dot[i][0] *= factor;
    s_dot[i][1] *= factor;
    s_dot[i][2] *= factor;
  }
}

/* ----------------------------------------------------------------------
   half-step update of the spin velocities
------------------------------------------------------------------------- */

void FixNHTSpin::spin_kick()
{
  double **sp = atom->sp;
  double **s_dot = atom->darray[index_vs];
  double **fm = atom->fm;
  double *s_mass = atom->dvector[index_sm];
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (s_mass[i] <= 0.0) continue;

    // At an isolated zero crossing fm cannot be inverted because its
    // |S|/hbar encoding vanishes.  Skip only that kick; the drift below keeps
    // the Cartesian spin moving and leaves the zero-modulus state immediately.

    if (sp[i][3] == 0.0) continue;

    // A TSPIN-compatible fm encodes the full spin force in rad.THz;
    // hbar/|S| converts it back to eV/muB, radial component included.

    const double dtfsp = dtf * hbar / (s_mass[i] * sp[i][3]);

    s_dot[i][0] += dtfsp * fm[i][0];
    s_dot[i][1] += dtfsp * fm[i][1];
    s_dot[i][2] += dtfsp * fm[i][2];
  }
}

/* ----------------------------------------------------------------------
   full-step update of the spins
------------------------------------------------------------------------- */

void FixNHTSpin::spin_drift()
{
  double **sp = atom->sp;
  double **s_dot = atom->darray[index_vs];
  double *s_mass = atom->dvector[index_sm];
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (s_mass[i] <= 0.0) continue;

    // rebuild the un-normalized spin, advance it, then split it again
    // into a direction and a modulus

    double s[3];
    s[0] = sp[i][3] * sp[i][0] + dtv * s_dot[i][0];
    s[1] = sp[i][3] * sp[i][1] + dtv * s_dot[i][1];
    s[2] = sp[i][3] * sp[i][2] + dtv * s_dot[i][2];

    const double smag = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    sp[i][3] = smag;
    if (smag > 0.0) {
      sp[i][0] = s[0] / smag;
      sp[i][1] = s[1] / smag;
      sp[i][2] = s[2] / smag;
    }
  }
}
