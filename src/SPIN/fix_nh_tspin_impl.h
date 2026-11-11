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

   Member definitions of FixNHTSpinT, included by the sources that
   instantiate it: fix_nh_tspin.cpp for fix nh and fix_nh_tspin_kokkos.cpp
   for fix nh/kk.

   In addition to the lattice degrees of freedom handled by the parent
   class, the spins are integrated with velocity-Verlet and their spin
   velocities are coupled to a second Nose-Hoover chain.  That chain is
   independent of the lattice chain but is driven by the same target
   temperature and the same damping parameter, so that a spin-lattice
   simulation is thermostatted as a whole.
------------------------------------------------------------------------- */

#ifndef LMP_FIX_NH_TSPIN_IMPL_H
#define LMP_FIX_NH_TSPIN_IMPL_H

#include "fix_nh_tspin.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "tspin.h"
#include "update.h"

#include <string>

namespace LAMMPS_NS {

/* ---------------------------------------------------------------------- */

template <class NH>
FixNHTSpinT<NH>::FixNHTSpinT(LAMMPS *lmp, int narg, char **arg) :
    NH(lmp, narg, arg), TSpinNH(lmp)
{
  if (!this->atom->sp_flag) this->error->all(FLERR, "Fix {} requires atom style spin", this->style);

  // the spin degrees of freedom and the chain masses are set once at setup

  this->dynamic_group_allow = 0;

  // parse the keywords that the parent class skipped

  parse_keywords(this->style, narg, arg, this->pstat_flag);
  create_chain(this->mtchain);
}

/* ----------------------------------------------------------------------
   a KOKKOS variant destroys its device copies too, which share the chain
------------------------------------------------------------------------- */

template <class NH> FixNHTSpinT<NH>::~FixNHTSpinT()
{
  if (this->copymode) return;
  destroy_chain();
}

/* ----------------------------------------------------------------------
   the spin velocity and the spin mass live in custom per-atom properties, so
   that they migrate with the atoms and are written to restart files without a
   new atom style.  Modify::add_fix() may only be called from here, once this
   fix is fully registered.
------------------------------------------------------------------------- */

template <class NH> void FixNHTSpinT<NH>::post_constructor()
{
  tspin_create_state(this->modify, this->error);
  tspin_bind_state(this->atom, this->error, std::string("Fix ") + this->style, index_vs, index_sm);

  // assign the spin masses right away, so that the velocity/tspin command can
  // be used before the first run

  if (spinmass_flag) set_spin_mass(this->groupbit);
}

/* ---------------------------------------------------------------------- */

template <class NH> void FixNHTSpinT<NH>::init()
{
  NH::init();

  // the state fix can be removed between runs, so bind to it again here

  tspin_bind_state(this->atom, this->error, std::string("Fix ") + this->style, index_vs, index_sm);

  // the spin force is not stored per rRESPA level, so the spin update cannot
  // be split over the levels

  if (utils::strmatch(this->update->integrate_style, "^respa"))
    this->error->all(FLERR, "Fix {} is not compatible with run_style respa", this->style);

  reset_dt();
}

/* ---------------------------------------------------------------------- */

template <class NH> void FixNHTSpinT<NH>::reset_dt()
{
  NH::reset_dt();
  hbar = this->force->hplanck / MathConst::MY_2PI;
}

/* ---------------------------------------------------------------------- */

template <class NH> void FixNHTSpinT<NH>::setup(int vflag)
{
  if (spinmass_flag) set_spin_mass(this->groupbit);

  if (!spin_flag) {
    tdof_spin = 0.0;
    NH::setup(vflag);
    return;
  }

  count_spin_dof(this->style, this->groupbit);
  NH::setup(vflag);
  if (this->tstat_flag) init_chain_masses(chain_step());
}

/* ----------------------------------------------------------------------
   time step factors and temperature of the lattice chain
------------------------------------------------------------------------- */

template <class NH> TSpinNHStep FixNHTSpinT<NH>::chain_step() const
{
  return {this->nc_tchain, this->dt4, this->dt8, this->dthalf, this->tdrag_factor,
          this->boltz, this->t_target, this->t_freq};
}

/* ----------------------------------------------------------------------
   the parent class calls this to propagate the lattice chain
   also propagate the spin chain, using the same target temperature
   with a frozen lattice the lattice chain is skipped entirely, so that it
   does not drift and pollute the conserved quantity
------------------------------------------------------------------------- */

template <class NH> void FixNHTSpinT<NH>::nhc_temp_integrate()
{
  if (lattice_flag) NH::nhc_temp_integrate();
  if (spin_flag) integrate_chain(chain_step());
}

/* ----------------------------------------------------------------------
   half-step velocity update, for the lattice and for the spins
------------------------------------------------------------------------- */

template <class NH> void FixNHTSpinT<NH>::nve_v()
{
  if (lattice_flag) NH::nve_v();
  if (spin_flag) spin_kick();
}

/* ----------------------------------------------------------------------
   full-step position update, for the lattice and for the spins
------------------------------------------------------------------------- */

template <class NH> void FixNHTSpinT<NH>::nve_x()
{
  if (lattice_flag) NH::nve_x();
  if (spin_flag) spin_drift();
}

/* ----------------------------------------------------------------------
   the lattice thermostat must not rescale the velocities of a frozen
   lattice; the spin velocities are rescaled by the spin chain instead
------------------------------------------------------------------------- */

template <class NH> void FixNHTSpinT<NH>::nh_v_temp()
{
  if (lattice_flag) NH::nh_v_temp();
}

/* ----------------------------------------------------------------------
   add the energy of the spin chain to the conserved quantity
------------------------------------------------------------------------- */

template <class NH> double FixNHTSpinT<NH>::compute_scalar()
{
  double energy = lattice_flag ? NH::compute_scalar() : 0.0;
  if (!this->tstat_flag || !spin_flag) return energy;
  return energy + chain_energy(chain_step());
}

/* ---------------------------------------------------------------------- */

template <class NH> int FixNHTSpinT<NH>::size_restart_global()
{
  return NH::size_restart_global() + 2 * this->mtchain;
}

/* ----------------------------------------------------------------------
   the restart data of the parent class, in its own layout, followed by the
   spin chain
------------------------------------------------------------------------- */

template <class NH> int FixNHTSpinT<NH>::pack_restart_data(double *list)
{
  return pack_chain(list, NH::pack_restart_data(list));
}

/* ---------------------------------------------------------------------- */

template <class NH> void FixNHTSpinT<NH>::restart(char *buf)
{
  const auto list = (const double *) buf;
  const int n = NH::unpack_restart_data(list);

  // the spin chain follows the data of the parent class and has the length
  // of the stored lattice chain; like that chain, it is used only when its
  // length matches the current one

  const int m = static_cast<int>(list[0]) ? static_cast<int>(list[1]) : 0;
  if (this->tstat_flag && (m == this->mtchain)) unpack_chain(list, n);
}

}    // namespace LAMMPS_NS

#endif
