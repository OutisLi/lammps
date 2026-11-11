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
   Spin part of the Nose-Hoover tspin styles: the keywords, the spin degrees
   of freedom and the Nose-Hoover chain acting on the spin velocities, with
   its energy and restart data.  FixNHTSpinT joins it to a parent Nose-Hoover
   fix, fix nh for fix nvt/tspin, npt/tspin and nph/tspin and fix nh/kk for
   their KOKKOS variants.  The per-atom work on the spin velocities is left
   to the deriving fix, which runs it where its atom data live.
------------------------------------------------------------------------- */

#ifndef LMP_TSPIN_NH_H
#define LMP_TSPIN_NH_H

#include <cstdio>

namespace LAMMPS_NS {

class LAMMPS;

// time step factors and temperature of the parent Nose-Hoover chain that
// the spin chain follows
struct TSpinNHStep {
  int nc_tchain;
  double dt4, dt8, dthalf, tdrag_factor, boltz, t_target, t_freq;
};

class TSpinNH {
 public:
  int lattice_flag;    // read by PairSpin::init_style()

 protected:
  int spin_flag;        // 1 if the spin degrees of freedom are integrated
  int spinmass_flag;    // 1 if the spinmass keyword was used, then this fix owns s_mass
  double spinmass;      // spin mass in units of the atomic mass of the type
  int index_vs;         // custom per-atom property holding the spin velocity
  int index_sm;         // custom per-atom property holding the spin mass
  double hbar;          // Planck constant over 2 pi, in eV/(rad.THz)

  double t_current_spin, ke_target_spin, tdof_spin;

  // Nose-Hoover chain acting on the spin velocities, parallel to the chain
  // of the parent class and driven by the same target temperature

  int mtchain_spin;
  double *etas, *etas_dot, *etas_dotdot, *etas_mass;

  explicit TSpinNH(LAMMPS *);

  // a fix derived from this class is deleted through its Fix base
  ~TSpinNH() = default;

  void parse_keywords(const char *, int, char **, int);
  void create_chain(int);
  void destroy_chain();
  void set_spin_mass(int);
  void count_spin_dof(const char *, int);
  void init_chain_masses(const TSpinNHStep &);
  void integrate_chain(const TSpinNHStep &);
  double chain_energy(const TSpinNHStep &) const;
  int pack_chain(double *, int) const;
  void unpack_chain(const double *, int);

  // instantaneous temperature of the thermostatted spin degrees of freedom
  virtual double compute_spin_temp() = 0;
  // scale the thermostatted spin velocities by a factor
  virtual void nh_vs_scale(double) = 0;

 private:
  LAMMPS *tspin_lmp;
};

}    // namespace LAMMPS_NS

#endif
