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

#ifndef LMP_SPIN_COUNTER_RNG_H
#define LMP_SPIN_COUNTER_RNG_H

#include "lmptype.h"

#include <cmath>
#include <cstdint>

namespace LAMMPS_NS {

/**
 * Counter-based Gaussian random values for Newtonian spin dynamics.
 *
 * Every value is a pure function of its physical identity rather than an
 * MPI-rank-local generator state. This keeps trajectories invariant under
 * atom migration, domain decomposition, and restart. Stream identifiers are
 * stable domain separators: adding a consumer requires a distinct enumerator.
 */
class SpinCounterRNG {
 public:
  enum class Stream : std::uint64_t {
    VELOCITY_INITIALIZATION = 0x243f6a8885a308d3ULL,
    LANGEVIN_FORCE = 0x13198a2e03707344ULL
  };

  static double gaussian(int seed, bigint timestep, tagint atom_tag,
                         int component, Stream stream) noexcept
  {
    std::uint64_t counter =
      mix(static_cast<std::uint64_t>(seed) ^ static_cast<std::uint64_t>(stream));
    counter = combine(counter, static_cast<std::uint64_t>(timestep));
    counter = combine(counter, static_cast<std::uint64_t>(atom_tag));
    counter = combine(counter, static_cast<std::uint64_t>(component));

    const double u1 = uniform_open01(counter);
    const double u2 = uniform_open01(counter ^ GAUSSIAN_PAIR_SALT);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(TWO_PI * u2);
  }

 private:
  static constexpr std::uint64_t HASH_GOLDEN_RATIO = 0x9e3779b97f4a7c15ULL;
  static constexpr std::uint64_t HASH_MIX_1 = 0xbf58476d1ce4e5b9ULL;
  static constexpr std::uint64_t HASH_MIX_2 = 0x94d049bb133111ebULL;
  static constexpr std::uint64_t COMBINE_SALT = 0xd2b74407b1ce6e93ULL;
  static constexpr std::uint64_t GAUSSIAN_PAIR_SALT = 0xa0761d6478bd642fULL;
  static constexpr double INVERSE_53_BITS = 1.0 / 9007199254740992.0;
  static constexpr double TWO_PI = 6.283185307179586476925286766559;

  static std::uint64_t mix(std::uint64_t value) noexcept
  {
    value += HASH_GOLDEN_RATIO;
    value = (value ^ (value >> 30)) * HASH_MIX_1;
    value = (value ^ (value >> 27)) * HASH_MIX_2;
    return value ^ (value >> 31);
  }

  static std::uint64_t combine(std::uint64_t state, std::uint64_t value) noexcept
  {
    return mix(state ^ mix(value + COMBINE_SALT));
  }

  static double uniform_open01(std::uint64_t counter) noexcept
  {
    return (static_cast<double>(mix(counter) >> 11) + 0.5) * INVERSE_53_BITS;
  }
};

}    // namespace LAMMPS_NS

#endif
