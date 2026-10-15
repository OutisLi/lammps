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

#include "spin_counter_rng.h"

#include "gtest/gtest.h"

#include <cmath>

using namespace LAMMPS_NS;

TEST(SpinCounterRNG, FixedVectors)
{
  using Stream = SpinCounterRNG::Stream;

  EXPECT_NEAR(SpinCounterRNG::gaussian(12345, 0, 1, 0, Stream::VELOCITY_INITIALIZATION),
              1.0333481823427786, 1.0e-15);
  EXPECT_NEAR(SpinCounterRNG::gaussian(12345, 0, 1, 0, Stream::LANGEVIN_FORCE),
              -0.70513635874814495, 1.0e-15);
  EXPECT_NEAR(SpinCounterRNG::gaussian(987654321, 42, 922337, 2, Stream::LANGEVIN_FORCE),
              0.28850356613294525, 1.0e-15);
  EXPECT_NEAR(
    SpinCounterRNG::gaussian(7, 1000000, 2147483647, 1, Stream::VELOCITY_INITIALIZATION),
    0.40157288412769265, 1.0e-15);
}

TEST(SpinCounterRNG, CounterFieldsAndStreamsAreSeparated)
{
  using Stream = SpinCounterRNG::Stream;

  const double reference =
    SpinCounterRNG::gaussian(2026, 17, 42, 0, Stream::LANGEVIN_FORCE);
  EXPECT_DOUBLE_EQ(reference,
                   SpinCounterRNG::gaussian(2026, 17, 42, 0, Stream::LANGEVIN_FORCE));
  EXPECT_NE(reference,
            SpinCounterRNG::gaussian(2027, 17, 42, 0, Stream::LANGEVIN_FORCE));
  EXPECT_NE(reference,
            SpinCounterRNG::gaussian(2026, 18, 42, 0, Stream::LANGEVIN_FORCE));
  EXPECT_NE(reference,
            SpinCounterRNG::gaussian(2026, 17, 43, 0, Stream::LANGEVIN_FORCE));
  EXPECT_NE(reference,
            SpinCounterRNG::gaussian(2026, 17, 42, 1, Stream::LANGEVIN_FORCE));
  EXPECT_NE(reference,
            SpinCounterRNG::gaussian(2026, 17, 42, 0,
                                     Stream::VELOCITY_INITIALIZATION));
}

TEST(SpinCounterRNG, GaussianMoments)
{
  using Stream = SpinCounterRNG::Stream;

  constexpr int sample_count = 100000;
  double sum = 0.0;
  double sum_of_squares = 0.0;
  for (tagint tag = 1; tag <= sample_count; ++tag) {
    const double value =
      SpinCounterRNG::gaussian(314159, 2718, tag, tag % 3, Stream::LANGEVIN_FORCE);
    sum += value;
    sum_of_squares += value * value;
  }

  const double mean = sum / sample_count;
  const double variance = sum_of_squares / sample_count - mean * mean;
  EXPECT_LT(std::abs(mean), 1.0e-2);
  EXPECT_NEAR(variance, 1.0, 2.0e-2);
}
