// clang-format off
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

#ifndef LMP_MATH_ROUNDED_KOKKOS_H
#define LMP_MATH_ROUNDED_KOKKOS_H

#include "kokkos_type.h"

/* ----------------------------------------------------------------------
   floating-point operations that are rounded one at a time

   Device compilers may contract a*b+c into one fused multiply-add, which
   rounds once where the separate operations round twice; CUDA does so by
   default. On CUDA devices these functions evaluate each operation with its
   explicitly rounded intrinsic, which is never contracted, so a device kernel
   reproduces host arithmetic bit for bit where the host compiler keeps the
   operations apart as well, as x86-64 builds without FMA code generation do.
   Everywhere else they evaluate the plain operator.
------------------------------------------------------------------------- */

namespace LAMMPS_NS::MathRoundedKokkos {

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION double add_rn(double a, double b)
{
#if defined(__CUDA_ARCH__)
  return __dadd_rn(a, b);
#else
  return a + b;
#endif
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION double sub_rn(double a, double b)
{
#if defined(__CUDA_ARCH__)
  return __dsub_rn(a, b);
#else
  return a - b;
#endif
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION double mul_rn(double a, double b)
{
#if defined(__CUDA_ARCH__)
  return __dmul_rn(a, b);
#else
  return a * b;
#endif
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION double div_rn(double a, double b)
{
#if defined(__CUDA_ARCH__)
  return __ddiv_rn(a, b);
#else
  return a / b;
#endif
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION float add_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
  return __fadd_rn(a, b);
#else
  return a + b;
#endif
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION float sub_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
  return __fsub_rn(a, b);
#else
  return a - b;
#endif
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION float mul_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION float div_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}

}    // namespace LAMMPS_NS::MathRoundedKokkos

#endif
