// clang-format off
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

/* ----------------------------------------------------------------------
   Contributing authors: Stan Moore (SNL)
                         Mitch Murphy (alphataubio@gmail.com)
------------------------------------------------------------------------- */

#include "fix_nh_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "compute.h"
#include "domain_kokkos.h"
#include "error.h"
#include "force.h"
#include "irregular.h"
#include "kokkos.h"
#include "kspace.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double DELTAFLIP = 0.1;
static constexpr double TILTMAX = 1.5;
static constexpr double EPSILON = 1.0e-6;

enum{NOBIAS,BIAS};
enum{NONE,XYZ,XY,YZ,XZ};
enum{ISO,ANISO,TRICLINIC};

/* ----------------------------------------------------------------------
   arithmetic of the device-resident thermostat chain

   The device chain update must reproduce the host update of
   FixNH::nhc_temp_integrate() bit for bit. Device code contracts a*b+c into
   a fused multiply-add unless told otherwise, so every operation is rounded
   on its own with the explicitly rounded CUDA intrinsics, as on the host.

   exp() follows the double-precision exp of the GNU C library (since
   version 2.28; the algorithm of Arm's optimized-routines): x = k ln2/128 + r,
   exp(x) = 2^(k/128) exp(r), with 2^(i/128) tabulated as scale and tail bits
   and exp(r) a degree-5 polynomial. The device evaluates it with the fused
   multiply-adds of the x86-64 FMA variant that glibc selects on processors
   with FMA, which reproduces the host result bit for bit on such hosts; the
   host itself calls the C library.
------------------------------------------------------------------------- */

namespace {

template<class DeviceType>
constexpr bool chain_device_supported()
{
#if defined(KOKKOS_ENABLE_CUDA)
  return std::is_same_v<typename DeviceType::execution_space, Kokkos::Cuda>;
#else
  return false;
#endif
}

// tail and scale bits of 2^(i/128), i = 0..127, interleaved; the chain
// update reads them from the constant cache

#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDACC__)
__constant__ uint64_t EXP_TABLE[256] = {
  0x0000000000000000ULL, 0x3ff0000000000000ULL, 0x3c9b3b4f1a88bf6eULL, 0x3feff63da9fb3335ULL,
  0xbc7160139cd8dc5dULL, 0x3fefec9a3e778061ULL, 0xbc905e7a108766d1ULL, 0x3fefe315e86e7f85ULL,
  0x3c8cd2523567f613ULL, 0x3fefd9b0d3158574ULL, 0xbc8bce8023f98efaULL, 0x3fefd06b29ddf6deULL,
  0x3c60f74e61e6c861ULL, 0x3fefc74518759bc8ULL, 0x3c90a3e45b33d399ULL, 0x3fefbe3ecac6f383ULL,
  0x3c979aa65d837b6dULL, 0x3fefb5586cf9890fULL, 0x3c8eb51a92fdeffcULL, 0x3fefac922b7247f7ULL,
  0x3c3ebe3d702f9cd1ULL, 0x3fefa3ec32d3d1a2ULL, 0xbc6a033489906e0bULL, 0x3fef9b66affed31bULL,
  0xbc9556522a2fbd0eULL, 0x3fef9301d0125b51ULL, 0xbc5080ef8c4eea55ULL, 0x3fef8abdc06c31ccULL,
  0xbc91c923b9d5f416ULL, 0x3fef829aaea92de0ULL, 0x3c80d3e3e95c55afULL, 0x3fef7a98c8a58e51ULL,
  0xbc801b15eaa59348ULL, 0x3fef72b83c7d517bULL, 0xbc8f1ff055de323dULL, 0x3fef6af9388c8deaULL,
  0x3c8b898c3f1353bfULL, 0x3fef635beb6fcb75ULL, 0xbc96d99c7611eb26ULL, 0x3fef5be084045cd4ULL,
  0x3c9aecf73e3a2f60ULL, 0x3fef54873168b9aaULL, 0xbc8fe782cb86389dULL, 0x3fef4d5022fcd91dULL,
  0x3c8a6f4144a6c38dULL, 0x3fef463b88628cd6ULL, 0x3c807a05b0e4047dULL, 0x3fef3f49917ddc96ULL,
  0x3c968efde3a8a894ULL, 0x3fef387a6e756238ULL, 0x3c875e18f274487dULL, 0x3fef31ce4fb2a63fULL,
  0x3c80472b981fe7f2ULL, 0x3fef2b4565e27cddULL, 0xbc96b87b3f71085eULL, 0x3fef24dfe1f56381ULL,
  0x3c82f7e16d09ab31ULL, 0x3fef1e9df51fdee1ULL, 0xbc3d219b1a6fbffaULL, 0x3fef187fd0dad990ULL,
  0x3c8b3782720c0ab4ULL, 0x3fef1285a6e4030bULL, 0x3c6e149289cecb8fULL, 0x3fef0cafa93e2f56ULL,
  0x3c834d754db0abb6ULL, 0x3fef06fe0a31b715ULL, 0x3c864201e2ac744cULL, 0x3fef0170fc4cd831ULL,
  0x3c8fdd395dd3f84aULL, 0x3feefc08b26416ffULL, 0xbc86a3803b8e5b04ULL, 0x3feef6c55f929ff1ULL,
  0xbc924aedcc4b5068ULL, 0x3feef1a7373aa9cbULL, 0xbc9907f81b512d8eULL, 0x3feeecae6d05d866ULL,
  0xbc71d1e83e9436d2ULL, 0x3feee7db34e59ff7ULL, 0xbc991919b3ce1b15ULL, 0x3feee32dc313a8e5ULL,
  0x3c859f48a72a4c6dULL, 0x3feedea64c123422ULL, 0xbc9312607a28698aULL, 0x3feeda4504ac801cULL,
  0xbc58a78f4817895bULL, 0x3feed60a21f72e2aULL, 0xbc7c2c9b67499a1bULL, 0x3feed1f5d950a897ULL,
  0x3c4363ed60c2ac11ULL, 0x3feece086061892dULL, 0x3c9666093b0664efULL, 0x3feeca41ed1d0057ULL,
  0x3c6ecce1daa10379ULL, 0x3feec6a2b5c13cd0ULL, 0x3c93ff8e3f0f1230ULL, 0x3feec32af0d7d3deULL,
  0x3c7690cebb7aafb0ULL, 0x3feebfdad5362a27ULL, 0x3c931dbdeb54e077ULL, 0x3feebcb299fddd0dULL,
  0xbc8f94340071a38eULL, 0x3feeb9b2769d2ca7ULL, 0xbc87deccdc93a349ULL, 0x3feeb6daa2cf6642ULL,
  0xbc78dec6bd0f385fULL, 0x3feeb42b569d4f82ULL, 0xbc861246ec7b5cf6ULL, 0x3feeb1a4ca5d920fULL,
  0x3c93350518fdd78eULL, 0x3feeaf4736b527daULL, 0x3c7b98b72f8a9b05ULL, 0x3feead12d497c7fdULL,
  0x3c9063e1e21c5409ULL, 0x3feeab07dd485429ULL, 0x3c34c7855019c6eaULL, 0x3feea9268a5946b7ULL,
  0x3c9432e62b64c035ULL, 0x3feea76f15ad2148ULL, 0xbc8ce44a6199769fULL, 0x3feea5e1b976dc09ULL,
  0xbc8c33c53bef4da8ULL, 0x3feea47eb03a5585ULL, 0xbc845378892be9aeULL, 0x3feea34634ccc320ULL,
  0xbc93cedd78565858ULL, 0x3feea23882552225ULL, 0x3c5710aa807e1964ULL, 0x3feea155d44ca973ULL,
  0xbc93b3efbf5e2228ULL, 0x3feea09e667f3bcdULL, 0xbc6a12ad8734b982ULL, 0x3feea012750bdabfULL,
  0xbc6367efb86da9eeULL, 0x3fee9fb23c651a2fULL, 0xbc80dc3d54e08851ULL, 0x3fee9f7df9519484ULL,
  0xbc781f647e5a3ecfULL, 0x3fee9f75e8ec5f74ULL, 0xbc86ee4ac08b7db0ULL, 0x3fee9f9a48a58174ULL,
  0xbc8619321e55e68aULL, 0x3fee9feb564267c9ULL, 0x3c909ccb5e09d4d3ULL, 0x3feea0694fde5d3fULL,
  0xbc7b32dcb94da51dULL, 0x3feea11473eb0187ULL, 0x3c94ecfd5467c06bULL, 0x3feea1ed0130c132ULL,
  0x3c65ebe1abd66c55ULL, 0x3feea2f336cf4e62ULL, 0xbc88a1c52fb3cf42ULL, 0x3feea427543e1a12ULL,
  0xbc9369b6f13b3734ULL, 0x3feea589994cce13ULL, 0xbc805e843a19ff1eULL, 0x3feea71a4623c7adULL,
  0xbc94d450d872576eULL, 0x3feea8d99b4492edULL, 0x3c90ad675b0e8a00ULL, 0x3feeaac7d98a6699ULL,
  0x3c8db72fc1f0eab4ULL, 0x3feeace5422aa0dbULL, 0xbc65b6609cc5e7ffULL, 0x3feeaf3216b5448cULL,
  0x3c7bf68359f35f44ULL, 0x3feeb1ae99157736ULL, 0xbc93091fa71e3d83ULL, 0x3feeb45b0b91ffc6ULL,
  0xbc5da9b88b6c1e29ULL, 0x3feeb737b0cdc5e5ULL, 0xbc6c23f97c90b959ULL, 0x3feeba44cbc8520fULL,
  0xbc92434322f4f9aaULL, 0x3feebd829fde4e50ULL, 0xbc85ca6cd7668e4bULL, 0x3feec0f170ca07baULL,
  0x3c71affc2b91ce27ULL, 0x3feec49182a3f090ULL, 0x3c6dd235e10a73bbULL, 0x3feec86319e32323ULL,
  0xbc87c50422622263ULL, 0x3feecc667b5de565ULL, 0x3c8b1c86e3e231d5ULL, 0x3feed09bec4a2d33ULL,
  0xbc91bbd1d3bcbb15ULL, 0x3feed503b23e255dULL, 0x3c90cc319cee31d2ULL, 0x3feed99e1330b358ULL,
  0x3c8469846e735ab3ULL, 0x3feede6b5579fdbfULL, 0xbc82dfcd978e9db4ULL, 0x3feee36bbfd3f37aULL,
  0x3c8c1a7792cb3387ULL, 0x3feee89f995ad3adULL, 0xbc907b8f4ad1d9faULL, 0x3feeee07298db666ULL,
  0xbc55c3d956dcaebaULL, 0x3feef3a2b84f15fbULL, 0xbc90a40e3da6f640ULL, 0x3feef9728de5593aULL,
  0xbc68d6f438ad9334ULL, 0x3feeff76f2fb5e47ULL, 0xbc91eee26b588a35ULL, 0x3fef05b030a1064aULL,
  0x3c74ffd70a5fddcdULL, 0x3fef0c1e904bc1d2ULL, 0xbc91bdfbfa9298acULL, 0x3fef12c25bd71e09ULL,
  0x3c736eae30af0cb3ULL, 0x3fef199bdd85529cULL, 0x3c8ee3325c9ffd94ULL, 0x3fef20ab5fffd07aULL,
  0x3c84e08fd10959acULL, 0x3fef27f12e57d14bULL, 0x3c63cdaf384e1a67ULL, 0x3fef2f6d9406e7b5ULL,
  0x3c676b2c6c921968ULL, 0x3fef3720dcef9069ULL, 0xbc808a1883ccb5d2ULL, 0x3fef3f0b555dc3faULL,
  0xbc8fad5d3ffffa6fULL, 0x3fef472d4a07897cULL, 0xbc900dae3875a949ULL, 0x3fef4f87080d89f2ULL,
  0x3c74a385a63d07a7ULL, 0x3fef5818dcfba487ULL, 0xbc82919e2040220fULL, 0x3fef60e316c98398ULL,
  0x3c8e5a50d5c192acULL, 0x3fef69e603db3285ULL, 0x3c843a59ac016b4bULL, 0x3fef7321f301b460ULL,
  0xbc82d52107b43e1fULL, 0x3fef7c97337b9b5fULL, 0xbc892ab93b470dc9ULL, 0x3fef864614f5a129ULL,
  0x3c74b604603a88d3ULL, 0x3fef902ee78b3ff6ULL, 0x3c83c5ec519d7271ULL, 0x3fef9a51fbc74c83ULL,
  0xbc8ff7128fd391f0ULL, 0x3fefa4afa2a490daULL, 0xbc8dae98e223747dULL, 0x3fefaf482d8e67f1ULL,
  0x3c8ec3bc41aa2008ULL, 0x3fefba1bee615a27ULL, 0x3c842b94c3a9eb32ULL, 0x3fefc52b376bba97ULL,
  0x3c8a64a931d185eeULL, 0x3fefd0765b6e4540ULL, 0xbc8e37bae43be3edULL, 0x3fefdbfdad9cbe14ULL,
  0x3c77893b4d91cd9dULL, 0x3fefe7c1819e90d8ULL, 0x3c5305c14160cc89ULL, 0x3feff3c22b8f71f1ULL};
#endif

KOKKOS_INLINE_FUNCTION double nh_add(double a, double b)
{
#if defined(__CUDA_ARCH__)
  return __dadd_rn(a, b);
#else
  return a + b;
#endif
}

KOKKOS_INLINE_FUNCTION double nh_sub(double a, double b)
{
#if defined(__CUDA_ARCH__)
  return __dsub_rn(a, b);
#else
  return a - b;
#endif
}

KOKKOS_INLINE_FUNCTION double nh_mul(double a, double b)
{
#if defined(__CUDA_ARCH__)
  return __dmul_rn(a, b);
#else
  return a * b;
#endif
}

KOKKOS_INLINE_FUNCTION double nh_div(double a, double b)
{
#if defined(__CUDA_ARCH__)
  return __ddiv_rn(a, b);
#else
  return a / b;
#endif
}

KOKKOS_INLINE_FUNCTION double nh_exp(double x)
{
#if defined(__CUDA_ARCH__)
  const uint64_t *table = EXP_TABLE;
  constexpr double invln2n = 0x1.71547652b82fep+7;      // 128/ln2
  constexpr double shift = 0x1.8p52;
  constexpr double negln2hin = -0x1.62e42fefa0000p-8;   // -ln2/128, high part
  constexpr double negln2lon = -0x1.cf79abc9e3b3ap-47;  // -ln2/128, low part
  constexpr double c2 = 0x1.ffffffffffdbdp-2;
  constexpr double c3 = 0x1.555555555543cp-3;
  constexpr double c4 = 0x1.55555cf172b91p-5;
  constexpr double c5 = 0x1.1111167a4d017p-7;

  const uint64_t ix = static_cast<uint64_t>(__double_as_longlong(x));
  uint32_t abstop = static_cast<uint32_t>(ix >> 52) & 0x7ff;

  // |x| < 2^-54 (including zero), |x| >= 512, infinity and NaN

  if (abstop - 0x3c9u > 0x3eu) {
    if (static_cast<int32_t>(abstop - 0x3c9u) < 0) return __dadd_rn(x, 1.0);
    if (abstop > 0x408u) {
      if (ix == 0xfff0000000000000ULL) return 0.0;
      if (abstop == 0x7ffu) return __dadd_rn(x, 1.0);
      return (ix >> 63) ? 0.0 : __longlong_as_double(0x7ff0000000000000LL);
    }
    abstop = 0;    // 512 <= |x| < 1024: scale may leave the normal range
  }

  // k = round(x 128/ln2), r = x - k ln2/128, exp(x) = scale (1 + tmp)

  double kd = __fma_rn(x, invln2n, shift);
  const uint64_t ki = static_cast<uint64_t>(__double_as_longlong(kd));
  kd = __dsub_rn(kd, shift);
  double r = __fma_rn(kd, negln2hin, x);
  r = __fma_rn(kd, negln2lon, r);
  const uint64_t idx = 2 * (ki & 0x7f);
  uint64_t sbits = table[idx + 1] + (ki << 45);
  const double rtail = __dadd_rn(r, __longlong_as_double(static_cast<long long>(table[idx])));
  const double r2 = __dmul_rn(r, r);
  const double p23 = __fma_rn(r, c3, c2);
  const double p45 = __fma_rn(r, c5, c4);
  const double tmp = __fma_rn(__dmul_rn(r2, r2), p45, __fma_rn(p23, r2, rtail));

  if (abstop == 0) {
    if ((ki & 0x80000000ULL) == 0) {
      // k > 0: the exponent of scale may overflow, so scale by 2^1009 last
      sbits -= 1009ULL << 52;
      const double scale = __longlong_as_double(static_cast<long long>(sbits));
      return __dmul_rn(__fma_rn(scale, tmp, scale), 0x1p1009);
    }
    // k < 0: round in the normal range before scaling into the subnormal one
    sbits += 1022ULL << 52;
    const double scale = __longlong_as_double(static_cast<long long>(sbits));
    const double scaled_tmp = __dmul_rn(tmp, scale);
    double y = __dadd_rn(scale, scaled_tmp);
    if (1.0 > y) {
      const double hi = __dadd_rn(y, 1.0);
      double lo = __dadd_rn(__dsub_rn(scale, y), scaled_tmp);
      lo = __dadd_rn(__dadd_rn(__dsub_rn(1.0, hi), y), lo);
      y = __dsub_rn(__dadd_rn(lo, hi), 1.0);
      if (y == 0.0) return 0.0;
    }
    return __dmul_rn(y, 0x1p-1022);
  }
  const double scale = __longlong_as_double(static_cast<long long>(sbits));
  return __fma_rn(scale, tmp, scale);
#else
  return exp(x);
#endif
}

}    // namespace

/* ----------------------------------------------------------------------
   NVT,NPH,NPT integrators for improved Nose-Hoover equations of motion
 ---------------------------------------------------------------------- */

template<class DeviceType>
FixNHKokkos<DeviceType>::FixNHKokkos(LAMMPS *lmp, int narg, char **arg) : FixNH(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *)atom;
  domainKK = (DomainKokkos *) domain;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  chain_on_device = chain_device_newer = chain_refresh = false;
  chain_tfactor = chain_kecoef = chain_kt = chain_ncfac = 0.0;
  chain_dt8_arg = chain_dthalf_arg = chain_eta_step = 0.0;
  for (double &mass_ich : chain_mass) mass_ich = 0.0;
  temperatureKK = nullptr;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::init()
{
  FixNH::init();

  // warn if the temperature compute is not a KOKKOS style (e.g. set via
  // fix_modify temp to a non-kk compute): correct but forces per-step syncs.
  // the pressure compute has no /kk variant, so it is not checked.

  KokkosLMP::warn_nonkokkos_compute(lmp, style, temperature, "temperature");

  atomKK->k_mass.modify_host();
  atomKK->k_mass.sync<DeviceType>();

  // the chain stays on the device when no step of a thermostat update needs
  // the host: no barostat, no velocity bias, no spin velocities, and a
  // temperature whose scalar needs neither an MPI sum nor a recount

  temperatureKK = dynamic_cast<ComputeTempKokkos<DeviceType> *>(temperature);
  chain_on_device = chain_device_supported<DeviceType>() && tstat_flag && !pstat_flag &&
    which == NOBIAS && !spin_flag && mtchain <= CHAIN_MAX && temperatureKK &&
    temperatureKK->scalar_on_device();

  if (chain_on_device && !d_eta.data()) {
    d_kinetic = decltype(d_kinetic)("nh/kk:kinetic");
    d_t_current = decltype(d_t_current)("nh/kk:t_current");
    d_eta = decltype(d_eta)("nh/kk:eta",mtchain);
    d_eta_dot = decltype(d_eta_dot)("nh/kk:eta_dot",mtchain);
    d_eta_dotdot = decltype(d_eta_dotdot)("nh/kk:eta_dotdot",mtchain);
    d_factor_eta = decltype(d_factor_eta)("nh/kk:factor_eta",nc_tchain);
  }
}

/* ----------------------------------------------------------------------
   compute T,P before integrator starts
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::setup(int /*vflag*/)
{
  // tdof needed by compute_temp_target()

  if (temperature->kokkosable)
    t_current = temperature->compute_scalar();
  else {
    atomKK->sync(temperature->execution_space,temperature->datamask_read);
    t_current = temperature->compute_scalar();
    atomKK->modified(temperature->execution_space,temperature->datamask_modify);
    atomKK->sync(execution_space,temperature->datamask_modify);
  }

  tdof = temperature->dof;

  // t_target is needed by NPH and NPT in compute_scalar()
  // If no thermostat or using fix nphug,
  // t_target must be defined by other means.

  if (tstat_flag && strcmp(style,"nphug") != 0) {
    compute_temp_target();
  } else if (pstat_flag) {

    // t0 = reference temperature for masses
    // set equal to either ptemp or the current temperature
    // cannot be done in init() b/c temperature cannot be called there
    // is b/c Modify::init() inits computes after fixes due to dof dependence
    // error if T less than 1e-6
    // if it was read in from a restart file, leave it be

    if (t0 == 0.0) {
      if (p_temp_flag) {
        t0 = p_temp;
      } else {
        if (temperature->kokkosable)
          t0 = temperature->compute_scalar();
        else {
          atomKK->sync(temperature->execution_space,temperature->datamask_read);
          t0 = temperature->compute_scalar();
          atomKK->modified(temperature->execution_space,temperature->datamask_modify);
          atomKK->sync(execution_space,temperature->datamask_modify);
        }
        if (t0 < EPSILON)
          error->all(FLERR,"Current temperature too close to zero, consider using ptemp keyword");
      }
    }
    t_target = t0;
  }

  if (pstat_flag) compute_press_target();

  if (temperature->kokkosable)
    t_current = temperature->compute_scalar();
  else {
    atomKK->sync(temperature->execution_space,temperature->datamask_read);
    t_current = temperature->compute_scalar();
    atomKK->modified(temperature->execution_space,temperature->datamask_modify);
    atomKK->sync(execution_space,temperature->datamask_modify);
  }

  tdof = temperature->dof;

  if (pstat_flag) {
    //atomKK->sync(pressure->execution_space,pressure->datamask_read);
    //atomKK->modified(pressure->execution_space,pressure->datamask_modify);
    if (pstyle == ISO) pressure->compute_scalar();
    else pressure->compute_vector();
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  // masses and initial forces on thermostat variables

  if (tstat_flag) {
    eta_mass[0] = tdof * boltz * t_target / (t_freq*t_freq);
    for (int ich = 1; ich < mtchain; ich++)
      eta_mass[ich] = boltz * t_target / (t_freq*t_freq);
    for (int ich = 1; ich < mtchain; ich++) {
      eta_dotdot[ich] = (eta_mass[ich-1]*eta_dot[ich-1]*eta_dot[ich-1] -
                         boltz * t_target) / eta_mass[ich];
    }
  }

  // masses and initial forces on barostat variables

  if (pstat_flag) {
    double kt = boltz * t_target;
    double nkt = (atom->natoms + 1) * kt;

    for (int i = 0; i < 3; i++)
      if (p_flag[i])
        omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);

    if (pstyle == TRICLINIC) {
      for (int i = 3; i < 6; i++)
        if (p_flag[i]) omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);
    }

  // masses and initial forces on barostat thermostat variables

    if (mpchain) {
      etap_mass[0] = boltz * t_target / (p_freq_max*p_freq_max);
      for (int ich = 1; ich < mpchain; ich++)
        etap_mass[ich] = boltz * t_target / (p_freq_max*p_freq_max);
      for (int ich = 1; ich < mpchain; ich++)
        etap_dotdot[ich] =
          (etap_mass[ich-1]*etap_dot[ich-1]*etap_dot[ich-1] -
           boltz * t_target) / etap_mass[ich];
    }

  }

  if (chain_on_device) upload_chain();
}

/* ----------------------------------------------------------------------
   1st half of Verlet update
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  // update eta_press_dot

  if (pstat_flag && mpchain) nhc_press_integrate();

  // update eta_dot

  if (tstat_flag) {
    compute_temp_target();
    if (chain_on_device) nhc_temp_integrate_device(false);
    else nhc_temp_integrate();
  }

  // need to recompute pressure to account for change in KE
  // t_current is up-to-date, but compute_temperature is not
  // compute appropriately coupled elements of mvv_current

  if (pstat_flag) {
    if (!temperature->kokkosable)
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
    if (!pressure->kokkosable)
      atomKK->sync(pressure->execution_space,pressure->datamask_read);
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    if (!temperature->kokkosable) {
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
    }
    if (!pressure->kokkosable) {
      atomKK->modified(pressure->execution_space,pressure->datamask_modify);
      atomKK->sync(execution_space,pressure->datamask_modify);
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) {
    compute_press_target();
    nh_omega_dot();
    nh_v_press();
  }

  nve_v();

  // remap simulation box by 1/2 step

  if (pstat_flag) remap();

  nve_x();

  // remap simulation box by 1/2 step
  // redo KSpace coeffs since volume has changed

  if (pstat_flag) {
    remap();
    if (kspace_flag) force->kspace->setup();
  }
}

/* ----------------------------------------------------------------------
   2nd half of Verlet update
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::final_integrate()
{
  nve_v();

  // re-compute temp before nh_v_press()
  // only needed for temperature computes with BIAS on reneighboring steps:
  //   b/c some biases store per-atom values (e.g. temp/profile)
  //   per-atom values are invalid if reneigh/comm occurred
  //     since temp->compute() in initial_integrate()

  if (which == BIAS && neighbor->ago == 0) {
    if (temperature->kokkosable)
      t_current = temperature->compute_scalar();
    else {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      t_current = temperature->compute_scalar();
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
    }
  }

  if (pstat_flag) nh_v_press();

  // compute new T,P after velocities rescaled by nh_v_press()
  // compute appropriately coupled elements of mvv_current

  if (chain_on_device)
    chain_tfactor = temperatureKK->compute_scalar_device(d_kinetic);
  else if (temperature->kokkosable)
    t_current = temperature->compute_scalar();
  else {
    atomKK->sync(temperature->execution_space,temperature->datamask_read);
    t_current = temperature->compute_scalar();
    atomKK->modified(temperature->execution_space,temperature->datamask_modify);
    atomKK->sync(execution_space,temperature->datamask_modify);
  }
  tdof = temperature->dof;

  if (pstat_flag) {
    if (pstyle == ISO) {
      if (!pressure->kokkosable)
        atomKK->sync(pressure->execution_space,pressure->datamask_read);
      pressure->compute_scalar();
      if (!pressure->kokkosable) {
        atomKK->modified(pressure->execution_space,pressure->datamask_modify);
        atomKK->sync(execution_space,pressure->datamask_modify);
      }
    } else {
      if (!temperature->kokkosable)
        atomKK->sync(temperature->execution_space,temperature->datamask_read);
      if (!pressure->kokkosable)
        atomKK->sync(pressure->execution_space,pressure->datamask_read);
      temperature->compute_vector();
      pressure->compute_vector();
      if (!temperature->kokkosable) {
        atomKK->modified(temperature->execution_space,temperature->datamask_modify);
        atomKK->sync(execution_space,temperature->datamask_modify);
      }
      if (!pressure->kokkosable) {
        atomKK->modified(pressure->execution_space,pressure->datamask_modify);
        atomKK->sync(execution_space,pressure->datamask_modify);
      }
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) nh_omega_dot();

  // update eta_dot
  // update eta_press_dot

  if (tstat_flag) {
    if (chain_on_device) nhc_temp_integrate_device(true);
    else nhc_temp_integrate();
  }
  if (pstat_flag && mpchain) nhc_press_integrate();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::post_run()
{
  sync_chain_host();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::write_restart(FILE *fp)
{
  sync_chain_host();
  FixNH::write_restart(fp);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double FixNHKokkos<DeviceType>::compute_scalar()
{
  sync_chain_host();
  return FixNH::compute_scalar();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double FixNHKokkos<DeviceType>::compute_vector(int n)
{
  sync_chain_host();
  return FixNH::compute_vector(n);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void *FixNHKokkos<DeviceType>::extract(const char *str, int &dim)
{
  sync_chain_host();
  return FixNH::extract(str,dim);
}

/* ----------------------------------------------------------------------
   half-step update of the thermostat chain on the device, followed by the
   velocity scaling it determines; refresh starts the update from the
   kinetic sum that compute_scalar_device() reduced into d_kinetic
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::nhc_temp_integrate_device(bool refresh)
{
  chain_refresh = refresh;

  // the masses depend on host data only, so the host keeps them

  if (eta_mass_flag) {
    eta_mass[0] = tdof * boltz * t_target / (t_freq*t_freq);
    for (int ich = 1; ich < mtchain; ich++)
      eta_mass[ich] = boltz * t_target / (t_freq*t_freq);
  }
  for (int ich = 0; ich < mtchain; ich++) chain_mass[ich] = eta_mass[ich];
  chain_kecoef = tdof * boltz;
  chain_kt = boltz * t_target;
  chain_ncfac = 1.0/nc_tchain;
  chain_dt8_arg = -chain_ncfac*dt8;
  chain_dthalf_arg = -chain_ncfac*dthalf;
  chain_eta_step = chain_ncfac*dthalf;

  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  atomKK->sync(execution_space,V_MASK | MASK_MASK);

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNH_nhc_temp>(0,1),*this);
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNH_nh_v_temp_chain>(0,nlocal),*this);
  copymode = 0;

  atomKK->modified(execution_space,V_MASK);
  chain_device_newer = true;
}

/* ----------------------------------------------------------------------
   FixNH::nhc_temp_integrate() for one chain update in a single thread,
   operation for operation, with its host-only quantities precomputed in
   nhc_temp_integrate_device(); the velocity scaling of each chain loop is
   recorded in d_factor_eta and applied by TagFixNH_nh_v_temp_chain, since
   the chain update itself only tracks the temperature through the factors
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNHKokkos<DeviceType>::operator()(TagFixNH_nhc_temp, const int &) const {
  // the chain state is read once and kept in thread-local storage, so the
  // dependent operations of the update do not wait on device memory; the
  // extra dummy thermostat eta_dot[mtchain] is zero

  double eta_l[CHAIN_MAX], eta_dot_l[CHAIN_MAX+1], eta_dotdot_l[CHAIN_MAX];
  for (int ich = 0; ich < mtchain; ich++) {
    eta_l[ich] = d_eta(ich);
    eta_dot_l[ich] = d_eta_dot(ich);
    eta_dotdot_l[ich] = d_eta_dotdot(ich);
  }
  eta_dot_l[mtchain] = 0.0;

  double t = chain_refresh ? nh_mul(d_kinetic().t0,chain_tfactor) : d_t_current();
  double kecurrent = nh_mul(chain_kecoef,t);

  if (chain_mass[0] > 0.0)
    eta_dotdot_l[0] = nh_div(nh_sub(kecurrent,ke_target),chain_mass[0]);
  else eta_dotdot_l[0] = 0.0;

  double expfac;
  for (int iloop = 0; iloop < nc_tchain; iloop++) {

    for (int ich = mtchain-1; ich > 0; ich--) {
      expfac = nh_exp(nh_mul(chain_dt8_arg,eta_dot_l[ich+1]));
      eta_dot_l[ich] = nh_mul(eta_dot_l[ich],expfac);
      eta_dot_l[ich] = nh_add(eta_dot_l[ich],nh_mul(nh_mul(eta_dotdot_l[ich],chain_ncfac),dt4));
      eta_dot_l[ich] = nh_mul(eta_dot_l[ich],tdrag_factor);
      eta_dot_l[ich] = nh_mul(eta_dot_l[ich],expfac);
    }

    expfac = nh_exp(nh_mul(chain_dt8_arg,eta_dot_l[1]));
    eta_dot_l[0] = nh_mul(eta_dot_l[0],expfac);
    eta_dot_l[0] = nh_add(eta_dot_l[0],nh_mul(nh_mul(eta_dotdot_l[0],chain_ncfac),dt4));
    eta_dot_l[0] = nh_mul(eta_dot_l[0],tdrag_factor);
    eta_dot_l[0] = nh_mul(eta_dot_l[0],expfac);

    const double factor_eta_loop = nh_exp(nh_mul(chain_dthalf_arg,eta_dot_l[0]));
    d_factor_eta(iloop) = factor_eta_loop;

    // rescale temperature due to velocity scaling

    t = nh_mul(t,nh_mul(factor_eta_loop,factor_eta_loop));
    kecurrent = nh_mul(chain_kecoef,t);

    if (chain_mass[0] > 0.0)
      eta_dotdot_l[0] = nh_div(nh_sub(kecurrent,ke_target),chain_mass[0]);
    else eta_dotdot_l[0] = 0.0;

    for (int ich = 0; ich < mtchain; ich++)
      eta_l[ich] = nh_add(eta_l[ich],nh_mul(chain_eta_step,eta_dot_l[ich]));

    eta_dot_l[0] = nh_mul(eta_dot_l[0],expfac);
    eta_dot_l[0] = nh_add(eta_dot_l[0],nh_mul(nh_mul(eta_dotdot_l[0],chain_ncfac),dt4));
    eta_dot_l[0] = nh_mul(eta_dot_l[0],expfac);

    for (int ich = 1; ich < mtchain; ich++) {
      expfac = nh_exp(nh_mul(chain_dt8_arg,eta_dot_l[ich+1]));
      eta_dot_l[ich] = nh_mul(eta_dot_l[ich],expfac);
      eta_dotdot_l[ich] =
        nh_div(nh_sub(nh_mul(nh_mul(chain_mass[ich-1],eta_dot_l[ich-1]),eta_dot_l[ich-1]),
                      chain_kt),chain_mass[ich]);
      eta_dot_l[ich] = nh_add(eta_dot_l[ich],nh_mul(nh_mul(eta_dotdot_l[ich],chain_ncfac),dt4));
      eta_dot_l[ich] = nh_mul(eta_dot_l[ich],expfac);
    }
  }

  for (int ich = 0; ich < mtchain; ich++) {
    d_eta(ich) = eta_l[ich];
    d_eta_dot(ich) = eta_dot_l[ich];
    d_eta_dotdot(ich) = eta_dotdot_l[ich];
  }
  d_t_current() = t;
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNHKokkos<DeviceType>::operator()(TagFixNH_nh_v_temp_chain, const int &i) const {
  if (mask[i] & groupbit) {
    for (int iloop = 0; iloop < nc_tchain; iloop++) {
      const KK_FLOAT factor_eta_kk = static_cast<KK_FLOAT>(d_factor_eta(iloop));
      v(i,0) *= factor_eta_kk;
      v(i,1) *= factor_eta_kk;
      v(i,2) *= factor_eta_kk;
    }
  }
}

/* ----------------------------------------------------------------------
   copy the host chain state to the device, which owns it during a run
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::upload_chain()
{
  using HostView = Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(d_eta,HostView(eta,mtchain));
  Kokkos::deep_copy(d_eta_dot,HostView(eta_dot,mtchain));
  Kokkos::deep_copy(d_eta_dotdot,HostView(eta_dotdot,mtchain));
  Kokkos::deep_copy(d_t_current,t_current);
  chain_device_newer = false;
}

/* ----------------------------------------------------------------------
   bring the host chain state up to date with the device, if it is behind
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::sync_chain_host()
{
  if (!chain_device_newer) return;
  using HostView = Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(HostView(eta,mtchain),d_eta);
  Kokkos::deep_copy(HostView(eta_dot,mtchain),d_eta_dot);
  Kokkos::deep_copy(HostView(eta_dotdot,mtchain),d_eta_dotdot);
  Kokkos::deep_copy(t_current,d_t_current);
  chain_device_newer = false;
}

/* ----------------------------------------------------------------------
   change box size
   remap all atoms or dilate group atoms depending on allremap flag
   if rigid bodies exist, scale rigid body centers-of-mass
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::remap()
{
  double oldlo,oldhi;
  double expfac;
  double isofac;

  int nlocal = atom->nlocal;
  double *h = domain->h;

  // omega is not used, except for book-keeping

  for (int i = 0; i < 6; i++) omega[i] += dto*omega_dot[i];

  // convert pertinent atoms and rigid bodies to lamda coords

  // the group variants of x2lamda()/lamda2x() run on the device and do their
  // own sync/modified, so no host round trip of x is needed here

  if (allremap) domainKK->x2lamda(nlocal);
  else domainKK->x2lamda(nlocal,dilate_group_bit);

  if (!rfix.empty())
    error->all(FLERR,"Cannot (yet) use rigid bodies with fix nh and Kokkos");
  // for (auto &ifix : rfix) ifix->deform(0);

  // reset global and local box to new size/shape

  // this operation corresponds to applying the
  // translate and scale operations
  // corresponding to the solution of the following ODE:
  //
  // h_dot = omega_dot * h
  //
  // where h_dot, omega_dot and h are all upper-triangular
  // 3x3 tensors. In Voigt notation, the elements of the
  // RHS product tensor are:
  // h_dot = [0*0, 1*1, 2*2, 1*3+3*2, 0*4+5*3+4*2, 0*5+5*1]
  //
  // Ordering of operations preserves time symmetry.

  double dto2 = dto/2.0;
  double dto4 = dto/4.0;
  double dto8 = dto/8.0;

  // off-diagonal components, first half

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dto2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dto2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }
  }

  // scale diagonal components
  // scale tilt factors with cell, if set

  if (isochoric) isofac = vol_start;

  if (p_flag[0]) {
    oldlo = domain->boxlo[0];
    oldhi = domain->boxhi[0];
    expfac = exp(dto*omega_dot[0]);
    domain->boxlo[0] = (oldlo-fixedpoint[0])*expfac + fixedpoint[0];
    domain->boxhi[0] = (oldhi-fixedpoint[0])*expfac + fixedpoint[0];
    if (isochoric) isofac /= domain->boxhi[0] - domain->boxlo[0];
  }

  if (p_flag[1]) {
    oldlo = domain->boxlo[1];
    oldhi = domain->boxhi[1];
    expfac = exp(dto*omega_dot[1]);
    domain->boxlo[1] = (oldlo-fixedpoint[1])*expfac + fixedpoint[1];
    domain->boxhi[1] = (oldhi-fixedpoint[1])*expfac + fixedpoint[1];
    if (isochoric) isofac /= domain->boxhi[1] - domain->boxlo[1];
    if (scalexy) h[5] *= expfac;
  }

  if (p_flag[2]) {
    oldlo = domain->boxlo[2];
    oldhi = domain->boxhi[2];
    expfac = exp(dto*omega_dot[2]);
    domain->boxlo[2] = (oldlo-fixedpoint[2])*expfac + fixedpoint[2];
    domain->boxhi[2] = (oldhi-fixedpoint[2])*expfac + fixedpoint[2];
    if (isochoric) isofac /= domain->boxhi[2] - domain->boxlo[2];
    if (scalexz) h[4] *= expfac;
    if (scaleyz) h[3] *= expfac;
  }

  // isochoric dimensions: rescale so the reference volume is preserved
  // pure box arithmetic, identical to the CPU base class

  if (isochoric) {

    // We remove remaining dimensions so that only scale factors are left
    // in isofac

    for (int i = 0; i < 3; i++) {
      if (p_isoch[i] || !p_flag[i]) isofac /= (domain->boxhi[i]-domain->boxlo[i]);
    }
    int iso_sum = p_isoch[0] + p_isoch[1] + p_isoch[2];
    if (iso_sum == 2) isofac = sqrt(isofac);

    if (p_isoch[0]) {
      oldlo = domain->boxlo[0];
      oldhi = domain->boxhi[0];
      domain->boxlo[0] = (oldlo-fixedpoint[0])*isofac + fixedpoint[0];
      domain->boxhi[0] = (oldhi-fixedpoint[0])*isofac + fixedpoint[0];
    }
    if (p_isoch[1]) {
      oldlo = domain->boxlo[1];
      oldhi = domain->boxhi[1];
      domain->boxlo[1] = (oldlo-fixedpoint[1])*isofac + fixedpoint[1];
      domain->boxhi[1] = (oldhi-fixedpoint[1])*isofac + fixedpoint[1];
      if (scalexy) h[5] *= isofac;
    }
    if (p_isoch[2]) {
      oldlo = domain->boxlo[2];
      oldhi = domain->boxhi[2];
      domain->boxlo[2] = (oldlo-fixedpoint[2])*isofac + fixedpoint[2];
      domain->boxhi[2] = (oldhi-fixedpoint[2])*isofac + fixedpoint[2];
      if (scalexz) h[4] *= isofac;
      if (scaleyz) h[3] *= isofac;
    }
  }

  // off-diagonal components, second half

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dto2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dto2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

  }

  domain->yz = h[3];
  domain->xz = h[4];
  domain->xy = h[5];

  // tilt factor to cell length ratio can not exceed TILTMAX in one step

  if (domain->yz < -TILTMAX*domain->yprd ||
      domain->yz > TILTMAX*domain->yprd ||
      domain->xz < -TILTMAX*domain->xprd ||
      domain->xz > TILTMAX*domain->xprd ||
      domain->xy < -TILTMAX*domain->xprd ||
      domain->xy > TILTMAX*domain->xprd)
    error->all(FLERR,"Fix npt/nph has tilted box too far in one step - "
               "periodic cell is too far from equilibrium state");

  domain->set_global_box();
  domain->set_local_box();

  // convert pertinent atoms and rigid bodies back to box coords

  if (allremap) domainKK->lamda2x(nlocal);
  else domainKK->lamda2x(nlocal,dilate_group_bit);

  // for (auto &ifix : rfix) ifix->deform(1);
}

/* ----------------------------------------------------------------------
   perform half-step barostat scaling of velocities
-----------------------------------------------------------------------*/

template<class DeviceType>
void FixNHKokkos<DeviceType>::nh_v_press()
{
  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  factor[0] = static_cast<KK_FLOAT>(exp(-dt4*(omega_dot[0]+mtk_term2)));
  factor[1] = static_cast<KK_FLOAT>(exp(-dt4*(omega_dot[1]+mtk_term2)));
  factor[2] = static_cast<KK_FLOAT>(exp(-dt4*(omega_dot[2]+mtk_term2)));

  if (which == BIAS) {
    if (temperature->kokkosable) temperature->remove_bias_all_kk();
    else {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      temperature->remove_bias_all();
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
    }
  }

  atomKK->sync(execution_space,V_MASK | MASK_MASK);

  copymode = 1;
  if (pstyle == TRICLINIC)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNH_nh_v_press<1> >(0,nlocal),*this);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNH_nh_v_press<0> >(0,nlocal),*this);
  copymode = 0;

  atomKK->modified(execution_space,V_MASK);

  if (which == BIAS) {
    if (temperature->kokkosable) temperature->restore_bias_all_kk();
    else {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      temperature->restore_bias_all();
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
    }
  }
}

template<class DeviceType>
template<int TRICLINIC_FLAG>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNHKokkos<DeviceType>::operator()(TagFixNH_nh_v_press<TRICLINIC_FLAG>, const int &i) const {
  if (mask[i] & groupbit) {
    v(i,0) *= factor[0];
    v(i,1) *= factor[1];
    v(i,2) *= factor[2];
    if (TRICLINIC_FLAG) {
      const KK_FLOAT dthalf_kk = static_cast<KK_FLOAT>(dthalf);
      const KK_FLOAT omega_dot3_kk = static_cast<KK_FLOAT>(omega_dot[3]);
      const KK_FLOAT omega_dot4_kk = static_cast<KK_FLOAT>(omega_dot[4]);
      const KK_FLOAT omega_dot5_kk = static_cast<KK_FLOAT>(omega_dot[5]);
      v(i,0) += -dthalf_kk*(v(i,1)*omega_dot5_kk + v(i,2)*omega_dot4_kk);
      v(i,1) += -dthalf_kk*v(i,2)*omega_dot3_kk;
    }
    v(i,0) *= factor[0];
    v(i,1) *= factor[1];
    v(i,2) *= factor[2];
  }
}

/* ----------------------------------------------------------------------
   perform half-step update of velocities
-----------------------------------------------------------------------*/

template<class DeviceType>
void FixNHKokkos<DeviceType>::nve_v()
{
  atomKK->sync(execution_space,X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);

  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  copymode = 1;
  if (rmass.data())
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNH_nve_v<1> >(0,nlocal),*this);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNH_nve_v<0> >(0,nlocal),*this);
  copymode = 0;

  atomKK->modified(execution_space,V_MASK);
}

template<class DeviceType>
template<int RMASS>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNHKokkos<DeviceType>::operator()(TagFixNH_nve_v<RMASS>, const int &i) const {
  const KK_FLOAT dtf_kk = static_cast<KK_FLOAT>(dtf);
  if (RMASS) {
    if (mask[i] & groupbit) {
      const KK_FLOAT dtfm = dtf_kk / rmass[i];
      v(i,0) += dtfm*static_cast<KK_FLOAT>(f(i,0));
      v(i,1) += dtfm*static_cast<KK_FLOAT>(f(i,1));
      v(i,2) += dtfm*static_cast<KK_FLOAT>(f(i,2));
    }
  } else {
    if (mask[i] & groupbit) {
      const KK_FLOAT dtfm = dtf_kk / mass[type[i]];
      v(i,0) += dtfm*static_cast<KK_FLOAT>(f(i,0));
      v(i,1) += dtfm*static_cast<KK_FLOAT>(f(i,1));
      v(i,2) += dtfm*static_cast<KK_FLOAT>(f(i,2));
    }
  }
}

/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

template<class DeviceType>
void FixNHKokkos<DeviceType>::nve_x()
{
  atomKK->sync(execution_space,X_MASK | V_MASK | MASK_MASK);

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  // x update by full step only for atoms in group

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNH_nve_x>(0,nlocal),*this);
  copymode = 0;

  // claim the coordinates after the kernel has written them, not before: a
  // claim made up front can be taken by a copy that runs in between, which
  // leaves the coordinates this kernel writes unclaimed

  atomKK->modified(execution_space,X_MASK);
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNHKokkos<DeviceType>::operator()(TagFixNH_nve_x, const int &i) const {
  const KK_FLOAT dtv_kk = static_cast<KK_FLOAT>(dtv);
  if (mask[i] & groupbit) {
    x(i,0) += dtv_kk * v(i,0);
    x(i,1) += dtv_kk * v(i,1);
    x(i,2) += dtv_kk * v(i,2);
  }
}

/* ----------------------------------------------------------------------
   perform half-step thermostat scaling of velocities
-----------------------------------------------------------------------*/

template<class DeviceType>
void FixNHKokkos<DeviceType>::nh_v_temp()
{
  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  if (which == BIAS) {
    if (temperature->kokkosable) temperature->remove_bias_all_kk();
    else {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      temperature->remove_bias_all();
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
    }
  }

  atomKK->sync(execution_space,V_MASK | MASK_MASK);

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNH_nh_v_temp>(0,nlocal),*this);
  copymode = 0;

  atomKK->modified(execution_space,V_MASK);

  if (which == BIAS) {
    if (temperature->kokkosable) temperature->restore_bias_all_kk();
    else {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      temperature->restore_bias_all();
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
    }
  }
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixNHKokkos<DeviceType>::operator()(TagFixNH_nh_v_temp, const int &i) const {
  const KK_FLOAT factor_eta_kk = static_cast<KK_FLOAT>(factor_eta);
  if (mask[i] & groupbit) {
    v(i,0) *= factor_eta_kk;
    v(i,1) *= factor_eta_kk;
    v(i,2) *= factor_eta_kk;
  }
}

/* ----------------------------------------------------------------------
  if any tilt ratios exceed limits, set flip = 1 and compute new tilt values
  do not flip in x or y if non-periodic (can tilt but not flip)
    this is b/c the box length would be changed (dramatically) by flip
  if yz tilt exceeded, adjust C vector by one B vector
  if xz tilt exceeded, adjust C vector by one A vector
  if xy tilt exceeded, adjust B vector by one A vector
  check yz first since it may change xz, then xz check comes after
  if any flip occurs, create new box in domain
  image_flip() adjusts image flags due to box shape change induced by flip
  remap() puts atoms outside the new box back into the new box
  perform irregular on atoms in lamda coords to migrate atoms to new procs
  important that image_flip comes before remap, since remap may change
    image flags to new values, making eqs in doc of Domain:image_flip incorrect
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::pre_exchange()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;

  // flip is only triggered when tilt exceeds 0.5 by DELTAFLIP
  // this avoids immediate re-flipping due to tilt oscillations

  double xtiltmax = (0.5+DELTAFLIP)*xprd;
  double ytiltmax = (0.5+DELTAFLIP)*yprd;

  int flipxy,flipxz,flipyz;
  flipxy = flipxz = flipyz = 0;

  if (domain->yperiodic) {
    if (domain->yz < -ytiltmax) {
      domain->yz += yprd;
      domain->xz += domain->xy;
      flipyz = 1;
    } else if (domain->yz >= ytiltmax) {
      domain->yz -= yprd;
      domain->xz -= domain->xy;
      flipyz = -1;
    }
  }

  if (domain->xperiodic) {
    if (domain->xz < -xtiltmax) {
      domain->xz += xprd;
      flipxz = 1;
    } else if (domain->xz >= xtiltmax) {
      domain->xz -= xprd;
      flipxz = -1;
    }
    if (domain->xy < -xtiltmax) {
      domain->xy += xprd;
      flipxy = 1;
    } else if (domain->xy >= xtiltmax) {
      domain->xy -= xprd;
      flipxy = -1;
    }
  }

  int flip = 0;
  if (flipxy || flipxz || flipyz) flip = 1;

  if (flip) {
    domain->set_global_box();
    domain->set_local_box();

    domainKK->image_flip(flipxy,flipxz,flipyz);

    domainKK->remap_all();

    domainKK->x2lamda(atom->nlocal);
    atomKK->sync(Host,ALL_MASK);
    irregular->migrate_atoms();
    atomKK->modified(Host,ALL_MASK);
    domainKK->lamda2x(atom->nlocal);
  }
}

namespace LAMMPS_NS {
template class FixNHKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixNHKokkos<LMPHostType>;
#endif
}
