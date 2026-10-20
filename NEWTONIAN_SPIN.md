# Newtonian Variable-Magnitude Spin Dynamics for DeePMD

This fork adds Cartesian Newtonian dynamics for DeePMD spin models whose magnetic output is the
conservative generalized force

```text
force_mag = -dE/dS
```

The implementation evolves the complete spin vector `S`, including its
magnitude. It is not the fixed-magnitude Landau-Lifshitz dynamics implemented
by upstream LAMMPS, and it is not an implementation of the
Landau-Lifshitz-Bloch equation.

This document has two parts:

1. installation and usage;
1. design changes, migration guidance, and correctness validation.

## Part I: Installation and usage

### 1. Scope and requirements

The deployment path is the Kokkos device pair style of the DeePMD-kit LAMMPS
plugin:

- LAMMPS `atom_style spin/kk` and `run_style verlet/kk`;
- `pair_style dpa4spin/kk`, or its `dpa4spin/kk/device` alias, with
  `magnetic_force cartesian`;
- a CUDA native-spin PT2 model using the `dpa4c_canonical` lower schema;
- magnetic output `force_mag = -dE/dS`;
- `units metal`;
- single-rank and MPI domain-decomposed execution.

Non-spin simulations retain the ordinary LAMMPS integration path. The
`deepmd` and `deepmd/kk` pair styles of the plugin serve them as they do with
upstream LAMMPS, and standard `nve`, `nvt`, `npt`, and `nph` fixes do not
enable spin dynamics by default. The Kokkos suffix is supported for these
non-spin simulations; the restriction on suffixed Nose-Hoover fixes applies
only when spin integration is enabled. Classical potentials require their
usual packages.

The names `nve/spin` and `langevin/spin` in this fork implement Newtonian
dynamics, so upstream fixed-magnitude spin inputs are a separate case and
cannot be treated as non-spin regression inputs.

Required build dependencies:

- a C++20 compiler compatible with CUDA and the selected libtorch;
- CMake 3.27 or newer, Git, and Ninja or Make;
- an MPI implementation when `BUILD_MPI=ON`;
- DeePMD-kit with its C/C++ interface and LAMMPS plugin, at a revision that
  writes the Cartesian magnetic force in the `dpa4spin` styles
  ([deepmd-kit#6084](https://github.com/deepmodeling/deepmd-kit/pull/6084));
- the PyTorch Exportable backend, its matching libtorch, and CUDA.

LAMMPS, the DeePMD-kit C/C++ interface with its plugin, the MPI compilers,
and the MPI launcher must use the same MPI library. Check the MPI library paths
printed by CMake as well as the compiler wrappers, and do not mix libraries or
PMIx components from unrelated system or Conda MPI installations. `lmp -h`
loads no DeePMD-kit library, and the plugin loads the DeePMD-kit operator
library only when a PT2 model is opened, so neither `lmp -h` nor the plugin
listing of section 4 verifies the complete library chain.

### 2. Obtain the source and select DeePMD

The `outisli` branch integrates upstream `develop` updates. The
`dpa4spin-4Jul2026-v1` tag preserves the validated LAMMPS 4 Jul 2026
baseline for reproducible builds.

```bash
git clone --branch outisli https://github.com/OutisLi/lammps.git lammps-outisli
cd lammps-outisli

SOURCE_ROOT=$PWD
INSTALL_ROOT=/path/to/lammps-newtonian-spin
DEEPMD_SOURCE_ROOT=/path/to/deepmd-kit
DEEPMD_ROOT=/path/to/deepmd/install
```

`DEEPMD_SOURCE_ROOT` is the DeePMD-kit checkout whose C/C++ interface builds
the plugin in section 3, and `DEEPMD_ROOT` is its install prefix.

### 3. Build LAMMPS and the DeePMD-kit plugin

```bash
cmake -S cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_MPI=ON \
  -DBUILD_OMP=ON \
  -DPKG_SPIN=ON \
  -DPKG_KSPACE=ON \
  -DPKG_EXTRA-FIX=ON \
  -DPKG_PLUGIN=ON \
  -DPKG_KOKKOS=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_HOPPER90=ON \
  -DCMAKE_CXX_COMPILER="${SOURCE_ROOT}/lib/kokkos/bin/nvcc_wrapper" \
  -DMPI_CXX_COMPILER=/path/to/matching/mpicxx \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_ROOT}" \
  -DLAMMPS_INSTALL_RPATH=ON

cmake --build build --parallel
cmake --install build
```

Replace `Kokkos_ARCH_HOPPER90` with the architecture matching the deployment
GPU. The Kokkos architecture, PT2 artifact, CUDA, and libtorch must be
compatible with that device. If MPI is not needed, use `-DBUILD_MPI=OFF` and
omit `MPI_CXX_COMPILER`.

Then configure the DeePMD-kit C/C++ interface against this source tree, with
the backend options of its installation guide, so that it builds the plugin
with the Kokkos styles. CMake looks up the MPI of C and of C++ separately, so
name both compiler wrappers, and configure a new build directory when the MPI
library changes:

```bash
cmake -S "${DEEPMD_SOURCE_ROOT}/source" -B "${DEEPMD_SOURCE_ROOT}/source/build" \
  -DCMAKE_INSTALL_PREFIX="${DEEPMD_ROOT}" \
  -DLAMMPS_SOURCE_ROOT="${SOURCE_ROOT}" \
  -DDEEPMD_LAMMPS_KOKKOS=ON \
  -DKokkos_DIR="${SOURCE_ROOT}/build/lib/kokkos" \
  -DKokkos_COMPILE_LAUNCHER="${SOURCE_ROOT}/lib/kokkos/bin/kokkos_launch_compiler" \
  -DKokkos_NVCC_WRAPPER="${SOURCE_ROOT}/lib/kokkos/bin/nvcc_wrapper" \
  -DMPI_C_COMPILER=/path/to/matching/mpicc \
  -DMPI_CXX_COMPILER=/path/to/matching/mpicxx

cmake --build "${DEEPMD_SOURCE_ROOT}/source/build" --parallel
cmake --install "${DEEPMD_SOURCE_ROOT}/source/build"
```

The plugin is installed as `${DEEPMD_ROOT}/lib/deepmd_lmp/dpplugin.so`. It is
compiled against the headers and the Kokkos configuration of this LAMMPS
build, so rebuild and reinstall it whenever LAMMPS is rebuilt. A change of the
DeePMD-kit sources needs the DeePMD-kit build and install above repeated, not
a LAMMPS rebuild.

### 4. Verify and launch

LAMMPS loads every file ending in `plugin.so` in the directories of
`LAMMPS_PLUGIN_PATH` when it creates an instance. It opens the plugin through
its link in `lib/deepmd_lmp`, from where the RUNPATH of the plugin,
`$ORIGIN:$ORIGIN/..`, reaches the DeePMD-kit libraries in `lib`, so the plugin
path alone loads it:

```bash
export LAMMPS_PLUGIN_PATH="${DEEPMD_ROOT}/lib/deepmd_lmp"
"${INSTALL_ROOT}/bin/lmp" -h
echo "plugin list" | "${INSTALL_ROOT}/bin/lmp" -log none
ldd "${INSTALL_ROOT}/bin/lmp"
ldd "${DEEPMD_ROOT}/lib/deepmd_lmp/dpplugin.so"
ldd "${DEEPMD_ROOT}/lib/libdeepmd_op_pt.so"
```

`-h` exits before any plugin is loaded and lists the compiled-in styles, which
must include `spin/kk`, `nve/spin`, `nve/spin/kk`, `langevin/spin`,
`spring/spin/mod`, `msd/spin`, and `msd/spin/mod`; `plugin list` must show the
pair styles `dpa4spin/kk` and `dpa4spin/kk/device`.
Every library reported by `ldd` must resolve to the intended DeePMD,
libtorch, CUDA, and MPI installation. These commands target the Linux
deployment host; a Mac checkout used to manage this repository does not
need a LAMMPS installation.

Launch a single GPU with:

```bash
"${INSTALL_ROOT}/bin/lmp" -k on g 1 -in in.lammps
```

Select the accelerated atom, pair, and run styles explicitly in the input,
together with the Cartesian magnetic force of the pair style (section 8).
The simulation box must exist before selecting the run style:

```lammps
atom_style spin/kk
read_data conf.lmp
run_style verlet/kk
pair_style dpa4spin/kk model.pt2 magnetic_force cartesian
fix integrate all nve/spin/kk lattice_flag yes spin_flag yes
```

`fix nve/spin/kk` integrates the lattice and the spins on the device, so on
one rank with periodic boundaries a Newtonian NVE step moves no per-atom data
between host and device; a global `-sf kk` selects it for `fix nve/spin`.
With more than one rank or a fixed boundary, the legacy exchange of
`spin/kk` (section 19.2) moves the per-atom arrays through the host at each
neighbor-list rebuild. Keep the Newtonian Nose-Hoover styles
unsuffixed: `-sf kk` would select `nvt/kk` or `npt/kk`, which do not
implement this fork's spin state. The `spin/kk` atom style keeps the
Newtonian spin velocity and inertia in dual views like the other per-atom
arrays. The host styles `fix nve/spin`, `fix langevin/spin`, and the
Nose-Hoover extension read and write them on the host. While such a
time-integration fix runs on the host, the KOKKOS package packs the forward
communication of ghost atoms on the host, unless `package kokkos comm` or
`comm/forward` is set explicitly.

### 5. Export a compatible DeePMD native-spin model

Use a native-spin DPA4C model eligible for compact canonical export, in a
freeze environment compatible with the deployment host. For a compatible
PyTorch Exportable checkpoint, the DeePMD compression workflow is:

```bash
export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0

dp --pt-expt freeze \
  -c /path/to/model.ckpt.pt \
  -o model \
  --compress
```

Check the resulting `model.pt2` metadata before running LAMMPS:

```bash
python - <<'PY'
import json
import zipfile

with zipfile.ZipFile("model.pt2") as archive:
    metadata = json.loads(archive.read("model/extra/metadata.json"))

assert metadata["is_spin"] is True
assert metadata["spin_scheme"] == "native"
assert metadata["lower_input_kind"] == "dpa4c_canonical"
assert metadata["canonical_index_dtype"] == "uint32"
print(json.dumps(metadata, indent=2))
PY
```

The Kokkos device path cannot serve an ordinary `graph`, `edge_vec`, or
`nlist` artifact, including a DPA4/SeZM artifact that requires a host pair
style. A filename ending in `.pt2` does not establish compatibility.

The artifact is tied to its GPU architecture, CUDA, libtorch ABI, and
AOTInductor version; re-export when these deployment properties change.
The compact canonical path does not require a nested with-communication
artifact for MPI execution.

### 6. Spin data representation

`atom_style spin` represents each spin with:

- `sp[i][0:3]`: unit direction;
- `sp[i][3]`: magnitude;
- physical vector: `S_i = sp[i][0:3] * sp[i][3]`.

An `Atoms # spin` record has the form:

```text
atom-ID atom-type x y z direction-x direction-y direction-z magnitude
```

Example:

```text
1 1 0.0 0.0 0.0 0.0 0.0 1.0 2.2
```

The package adds:

- a Cartesian spin velocity `dS/dt`;
- a positive generalized spin inertia.

There is no duplicate Cartesian spin coordinate. The physical spin is always
reconstructed from the upstream `sp` representation, which remains the
single source of truth.

Atoms with zero initial spin magnitude receive zero spin inertia and remain
inactive in Newtonian spin integration.

### 7. Unit convention

Newtonian spin dynamics is restricted to `units metal`.

If spin is measured in Bohr magnetons, the `spmass` scale has units of
angstrom-squared per Bohr-magneton-squared. The effective generalized inertia
of atom `i` is

```text
I_i = atomic_mass_i * spmass
```

with units

```text
amu * angstrom^2 / Bohr-magneton^2
```

This convention makes the existing LAMMPS `ftm2v` and `mvv2e` conversion
factors applicable to the generalized Cartesian spin coordinate.

### 8. DeePMD pair style and element mapping

Use the Kokkos style of the plugin with a native-spin compact canonical PT2
artifact and `atom_style spin/kk`, and select the Cartesian magnetic force:

```lammps
pair_style dpa4spin/kk model.pt2 magnetic_force cartesian
pair_coeff * * Fe
```

List element names explicitly in `pair_coeff`. A bare `pair_coeff * *` maps
LAMMPS types to the first model types by position and is unsafe for models
whose type map contains more species than the simulated system.

Under `magnetic_force cartesian`, which requires `units metal`, the pair
style writes the raw `-dE/dS` and declares that convention (section 20.2).
Under the default `magnetic_force precession` it writes the precession field
that the upstream SPIN package expects, which the Newtonian integrators reject,
as they reject the `deepspin` style. The host style `dpa4spin` declares the
Cartesian convention under `magnetic_force cartesian` as well, so the
integrators accept it too; the validation of Part II covers the device style.
The host alias `dpa4spin/kk/host` stops with an error, because the Kokkos
style runs on the GPU only.

Use a direct pair style. Hybrid combinations are rejected because their
sub-styles may accumulate different magnetic-force conventions into `fm`.

### 9. Initialize lattice and spin velocities

Initialize both velocity sets with:

```lammps
velocity all create 300.0 43821 spin yes spmass 0.1
velocity all zero linear
```

The first command:

- initializes ordinary atomic velocities;
- initializes Cartesian spin velocities;
- assigns spin inertia;
- rescales active spin modes to the requested temperature.

The optional second command removes ordinary center-of-mass velocity. It does
not erase spin velocity.

Spin velocity initialization uses a counter-based Gaussian generator keyed by
seed, atom tag, component, and a domain-separated stream. The result is
independent of MPI decomposition and atom migration.

### 10. Newtonian NVE

```lammps
fix integrate all nve/spin lattice_flag yes spin_flag yes
```

The flags independently select lattice and Cartesian spin integration:

- `lattice_flag yes`: integrate positions and atomic velocities;
- `spin_flag yes`: integrate spin vectors and spin velocities.

The spin update uses velocity-Verlet:

```text
V_S(t + dt/2) = V_S(t) + dt * F_S(t) / (2 I)
S(t + dt)     = S(t) + dt * V_S(t + dt/2)
V_S(t + dt)   = V_S(t + dt/2) + dt * F_S(t + dt) / (2 I)
```

After the drift, `S` is decomposed back into unit direction and magnitude.
The magnitude is not constrained.

Under the KOKKOS package `fix nve/spin/kk` performs the same update on the
device. Each operation is rounded on its own and evaluated in the order of
the host fix, so in double precision on CUDA devices its trajectories agree
with `fix nve/spin` bit for bit wherever the host compiler does not contract
multiply-adds. A spin vector whose magnitude becomes non-finite stops the run
at the next thermo output or at the end of the run, rather than in the step
in which it occurs.

### 11. Newtonian NVT with Langevin thermostats

A typical coupled lattice-spin NVT input uses separate Langevin forces:

```lammps
fix integrate all nve/spin lattice_flag yes spin_flag yes
fix lattice_bath all langevin 1300.0 1300.0 0.1 1007 zero no
fix spin_bath all langevin/spin 1300.0 1300.0 0.1 22086 zero yes
```

`fix langevin/spin` syntax:

```text
fix ID group langevin/spin Tstart Tstop damping seed [zero yes|no]
```

It applies Cartesian damping and Gaussian noise satisfying the
fluctuation-dissipation relation for the configured spin inertia.

Langevin noise is keyed by seed, timestep, atom tag, component, and the
`LANGEVIN_FORCE` stream. It is therefore invariant under restart and MPI
domain decomposition. The stream differs from velocity initialization even
when both commands use the same numeric seed.

### 12. Newtonian NVT and NPT with Nose-Hoover

The unsuffixed `nvt` and `npt` styles accept:

```text
mass SPIN_MASS_SCALE
rand SPIN_SEED
lattice yes|no
spin yes|no
```

Example NPT:

```lammps
fix ensemble all npt \
  temp 1300.0 1300.0 0.1 \
  iso 0.0 0.0 1.0 \
  mass 0.1 rand 2097 lattice yes spin yes
```

When spin integration is enabled:

- lattice and spin kinetic modes share one Nose-Hoover chain;
- the chain temperature includes both sets of kinetic modes;
- only lattice kinetic energy contributes to pressure and MTK barostat terms;
- spin kinetic energy is included in the extended thermal state but not in
  pressure.

`rand` supplies a fallback spin-velocity seed when spin velocities were not
initialized before the run. Explicit velocity initialization is recommended.

Newtonian spin NPT requires `lattice yes`.

### 13. Observables and restraints

Spin kinetic energy:

```lammps
thermo_style custom step ke ske pe etotal temp press
```

- `ske`: global spin kinetic energy;
- `etotal`: potential energy plus lattice and spin kinetic energy.

Cartesian spin-vector MSD:

```lammps
compute spin_msd all msd/spin
```

Spin-magnitude MSD:

```lammps
compute spin_modulus_msd all msd/spin/mod
```

Harmonic restraint on spin magnitude:

```lammps
fix spin_spring all spring/spin/mod K S0
fix_modify spin_spring energy yes
```

The restraint adds

```text
U = 0.5 * K * (|S| - S0)^2
```

and the corresponding radial generalized force.

### 14. Complete NVT example

```lammps
units metal
boundary p p p
atom_style spin/kk
atom_modify map yes

read_data conf.lmp
run_style verlet/kk
mass 1 55.845

neighbor 1.0 bin
timestep 0.0005

pair_style dpa4spin/kk model.pt2 magnetic_force cartesian
pair_coeff * * Fe

velocity all create 1300.0 14884 spin yes spmass 0.1
velocity all zero linear

fix integrate all nve/spin lattice_flag yes spin_flag yes
fix lattice_bath all langevin 1300.0 1300.0 0.1 1007 zero no
fix spin_bath all langevin/spin 1300.0 1300.0 0.1 22086 zero yes

compute spin_state all property/atom sp spx spy spz fmx fmy fmz
compute spin_msd all msd/spin

thermo 10
thermo_style custom step ke ske pe etotal temp press c_spin_msd[*]

dump trajectory all custom 100 trajectory.dump \
  id type x y z vx vy vz \
  c_spin_state[1] c_spin_state[2] c_spin_state[3] c_spin_state[4] \
  c_spin_state[5] c_spin_state[6] c_spin_state[7]

run 10000
write_restart state.restart
```

### 15. Restart

Spin direction, magnitude, velocity, and inertia participate in atom sorting,
MPI migration, exchange, and restart I/O.

Continue from a restart without reinitializing spin velocity:

```lammps
units metal
read_restart state.restart
run_style verlet/kk

pair_style dpa4spin/kk model.pt2 magnetic_force cartesian
pair_coeff * * Fe

fix integrate all nve/spin lattice_flag yes spin_flag yes
run 10000
```

Reissuing `velocity ... spin yes` after `read_restart` intentionally replaces
the stored spin velocities and should not be done for a continuous
trajectory.

### 16. Generic MPI and multi-GPU launch

Use the MPI launcher from the same MPI installation used to compile LAMMPS and
the DeePMD-kit C/C++ interface. On several GPUs this should be an MPI library
that can access GPU memory (CUDA-aware MPI), such as Open MPI with UCX: with an
MPI library without CUDA support, or with that support disabled, the KOKKOS
package copies the positions and forces of every step through host memory.
The Open MPI package of conda-forge, installed together with `ucx`, ships with
its CUDA support disabled. A launch on several GPUs enables it on its command
line, together with `UCX_MEMTYPE_CACHE=n`, without which UCX can take a GPU
buffer for host memory and abort. A run on one GPU leaves it off: the support
adds a memory-type query to every MPI call, which slows the step of a small
system. With that Open MPI, a launch on four GPUs reads:

```bash
MPIEXEC=/path/to/matching/mpiexec
LMP=/path/to/lammps-newtonian-spin/bin/lmp

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

"${MPIEXEC}" --mca opal_cuda_support 1 -x UCX_MEMTYPE_CACHE=n -n 4 "${LMP}" -k on g 4 -in in.lammps
```

The DeePMD loader assigns visible GPUs by node-local MPI rank. Expose at least
as many devices as ranks per node when one rank per GPU is desired.

With more than one rank, `atom_style spin/kk` moves migrating atoms with the
legacy exchange and prints a warning when another exchange mode was selected,
because the KOKKOS exchange does not carry the Newtonian spin velocity and
inertia. A fixed (`f`) boundary, through which atoms can be lost, selects the
legacy exchange on one rank as well.

The PT2 file and all input files must be visible to every rank. The compact
canonical graph path exchanges ghost contributions through the Kokkos pair
style; it does not require a nested with-communication artifact.

Correctness does not imply multi-GPU speedup. For small systems, duplicated
model execution and MPI communication can make multiple GPUs slower than one
GPU. Benchmark with the production atom count and decomposition.

### 17. Supported and rejected combinations

Supported:

- `units metal`;
- `pair_style dpa4spin/kk` or `dpa4spin/kk/device` of the DeePMD-kit plugin
  with `magnetic_force cartesian`;
- `atom_style spin/kk` and the Kokkos Verlet run style;
- Newtonian `nve/spin` on the host or as `nve/spin/kk` on the device, and
  host Newtonian `nvt` and `npt` fixes;
- single-rank and MPI domain decomposition;
- CUDA native-spin PT2 models with the `dpa4c_canonical` lower schema.

Rejected:

- any pair style that does not declare the Cartesian magnetic force
  convention, including the `dpa4spin` styles under the default
  `magnetic_force precession`, `deepspin`, and upstream fixed-magnitude
  `spin/*` force producers;
- hybrid pair styles, whose sub-styles may mix force conventions;
- RESPA;
- suffixed Nose-Hoover styles such as `npt/kk`, `npt/omp`, or `npt/gpu`;
- non-metal unit styles;
- Newtonian spin NPT with `lattice no`.

These combinations fail explicitly because their state, force-unit, or
per-level integration contract is incomplete.

## Part II: Design changes, migration, and correctness validation

### 18. Why upstream LAMMPS cannot directly integrate DeePMD `force_mag`

Upstream LAMMPS SPIN dynamics stores:

```text
sp[0:3] = unit direction
sp[3]   = fixed magnitude
fm      = precession field or angular-frequency quantity
```

Its `fix nve/spin` advances spin direction through a cross-product rotation
and preserves magnitude.

DeePMD spin models return:

```text
force_mag = -dE/dS
```

This is the Cartesian force conjugate to the complete spin vector. The
`deepspin` style, and the `dpa4spin` styles under their default
`magnetic_force precession`, convert it to a precession quantity proportional
to

```text
force_mag * |S| / hbar
```

That conversion discards the direct generalized-force interpretation needed
by a second-order Cartesian integrator. It also prevents direct evolution of
spin magnitude.

Under `magnetic_force cartesian` the `dpa4spin` styles keep the raw `-dE/dS`
value, which this fork uses in a Newtonian equation:

```text
I * d2S/dt2 = -dE/dS + thermostat forces
```

The force convention and the integrator must be changed together. Using the
raw force with upstream fixed-magnitude integration, or using the precession
field with Newtonian integration, is physically incorrect.

### 19. Source changes relative to upstream LAMMPS

#### 19.1 DeePMD pair styles

This fork carries no DeePMD-kit code. The DeePMD pair styles come from the
LAMMPS plugin of DeePMD-kit, which is built against this source tree. Under
`magnetic_force cartesian`, its `dpa4spin` styles:

- write the magnetic force as the raw `-dE/dS` under energy-unit conversion;
- omit the moment-magnitude and `hbar` conversion, including at zero moment;
- declare `fm_cartesian_force` through `Pair::extract`.

DeePMD-kit provides this convention since
[deepmd-kit#6084](https://github.com/deepmodeling/deepmd-kit/pull/6084), which
resolves [issue #6077](https://github.com/deepmodeling/deepmd-kit/issues/6077).
The DeePMD-kit C/C++ interface and its plugin are built against this source
tree and its MPI library (section 3).

#### 19.2 Per-atom state

`src/atom.{h,cpp}` and `src/SPIN/atom_vec_spin.{h,cpp}` add:

- `spin_velocity[nmax][3]`;
- `spin_mass[nmax]`.

The field registration covers:

- allocation and growth;
- copying during sorting;
- exchange during atom migration;
- restart serialization;
- initialization for `read_data` and `create_atoms`.

Ghost atoms do not carry spin velocity or inertia because DeePMD inference
only needs ghost `sp`. This avoids unnecessary per-step communication.

Under Kokkos, `src/KOKKOS/atom_vec_spin_kokkos.cpp` holds both arrays in dual
views (`k_spin_velocity` and `k_spin_mass`, with the masks `SPV_MASK` and
`SPM_MASK`), sorts them on the device with the other per-atom arrays, and
copies them to the device in `init()`, since commands between runs such as
`velocity` and `read_restart` write them on the host. The KOKKOS exchange
functors do not pack them and fill the slot of a departing atom without
them, so with more than one rank or a fixed boundary, where atoms migrate or
are lost, the style selects the legacy exchange, which handles both arrays
with the other registered fields.

#### 19.3 Newtonian integrator

`src/SPIN/fix_nve_spin.{h,cpp}` replaces the upstream fixed-magnitude
Suzuki-Trotter spin rotation with Cartesian velocity-Verlet.

The implementation:

- optionally integrates lattice and spin coordinates;
- reconstructs Cartesian `S` from direction and magnitude;
- writes the updated direction and magnitude back to `sp`;
- verifies that inertia was initialized;
- rejects incompatible force producers and integration modes.

`src/KOKKOS/fix_nve_spin_kokkos.{h,cpp}` implements `fix nve/spin/kk`. One
device kernel per half step applies the lattice and spin kicks and, in the
first half step, the drifts and the split of `S` into direction and
magnitude, with the separately rounded operations of
`src/KOKKOS/math_rounded_kokkos.h` in the order of the host fix. A
non-finite magnitude sets a device flag, which the fix reads at thermo
output steps and at the end of a run.

The host fix declares the per-atom arrays it reads and writes through
`datamask_read` and `datamask_modify`, so under Kokkos the package
synchronizes only positions, velocities, spins, spin velocities and
inertia, forces, magnetic forces, types, per-atom masses, and group masks
around it. `KokkosLMP::select_forward_comm()` in `src/KOKKOS/kokkos.cpp`
makes the KOKKOS comm styles pack forward communication on the host while a
time-integration fix runs on the host and no `comm` or `comm/forward` setting
was given; the coordinates are then not copied to the device for packing and
back for the next step. For the same reason `NeighborKokkos::check_distance()`
checks the displacements since the last neighbor-list build on the host while
the host holds the newer coordinates. With `fix nve/spin/kk` both stay on the
device.

#### 19.4 Langevin thermostat

`src/SPIN/fix_langevin_spin.{h,cpp}` implements Cartesian damping and Gaussian
noise. The noise amplitude follows the fluctuation-dissipation relation using
the configured spin inertia, timestep, temperature, and damping period.

The optional `zero yes` subtracts the global mean random force over active
spins.

#### 19.5 Counter-based random generator

`src/SPIN/spin_counter_rng.h` provides one header-only generator for:

- velocity initialization;
- Langevin force.

The key is:

```text
(seed, timestep, atom tag, component, stream)
```

The stable stream identifiers provide domain separation. The generator uses a
SplitMix64 counter mixer and Box-Muller Gaussian transform. It has no
rank-local mutable state.

Consequences:

- identical atom-tagged random values under 1, 2, or more MPI ranks;
- no random-sequence reset at restart;
- no change when atoms migrate between ranks;
- no correlation between velocity and Langevin streams when seeds match.

#### 19.6 Velocity command

`src/velocity.{h,cpp}` adds:

```text
spin yes
spmass VALUE
```

Spin velocities are sampled with variance proportional to inverse inertia and
then globally rescaled to the requested temperature. Group boundaries are
respected; initialization of one group does not alter another group.

#### 19.7 Nose-Hoover integration

`src/fix_nh.{h,cpp}` adds optional Newtonian spin integration to unsuffixed
`nvt` and `npt`.

The implementation:

- includes spin kinetic modes in the thermostat temperature;
- scales lattice and spin velocities with one Nose-Hoover chain;
- excludes spin kinetic energy from pressure and MTK barostat terms;
- retains the upstream temperature, degrees of freedom, and MTK pressure
  correction when spin integration is disabled, including accelerator styles;
- calculates and stores separate lattice and spin thermal state only when
  spin integration is enabled;
- rejects unsupported suffix, RESPA, and pressure combinations.

#### 19.8 Thermodynamic output

`src/thermo.{h,cpp}` adds:

- `ske`;
- spin kinetic energy in `etotal` when Newtonian spin state exists.

Non-spin atom styles preserve the original upstream `etotal = pe + ke`
behavior.

#### 19.9 Additional analysis and restraint styles

Added under `src/SPIN`:

- `compute msd/spin`;
- `compute msd/spin/mod`;
- `fix spring/spin/mod`.

Reference state is stored through LAMMPS fix callbacks so it follows atom
sorting, migration, and restart.

### 20. Design invariants

The following invariants are load-bearing and must be preserved in future
ports.

#### 20.1 One spin-coordinate source of truth

Do not add a second persistent Cartesian spin coordinate. Always reconstruct

```text
S = direction * magnitude
```

from `sp` and write updates back to `sp`.

#### 20.2 Explicit force convention

Every producer writing to `fm` in a Newtonian run must use Cartesian
`-dE/dS` units. Precession fields from upstream `spin/*` styles must not be
mixed into the same array.

The two conventions differ by the moment magnitude over `hbar`, a factor of
roughly three thousand for a typical moment. Neither the dimensions nor the
downstream behaviour reveal the mistake: integrating a precession field as a
force yields a plausible trajectory that is wrong. The convention is
therefore declared by the producer rather than inferred from the pair style
name. A pair style that writes the Cartesian force answers

```text
Pair::extract("fm_cartesian_force", dim)
```

with `dim` set to zero and a pointer to an `int` equal to one, and the
integrators refuse to run otherwise. `src/spin_force_convention.h` holds the
single check that `fix nve/spin` and the Nose-Hoover styles share.

A pair style added later is covered by the same check and needs no change to
the integrators. A `dpa4spin` style left at its default
`magnetic_force precession` therefore fails at setup with a clear message
instead of running the wrong dynamics.

#### 20.3 Complete state lifecycle

Spin velocity and inertia must participate in:

- allocation;
- initialization;
- sorting;
- atom migration;
- restart.

Missing any one of these paths produces decomposition-dependent or
uninitialized dynamics.

#### 20.4 Pressure excludes spin kinetic energy

Spin kinetic modes participate in the thermal ensemble but do not represent
translational momentum flux. Pressure and MTK barostat kinetic terms must use
lattice kinetic state only.

#### 20.5 Random values identify atoms, not ranks

Random keys must contain stable atom tags and must not contain MPI rank.
Langevin keys must also include timestep and a dedicated stream identifier.

#### 20.6 Unsupported paths fail before integration

Do not silently fall back for:

- RESPA without per-level spin force storage;
- accelerator suffixes without spin state support;
- hybrid force conventions;
- unsupported units.

### 21. Porting to another LAMMPS version

Do not apply this source tree as a blind textual patch. Port each
responsibility against the target LAMMPS API.

Recommended sequence:

1. Start from a clean target LAMMPS release.
1. Port `spin_velocity` and `spin_mass` registration to the target `Atom` API.
1. Port `AtomVecSpin` grow, copy, exchange, restart, create, and data
   initialization fields.
1. Port Newtonian `fix nve/spin` against the target integration callbacks.
1. Port spin Langevin and the counter RNG.
1. Port velocity initialization.
1. Port the optional FixNH integration only after reviewing all derived
   suffix classes in the target release.
1. Port thermo and analysis styles.
1. Reconfigure from an empty build directory.
1. Build the DeePMD-kit plugin against the port and confirm that
   `dpa4spin/kk` under `magnetic_force cartesian` declares the Cartesian
   convention.
1. Execute every validation layer in the following section.

LAMMPS APIs that commonly change between releases:

- AtomVec field registration;
- fix callback and restart interfaces;
- neighbor-list request APIs;
- Pair and Comm reverse communication;
- Kokkos state masks and DualView ownership;
- FixNH derived classes;
- CMake style discovery.

Never copy generated style headers, object files, or CMake caches into a
new port.

### 22. Correctness validation

Validation must be independent of a particular host, launcher workaround, or
GPU model. Hardware-specific launch commands are operational configuration,
not correctness evidence.

#### 22.1 Build and style registration

Use the CUDA/Kokkos configuration in section 3 with `-DENABLE_TESTING=ON` and
`-B build-test`, then build that directory:

```bash
cmake --build build-test --parallel
build-test/lmp -h
```

Confirm the styles that section 4 lists for `-h`; the DeePMD pair styles
appear in the plugin listing once a plugin built against this tree is on
`LAMMPS_PLUGIN_PATH`.

#### 22.2 Counter RNG unit test

```bash
cmake --build build-test --target test_spin_counter_rng --parallel
ctest --test-dir build-test --output-on-failure -R '^SpinCounterRNG$'
```

The test covers:

- stable fixed vectors;
- seed, timestep, tag, component, and stream separation;
- deterministic repeated calls;
- Gaussian mean and variance.

#### 22.3 Non-spin regression

Run non-spin `atom_style atomic` systems with NVE, NVT, isotropic NPT,
isotropic NPH, and anisotropic NPT. Compare the same seeded inputs against
the official base tag for host and Kokkos execution, including MPI runs.
Acceptance criteria:

- positions, velocities, temperature, pressure, and volume agree within the
  expected floating-point reduction tolerance;
- `etotal = pe + ke`;
- no spin state is allocated or accessed for non-spin atom styles;
- no spin-related MPI reduction is introduced in the thermostat path.

The model-independent Nose-Hoover regression compares host and Kokkos
observables for a dilute non-spin system. Build and run it with:

```bash
cmake --build build-test --target test_nonspin_nh --parallel
ctest --test-dir build-test --output-on-failure -R '^NonSpinNoseHoover$'
```

The test runs on the configured Kokkos device and skips a GPU configuration
when no compatible GPU is available.

#### 22.4 Spin state initialization

Read an `Atoms # spin` data file before calling `velocity ... spin yes` and
print `ske`.

Acceptance criteria:

- `ske = 0`;
- every spin velocity and inertia value is initialized;
- `fix nve/spin` fails clearly if spin inertia was not assigned.

#### 22.5 Magnetic-force finite difference

For one atom and one Cartesian spin component, including a zero moment and
a small nonzero moment, evaluate:

```text
F_fd = -(E(S + h e_a) - E(S - h e_a)) / (2 h)
```

Compare `F_fd` with the raw `fm` component that `pair_style dpa4spin/kk`
returns under `magnetic_force cartesian`.

Recommended initial settings:

- `h = 1e-6` in model spin units;
- double precision;
- relative tolerance `1e-5`, adjusted only after a convergence study.

Test all three Cartesian components and at least one non-collinear spin.

#### 22.6 Velocity-Verlet convergence

Use an analytical Cartesian harmonic spin potential:

```text
U(S) = 0.5 * K * |S|^2
F(S) = -K * S
```

Run with timestep `dt` and `dt/2`.

Acceptance criteria:

- global error decreases by approximately four under timestep halving;
- extended energy remains bounded;
- spin magnitude changes according to the analytical trajectory;
- no implicit magnitude renormalization occurs.

#### 22.7 Langevin statistics

For many independent harmonic spins:

- run until stationary;
- measure spin kinetic temperature;
- verify convergence to the target within statistical uncertainty;
- change damping and verify that equilibrium temperature is unchanged;
- verify identical tagged random forces after restart and MPI repartition.

#### 22.8 Restart continuity

Compare:

1. a continuous NVE trajectory;
1. the same trajectory split by `write_restart` and `read_restart`.

Compare by atom ID:

- unwrapped coordinate;
- physical spin vector;
- spin velocity;
- spin inertia;
- total and spin kinetic energy.

NVE restart should agree to floating-point roundoff. Counter-based Langevin
noise should also continue without repeating or changing its random stream.

#### 22.9 MPI decomposition parity

Create one common restart state and run deterministic NVE with 1, 2, and 4
MPI ranks.

Compare by atom ID after multiple steps:

- unwrapped positions;
- physical spin vectors;
- energy, spin kinetic energy, and pressure.

Suggested acceptance criteria for double precision:

```text
max coordinate difference < 1e-9 angstrom
max spin-component difference < 1e-8
thermodynamic values agree within their printed precision
```

This test exercises:

- native-spin compact canonical graph inference;
- ghost-spin forward communication;
- magnetic-force reverse communication;
- spin-state atom migration;
- rank-independent initialization.

#### 22.10 Workflow smoke tests

At minimum, maintain short deterministic or fixed-seed inputs for:

- NVE;
- NVT with lattice and spin Langevin;
- NPT with Newtonian spin Nose-Hoover coupling;
- thermodynamic integration with `spring/spin/mod`;
- `msd/spin` and `msd/spin/mod`.

Each input must:

- reach a nonzero final step;
- produce finite energy, pressure, spin magnitude, and MSD;
- write a valid data or restart file.

Short smoke tests prove interface integration, not physical equilibration.

#### 22.11 Negative tests

Verify explicit failure for:

- non-metal units;
- a pair style that does not declare `fm_cartesian_force`;
- RESPA;
- suffixed `nvt` or `npt`;
- direct use without initialized spin inertia;
- hybrid pair styles, including one whose sub-style declares the convention;
- non-canonical, non-native-spin, or CPU-only model artifacts with
  `dpa4spin/kk`;
- a `dpa4spin` style under the default `magnetic_force precession`.

#### 22.12 Performance validation

Benchmark only after correctness passes.

Report separately:

- PT2 load time;
- neighbor construction;
- DeePMD pair time;
- MPI communication;
- total steps per second;
- atom count and neighbors per atom;
- ranks and GPUs per node.

Do not infer multi-GPU speedup from a small system. Strong scaling requires
enough local atoms to amortize model replication and communication.

### 23. Release acceptance checklist

Before distributing a port:

- build from a clean source copy;
- run `test_spin_counter_rng`;
- run non-spin and read-data initialization regressions;
- run magnetic-force finite differences;
- run restart continuity;
- run 1/2/4-rank deterministic parity;
- run NVE, NVT, NPT, and thermodynamic-integration smoke tests;
- inspect `ldd` for ABI consistency;
- confirm native-spin compact canonical PT2 metadata;
- record the LAMMPS and DeePMD-kit revisions;
- remove build, install, cache, bytecode, editor, and simulation-output files;
- archive exactly one top-level source directory;
- publish a SHA256 checksum.

### 24. Known limitations

- Newtonian spin dynamics is not LLG or LLB.
- Only `units metal` is supported.
- Newtonian spin integration is validated through the `dpa4spin/kk` device
  path of the DeePMD-kit plugin under `magnetic_force cartesian`; that path
  requires a model with the compact canonical schema.
- The DeePMD-kit plugin must be built against this source tree, and the
  DeePMD-kit C/C++ interface against its MPI library.
- Upstream `spin/*` pair and precession force conventions cannot be mixed.
- RESPA is unsupported.
- Suffixed Newtonian spin Nose-Hoover styles are unsupported.
- `fix langevin/spin`, `fix spring/spin/mod`, and the Newtonian Nose-Hoover
  extension run on the host, so with any of them a Kokkos run moves per-atom
  data between host and device every step.
- The NVT/NPT extension uses one Nose-Hoover chain for lattice and spin
  kinetic modes; it does not provide independent parallel and transverse
  chains.
- AOTInductor PT2 artifacts are not portable across arbitrary CUDA, GPU, and
  libtorch environments.
- Counter-based trajectories are MPI-decomposition and restart invariant on a
  fixed numerical platform. Different standard-library implementations of
  `log` and `cos` may differ in the last floating-point bits across platforms.
