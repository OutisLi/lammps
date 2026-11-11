# DPA4Spin: Spin-Lattice Simulations with DeePMD-kit Models

This repository is a fork of the development version of LAMMPS. It runs the spin models of DeePMD-kit, through the pair styles `dpa4spin`, `dpa4spin/kk`, and `deepspin`, in molecular dynamics and energy minimization in which every atom carries a magnetic moment whose direction and magnitude both change. All commands of the official LAMMPS work as they do there.

This guide leads from the concepts and the installation to a first simulation, and then to thermostats, restraints, relaxation, output, GPU runs, and restarts. It ends with common errors, restrictions, the translation of inputs of earlier versions of this fork, and the differences of its KOKKOS package. The pages of the individual commands are those of the LAMMPS documentation, under `doc/src` in this repository.

## Concepts

In `atom_style spin`, each atom carries a position, a velocity, and a spin vector S, its magnetic moment. LAMMPS stores S as a unit direction and a magnitude in Bohr magnetons (μB). An atom of magnitude zero is non-magnetic.

The DeePMD-kit spin model gives the potential energy U of all positions and spin vectors. From it, the pair style computes the force on each atom, −∂U/∂x, and the magnetic force on each spin, −∂U/∂S, in eV/μB. The magnetic force has all three components, including the one along S, so the dynamics change the magnitude of a moment as well as its direction. LAMMPS stores the magnetic force in its per-atom array `fm` as |S|/ħ · (−∂U/∂S), in rad/ps; −∂U/∂S equals ħ · fm / |S|.

The inertial spin dynamics (TSPIN) styles of the SPIN package move the spins like particles. Each magnetic atom has a spin mass m_s, given as a multiple of its atomic mass, and a spin velocity dS/dt, and its spin vector follows m_s d²S/dt² = −∂U/∂S. One velocity-Verlet step advances the atoms and the spins together.

These dynamics sample the equilibrium distribution of positions and spins. They do not describe the physical precession of magnetic moments, so spin trajectories and spin time correlations carry no physical time scale. Configurational averages do not depend on the spin masses; the spin masses set how fast the spins move, and a smaller spin mass needs a smaller time step.

The spin velocities carry a spin kinetic energy, ½ Σ m_s |dS/dt|². The thermo keywords `ke`, `etotal`, and `econserve` do not include it; `compute temp/tspin` reports it together with the spin temperature.

## Installation

LAMMPS is built from this repository with the KOKKOS, SPIN, and PLUGIN packages. `Kokkos_ARCH_<GPU>` names the architecture of the GPU, such as `Kokkos_ARCH_AMPERE80`. A build for CPUs only omits `PKG_KOKKOS`, the three Kokkos options, and the `nvcc_wrapper` compiler. The executable `lmp` is installed in `bin` of this repository.

```bash
cmake -S cmake -B build -G Ninja \
      -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON \
      -D CMAKE_CXX_STANDARD=20 -D LAMMPS_INSTALL_RPATH=ON \
      -D PKG_SPIN=ON -D PKG_PLUGIN=ON -D PKG_KOKKOS=ON \
      -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_<GPU>=ON \
      -D CMAKE_CXX_COMPILER=$PWD/lib/kokkos/bin/nvcc_wrapper \
      -D CMAKE_INSTALL_PREFIX=$PWD
cmake --build build --parallel && cmake --install build
```

The DeePMD-kit pair styles come from a plugin, which is built with the DeePMD-kit C++ interface following the plugin-mode instructions of the DeePMD-kit installation guide, with these settings:

1. `LAMMPS_SOURCE_ROOT` points to this repository, and `CMAKE_CXX_STANDARD=20` matches the LAMMPS build.
1. `DEEPMD_LAMMPS_KOKKOS=ON`, with `Kokkos_DIR` set to `build/lib/kokkos` of this repository, adds the GPU pair styles `dpa4spin/kk` and `deepmd/kk`.
1. LAMMPS, the DeePMD-kit C++ interface, its plugin, and the MPI launcher use one MPI library.

A plugin is compiled against the source tree of one LAMMPS, so a plugin built for the official LAMMPS does not work with this fork.

LAMMPS loads the plugin from the directory named by `LAMMPS_PLUGIN_PATH`, which is `lib/deepmd_lmp` under the installation prefix of the DeePMD-kit C++ interface, written `${deepmd_root}` below. The first command below lists the TSPIN styles of the build, and the second the pair styles of the plugin, among them `dpa4spin`, `dpa4spin/kk`, and `deepspin`:

```bash
export LAMMPS_PLUGIN_PATH=${deepmd_root}/lib/deepmd_lmp
lmp -h | grep tspin
echo "plugin list" | lmp -log none
```

## Preparing a simulation

### The model

The pair style depends on how the model was frozen:

| Model                                                              | Pair style on a CPU or GPU | Pair style with KOKKOS on a GPU                                                              |
| ------------------------------------------------------------------ | -------------------------- | -------------------------------------------------------------------------------------------- |
| DPA4 or DPA4C spin model frozen with `dp --pt-expt freeze`         | `dpa4spin`                 | `dpa4spin/kk`, for a compressed DPA4C model or a single-precision DPA4 model frozen on a GPU |
| spin model frozen with `dp --pt freeze`, or with atomic parameters | `deepspin`                 | none; `deepspin` runs on the host                                                            |

`pair_coeff * *` maps the atom types, in order, to element names of the model. `dpa4spin` needs `units metal` or another physical unit style, and on a single MPI rank an atom map, `atom_modify map yes`. A model trained with a charge and spin embedding takes its values through the `charge_spin` keyword of `dpa4spin`. The DeePMD-kit documentation of these pair styles lists all their keywords.

### The data file

A data file for `atom_style spin` lists each atom in the `Atoms` section as `atom-ID atom-type x y z spx spy spz sp`. The three values `spx spy spz` give the direction of the moment; LAMMPS normalizes them, so they need not form a unit vector, but they must not all be zero. The value `sp` is the magnitude in μB. A non-magnetic atom has magnitude 0 and any nonzero direction; it gets no spin mass and stays non-magnetic. Two atoms of a bcc iron cell:

```
# bcc Fe

2 atoms
1 atom types

0.0 2.8665 xlo xhi
0.0 2.8665 ylo yhi
0.0 2.8665 zlo zhi

Masses

1 55.845

Atoms

1 1 0.00000 0.00000 0.00000  0.0 0.0 1.0  2.2
2 1 1.43325 1.43325 1.43325  0.0 0.0 1.0  2.2
```

## A first simulation

The input below runs constant-energy dynamics of bcc iron, with the lattice and the spins moving together from velocities drawn at 300 K:

```lammps
units           metal
atom_style      spin
atom_modify     map yes
read_data       conf.lmp

pair_style      dpa4spin model.pt2
pair_coeff      * * Fe

timestep        0.001
fix             1 all nve/tspin spinmass 0.1
velocity        all create 300 4928 mom yes
velocity/tspin  all create 300 4929

compute         ts all temp/tspin
variable        econs equal etotal+c_ts[1]
thermo_style    custom step temp c_ts pe etotal v_econs press
thermo          100
run             10000
```

`fix nve/tspin` integrates the atoms and the spins, and its `spinmass` keyword gives every magnetic atom a spin mass of 0.1 times its atomic mass, which a time step of 1 fs integrates well. The fix creates the per-atom spin velocities and spin masses, so `velocity/tspin` and `compute temp/tspin` come after it. `velocity` draws the atomic velocities and `velocity/tspin` the spin velocities, each at 300 K with a seed of its own. Without a thermostat, part of this kinetic energy turns into potential energy, so the temperatures settle below 300 K.

`compute temp/tspin` gives the spin temperature as `c_ts` and the spin kinetic energy as `c_ts[1]`. Their sum with `etotal`, `v_econs`, is the conserved energy of the dynamics. A steady drift of `v_econs` means that the time step is too large; for the spins, a smaller time step or a larger spin mass reduces it.

The input runs with `lmp -in in.lammps`; the section on GPU runs describes the same input on a GPU.

## Temperature and pressure

`fix nvt/tspin` thermostats the lattice and the spins with two Nose-Hoover chains, which share the target temperature and the damping time. `fix npt/tspin` adds a barostat on the box, and `fix nph/tspin` uses the barostat alone, without thermostats. Each line below is an alternative to the integrator of the first simulation:

```lammps
fix             1 all nvt/tspin temp 300 300 0.1 spinmass 0.1
fix             1 all npt/tspin temp 300 300 0.1 iso 0 0 1.0 spinmass 0.1
fix             1 all nvt/tspin temp 300 300 0.1 spinmass 0.1 lattice frozen
```

The keyword `lattice frozen` holds the atoms in place and samples the spins only, and `spin frozen` holds the spins and samples the lattice only; a barostat needs a moving lattice. With constant targets, no `drag`, and all energy terms included in `pe`, `econserve+c_ts[1]` is the conserved energy of these integrators.

Langevin baths act separately on the two kinds of motion. `fix langevin` acts on the atoms and `fix langevin/tspin` on the spins, both together with `fix nve/tspin`. A bath exchanges energy with the system, so no energy is conserved, and `fix langevin/tspin` must not act on spins that a Nose-Hoover integrator thermostats.

```lammps
fix             1 all nve/tspin spinmass 0.1
fix             2 all langevin 300 300 0.1 14303 zero yes
fix             3 all langevin/tspin 300 300 0.1 56848 zero yes
```

## Restraints and scaled interactions

`fix spring/tspin K S0` adds the energy ½ K (|S| − S0)² to every spin of its group, with K in eV/μB² and S0 in μB; `fix_modify energy yes` includes this energy in `pe`. A magnetic interaction that gives no restoring force along S needs such a restraint to keep the magnitudes bounded.

A DeePMD-kit spin style combines with styles that write no magnetic force, such as `zbl` under `pair_style hybrid/overlay`. `fix adapt` multiplies the energy, forces, and magnetic forces of `dpa4spin` by its pair parameter `scale`, which requires `pair_coeff * *` for that style. A `fix adapt` with N > 0 initializes the pair styles again every N steps, which can double the time per step of a small system, so a scale that stays constant during the run is set once with N = 0:

```lammps
pair_style      hybrid/overlay dpa4spin model.pt2 zbl 1.0 2.0
pair_coeff      * * dpa4spin Fe
pair_coeff      1 1 zbl 26.0 26.0
fix             s1 all spring/tspin 0.8 2.4
fix_modify      s1 energy yes
variable        lambda equal 0.05
fix             scale all adapt 0 pair dpa4spin scale * * v_lambda
```

## Relaxation

`min_style spin/cg/r3` relaxes the atom positions and the full spin vectors together with conjugate gradients. The minimizers `cg`, `sd`, `quickmin`, and `fire` relax the atom positions with the spins fixed, and `min_style spin`, `spin/cg`, and `spin/lbfgs` rotate the spins at fixed magnitudes. `fix spring/tspin` takes part in these minimizations, and `fix box/relax` relaxes the box with `cg` and with `spin/cg/r3`.

A line search moves all variables of a minimization by one step length. With `fix box/relax`, its `vmax` setting can limit that step so far that the spins barely move while the box relaxes. Relaxing the box with `cg` and the positions and spins with `spin/cg/r3` in turn, until the energy no longer changes, converges faster:

```lammps
variable        epre equal 1.0e30
variable        iter loop 30
label           relax
fix             boxr all box/relax iso 0.0
min_style       cg
minimize        1e-12 1e-12 2000 2000
unfix           boxr
min_style       spin/cg/r3
minimize        1e-12 1e-12 2000 2000
variable        enow equal pe
variable        de equal abs(${enow}-${epre})
if "${de} < 1e-6" then "jump SELF relaxed"
variable        epre equal ${enow}
next            iter
jump            SELF relax
label           relaxed
write_data      relaxed.data
```

The force tolerance of `minimize` applies to the forces on the atoms and on the spins together, the latter in eV/μB. The `dmax` setting of `min_modify` limits the change of every spin component in one line search, in μB, as it limits that of every coordinate.

## Output

`compute property/atom` gives the moments and magnetic forces of single atoms for dumps; `fmx fmy fmz` hold the stored magnetic force |S|/ħ · (−∂U/∂S). `compute msd/spin` gives the mean-squared displacement of the spin vectors from their values when the compute is defined, and `compute msd/spin/mod` that of the spin magnitudes. Both average over all atoms of their group, so a group of the magnetic atoms excludes the non-magnetic ones.

```lammps
compute         ts all temp/tspin
compute         sp all property/atom sp spx spy spz fmx fmy fmz
dump            1 all custom 1000 traj.dump id type x y z c_sp[*]
group           mag type 1
compute         smsd mag msd/spin
compute         mmsd mag msd/spin/mod
thermo_style    custom step temp c_ts c_smsd[4] c_mmsd
```

`compute spin` gives, as its first four values, the mean direction of the spins of its group and its length; these weight every atom equally, whatever its magnitude. Its fifth value, the magnetic energy, excludes the energy of DeePMD-kit models, and its sixth applies to spins of fixed magnitude; `pe` and `compute temp/tspin` take their place. The per-atom virial of `dpa4spin` comes from `compute centroid/stress/atom`. With several models on the `pair_style` line, `dpa4spin` also writes the model deviation of forces and magnetic forces, as described in the DeePMD-kit documentation.

## GPU runs

With `-sf kk`, LAMMPS runs the KOKKOS variants of the pair style, of the integrators, thermostats, and restraints, and of the minimizers `cg`, `sd`, `quickmin`, `fire`, and `spin/cg/r3` on the GPU, together with `atom_style spin/kk`. The pair style becomes `dpa4spin/kk`, which reads only the models of the table in the section on the model; `deepspin` and the computes of this guide run on the host. One MPI rank uses one GPU:

```bash
lmp -k on g 1 -sf kk -in in.lammps
mpirun -np 4 lmp -k on g 4 -sf kk -in in.lammps
```

Runs on several GPUs exchange atom data every step and run fastest with an MPI library that accesses GPU memory, as the DeePMD-kit installation guide describes.

The KOKKOS variants produce the trajectories of the host styles up to round-off, and converged minimizations reach the minima of the host styles. `fix langevin/tspin/kk` draws its random numbers from a generator of its own and samples the same bath along a different trajectory. The KOKKOS minimizers use the `quadratic` line search, the default of `min_modify line`. While `compute msd/spin` or `compute msd/spin/mod` is defined, atoms are sorted and moved between MPI ranks on the host, which slows a GPU run; these computes are best defined only when needed.

## Restarts

`write_restart` or the `restart` command writes restart files, which carry the spin velocities, the spin masses, and the state of the spin thermostat chain. After `read_restart`, the input defines the pair style and the integrator again:

```lammps
read_restart    state.restart
pair_style      dpa4spin model.pt2
pair_coeff      * * Fe
fix             1 all nvt/tspin temp 300 300 0.1
restart         10000 next.restart
run             10000
```

A restart file continues a run in the mode that wrote it, with or without `-sf kk`. In the other mode the spin velocities and spin masses are not restored, and LAMMPS lists them under the unused restart information of its log; such a run starts from a data file written by `write_data` and draws the spin velocities again with `velocity/tspin`. An integrator with a thermostat chain of a different length than in the restart file starts both chains afresh, as `fix nh` does.

## Common errors

| Message or symptom                                                                                                                                          | Cause and remedy                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `Fix nve/tspin requires a spin mass for every magnetic atom in the group; use the spinmass keyword or the velocity/tspin command`                           | A magnetic atom has no spin mass. The `spinmass` keyword of the integrator or of `velocity/tspin` assigns it.              |
| `Velocity/tspin needs the internal per-atom state of the tspin styles; define one of the tspin integrators first ...`, or the same for `Compute temp/tspin` | `velocity/tspin` or `compute temp/tspin` comes before the TSPIN integrator.                                                |
| `pair style dpa4spin needs an atom map on a single rank; add 'atom_modify map yes' to the input.`                                                           | The input lacks `atom_modify map yes`.                                                                                     |
| `pair style dpa4spin/kk requires models frozen with the compact canonical graph lower; ...`                                                                 | The model does not run on the GPU pair style; the table in the section on the model lists the models that do.              |
| `pair style dpa4spin sets one scale for every atom type, i.e. pair_coeff * *.`                                                                              | `pair_coeff` of `dpa4spin` names types other than `* *`.                                                                   |
| `Unsupported model file format`                                                                                                                             | A word after the model file, such as the removed `magnetic_force cartesian`, is read as a further model file.              |
| `Pair style ... does not provide the derivative of the energy with respect to the full spin vector and cannot be used with the tspin integrators`           | A SPIN pair style for spins of fixed magnitude is combined with the TSPIN styles.                                          |
| `Kokkos minimize only supports the 'min_modify line quadratic' option`                                                                                      | A KOKKOS minimizer runs with another line search.                                                                          |
| `Fix nve/tspin is not compatible with run_style respa`                                                                                                      | The TSPIN integrators run with `run_style verlet` only.                                                                    |
| The spins do not move, without an error                                                                                                                     | `fix nve/spin` is used; it ignores the magnetic force of the DeePMD-kit pair styles. The TSPIN integrators take its place. |

## Restrictions

- The TSPIN styles require `atom_style spin` and reject dynamic groups. The TSPIN integrators also reject `run_style respa` and must not be combined with `fix nve/spin`.
- Pair styles of the SPIN package that provide no full spin gradient are rejected by the TSPIN integrators and by `min_style spin/cg/r3`.
- The compatibility of other magnetic fixes, such as `fix precession/spin`, with the TSPIN styles is not checked.

## Inputs of earlier versions of this fork

Earlier versions of this fork implemented the same dynamics under other style names. Their inputs translate as follows; `spinmass` has the meaning of the former `spmass`.

| Earlier input                                                   | Current input                                                                                           |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `pair_style dpa4spin model.pt2 magnetic_force cartesian`        | `pair_style dpa4spin model.pt2`                                                                         |
| `fix 1 all nve/spin lattice_flag 1 spin_flag 1`                 | `fix 1 all nve/tspin`                                                                                   |
| `fix 1 all npt temp ... iso ... mass ms rand seed`              | `fix 1 all npt/tspin temp ... iso ... spinmass ms`                                                      |
| `fix 3 all langevin/spin T T damp seed`                         | `fix 3 all langevin/tspin T T damp seed`                                                                |
| `fix 1 all spring/spin/mod K S0`                                | `fix 1 all spring/tspin K S0`                                                                           |
| `velocity all create T seed spin yes spmass ms`                 | `velocity all create T seed` and, after the integrator, `velocity/tspin all create T seed2 spinmass ms` |
| `suffix off` and `suffix on` around the Nose-Hoover integrator  | removed; every TSPIN style has a KOKKOS variant                                                         |
| thermo keyword `ske`                                            | `c_ts[1]` of `compute ts all temp/tspin`                                                                |
| thermo keyword `etotal`, which included the spin kinetic energy | `etotal+c_ts[1]`                                                                                        |

Restart files of earlier versions are not supported; such runs start from a data file written by `write_data`.

## Differences of the KOKKOS package

Two differences from the official KOKKOS package apply to every simulation and need no change of the input.

Repeated runs are reproducible. The official KOKKOS package orders the atoms within one sorting bin differently from run to run, so two runs of one input differ in their last digits, and a thermostat amplifies the difference. This fork orders the atoms of a bin by position. On one MPI rank, with a pair style whose forces do not depend on the order of its neighbor list, repeated runs of one input with the same build and hardware then give identical trajectories.

`fix nvt/kk` integrates its Nose-Hoover chain on a CUDA GPU when it runs on one MPI rank and its temperature compute is a `compute temp/kk` of a static group, such as the compute it creates by default. The trajectory agrees with that of the chain on the host to round-off, and small systems run faster.
