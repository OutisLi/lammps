// SPDX-License-Identifier: LGPL-3.0-or-later

#include "info.h"
#include "input.h"
#include "lammps.h"
#include "library.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mpi.h>
#include <string>

using namespace LAMMPS_NS;

namespace {

std::array<double, 6> run_ensemble(const std::string &style, bool accelerated, bool gpu)
{
    LAMMPS::argv args = {"NonSpinNoseHoover", "-log", "none", "-screen", "none", "-nocite"};
    if (accelerated) {
        args.insert(args.end(), {"-k", "on", gpu ? "g" : "t", "1", "-sf", "kk",
                                 "-pk", "kokkos", "neigh", "half", "newton", "on"});
    }
    auto lmp = std::make_unique<LAMMPS>(args, MPI_COMM_WORLD);
    auto command = [&](const std::string &line) { lmp->input->one(line); };

    // The cutoff is below the initial nearest-neighbor distance. This isolates
    // thermostat and barostat behavior from pair-force reduction differences.
    command("units lj");
    command("atom_style atomic");
    command("lattice fcc 0.8");
    command("region box block 0 2 0 2 0 2");
    command("create_box 1 box");
    command("create_atoms 1 box");
    command("mass 1 1.0");
    command("pair_style lj/cut 1.0");
    command("pair_coeff * * 1.0 1.0");
    command("velocity all create 0.85 4928459 mom yes rot no dist gaussian");
    command("neighbor 0.1 bin");
    command("neigh_modify every 1 delay 0 check no");
    command("timestep 0.001");
    std::string integration = "fix integrate all " + style;
    if (style == "nvt" || style == "npt") integration += " temp 0.85 0.85 0.2";
    if (style == "npt" || style == "nph") integration += " iso 0.5 0.5 1.0";
    command(integration);
    command("thermo_style custom step temp pe ke etotal press vol");
    command("run 20 post no");

    return {lammps_get_thermo(lmp.get(), "temp"), lammps_get_thermo(lmp.get(), "pe"),
            lammps_get_thermo(lmp.get(), "ke"), lammps_get_thermo(lmp.get(), "etotal"),
            lammps_get_thermo(lmp.get(), "press"), lammps_get_thermo(lmp.get(), "vol")};
}

class NonSpinNoseHooverTest : public ::testing::TestWithParam<const char *> {};

TEST_P(NonSpinNoseHooverTest, KokkosMatchesHost)
{
    const bool gpu = Info::has_accelerator_feature("KOKKOS", "api", "cuda") ||
        Info::has_accelerator_feature("KOKKOS", "api", "hip") ||
        Info::has_accelerator_feature("KOKKOS", "api", "sycl");
    if (gpu && !Info::has_kokkos_gpu_device()) GTEST_SKIP() << "No compatible GPU available";

    const auto reference = run_ensemble(GetParam(), false, gpu);
    const auto accelerated = run_ensemble(GetParam(), true, gpu);
    for (std::size_t i = 0; i < reference.size(); ++i) {
        ASSERT_TRUE(std::isfinite(reference[i]));
        ASSERT_TRUE(std::isfinite(accelerated[i]));
        EXPECT_NEAR(reference[i], accelerated[i], 1.0e-10 * std::max(1.0, std::abs(reference[i])))
            << "thermodynamic observable " << i;
    }
    EXPECT_NEAR(reference[3], reference[1] + reference[2], 1.0e-12);
    EXPECT_NEAR(accelerated[3], accelerated[1] + accelerated[2], 1.0e-12);
}

INSTANTIATE_TEST_SUITE_P(OrdinaryIntegration, NonSpinNoseHooverTest,
                        ::testing::Values("nve", "nvt", "npt", "nph"));

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    lammps_kokkos_finalize();
    MPI_Finalize();
    return result;
}
