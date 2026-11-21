import numpy as np

from src.halos import compute_halo_mass_function, find_halos


def test_uniform_zero_overdensity_has_no_halos() -> None:
    assert find_halos(np.zeros((32, 32)), box_size_mpc=32.0) == []


def test_gaussian_peak_has_positive_mass_and_accurate_centroid() -> None:
    grid_size = 32
    peak_x, peak_y = 14, 19
    grid_x, grid_y = np.indices((grid_size, grid_size))
    overdensity = 8.0 * np.exp(
        -(
            (grid_x - peak_x) ** 2
            + (grid_y - peak_y) ** 2
        )
        / (2.0 * 2.0**2)
    )

    halos = find_halos(overdensity, box_size_mpc=32.0)

    assert len(halos) == 1
    assert halos[0]["mass_1e12_msun"] > 0.0
    assert np.isclose(halos[0]["x_mpc"], peak_x, atol=0.1)
    assert np.isclose(halos[0]["y_mpc"], peak_y, atol=0.1)


def test_cluster_mass_scale_tracks_matter_density() -> None:
    overdensity = np.zeros((160, 160))
    overdensity[80, 80] = 10.0

    standard_halos = find_halos(
        overdensity, box_size_mpc=100.0, omega_m=0.31, smooth_sigma=0.0
    )
    low_matter_halos = find_halos(
        overdensity, box_size_mpc=100.0, omega_m=0.05, smooth_sigma=0.0
    )

    mass = standard_halos[0]["mass_1e12_msun"]
    assert 10.0 <= mass <= 100.0
    assert np.isclose(
        low_matter_halos[0]["mass_1e12_msun"] / mass,
        0.05 / 0.31,
    )


def test_halo_split_across_periodic_boundary_is_merged() -> None:
    grid_size = 32
    peak_y = 16
    grid_x, grid_y = np.indices((grid_size, grid_size))
    periodic_x_distance = np.minimum(grid_x, grid_size - grid_x)
    overdensity = 8.0 * np.exp(
        -(
            periodic_x_distance**2
            + (grid_y - peak_y) ** 2
        )
        / (2.0 * 2.0**2)
    )

    halos = find_halos(overdensity, box_size_mpc=32.0)

    assert len(halos) == 1
    assert min(halos[0]["x_mpc"], 32.0 - halos[0]["x_mpc"]) < 0.1
    assert np.isclose(halos[0]["y_mpc"], peak_y, atol=0.1)


def test_cumulative_halo_mass_function_is_non_increasing() -> None:
    halos = [
        {"mass_1e12_msun": mass}
        for mass in (0.5, 1.0, 3.0, 20.0, 120.0)
    ]

    mass_bins, cumulative_counts = compute_halo_mass_function(halos)

    assert len(mass_bins) == 10
    assert np.all(np.diff(mass_bins) > 0.0)
    assert np.all(np.diff(cumulative_counts) <= 0)
