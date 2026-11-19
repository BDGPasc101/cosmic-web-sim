"""Focused tests for periodic CIC assignment and PM evolution."""

import numpy as np

from src.cosmology import Cosmology
from src.initial_conditions import (
    generate_initial_conditions,
    generate_particle_positions,
)
from src.simulation import Simulation
from src.solver import (
    cloud_in_cell,
    interpolate_grid_to_particles,
    mass_grid_to_overdensity,
    solve_gravitational_acceleration,
)


def test_cloud_in_cell_conserves_total_mass() -> None:
    positions = np.array([[0.1, 0.2], [2.4, 3.1], [8.0, -0.5]])
    masses = np.array([1.5, 2.0, 4.25])

    deposited = cloud_in_cell(positions, box_size=8.0, grid_size=16, particle_masses=masses)

    assert np.isclose(deposited.sum(), masses.sum(), rtol=0.0, atol=1e-12)


def test_periodic_wrapping_is_finite_and_equivalent() -> None:
    outside = np.array([[-0.25, 8.5], [16.25, -8.5]])
    wrapped = np.mod(outside, 8.0)

    outside_grid = cloud_in_cell(outside, box_size=8.0, grid_size=16)
    wrapped_grid = cloud_in_cell(wrapped, box_size=8.0, grid_size=16)

    assert np.all(np.isfinite(outside_grid))
    assert np.allclose(outside_grid, wrapped_grid, rtol=0.0, atol=1e-14)


def test_uniform_density_has_zero_acceleration() -> None:
    uniform_density = np.zeros((12, 12))

    acceleration = solve_gravitational_acceleration(uniform_density, box_size=10.0)

    assert np.array_equal(acceleration, np.zeros_like(acceleration))


def test_cic_force_interpolation_conserves_total_momentum() -> None:
    rng = np.random.default_rng(42)
    positions = rng.uniform(0.0, 10.0, size=(100, 2))
    masses = rng.uniform(0.1, 2.0, size=positions.shape[0])
    mass_grid = cloud_in_cell(positions, 10.0, 24, masses)
    overdensity = mass_grid_to_overdensity(mass_grid)
    acceleration_grid = solve_gravitational_acceleration(overdensity, 10.0)
    particle_acceleration = interpolate_grid_to_particles(
        acceleration_grid, positions, 10.0
    )

    net_force = np.sum(masses[:, None] * particle_acceleration, axis=0)

    assert np.allclose(net_force, np.zeros(2), atol=1e-12)


def test_zero_net_self_force_random_particles() -> None:
    rng = np.random.default_rng(2026)
    box_size = 100.0
    particle_positions = rng.uniform(0.0, box_size, size=(500, 2))
    mass_grid = cloud_in_cell(particle_positions, box_size, grid_size=32)
    overdensity = mass_grid_to_overdensity(mass_grid)
    acceleration_grid = solve_gravitational_acceleration(overdensity, box_size)
    accelerations = interpolate_grid_to_particles(
        acceleration_grid, particle_positions, box_size
    )

    assert np.allclose(
        np.sum(accelerations, axis=0),
        np.zeros(2),
        rtol=0.0,
        atol=1e-10,
    )


def test_total_momentum_conservation_during_evolution() -> None:
    box_size = 100.0
    positions, momenta, _ = generate_initial_conditions(
        grid_size=16,
        box_size=box_size,
        seed=17,
    )
    initial_total_momentum = np.sum(momenta, axis=0)
    simulation = Simulation(
        positions,
        momenta,
        box_size=box_size,
        grid_size=16,
    )

    snapshots = simulation.run(start_redshift=50.0, end_redshift=0.0, snapshot_count=8)

    for snapshot in snapshots:
        assert np.allclose(
            np.sum(snapshot.momenta, axis=0),
            initial_total_momentum,
            rtol=0.0,
            atol=1e-10,
        )


def test_density_variance_grows_from_z50_to_z0() -> None:
    box_size = 100.0
    positions, momenta, _ = generate_initial_conditions(
        grid_size=16,
        box_size=box_size,
        seed=7,
        parameters=Cosmology(),
    )
    simulation = Simulation(
        positions,
        momenta,
        box_size=box_size,
        grid_size=16,
    )

    snapshots = simulation.run(snapshot_count=51)
    variances = np.array([np.std(snapshot.density_grid) for snapshot in snapshots])

    assert np.all(np.diff(variances) > 0.0)


def test_initial_rms_displacement_is_about_point_15_grid_cells() -> None:
    grid_size = 16
    box_size = 100.0
    positions, _, _ = generate_initial_conditions(
        grid_size=grid_size,
        box_size=box_size,
        seed=7,
    )
    displacement = positions - generate_particle_positions(grid_size, box_size)
    displacement -= box_size * np.rint(displacement / box_size)
    rms_cells = np.sqrt(np.mean(np.sum(displacement**2, axis=1))) / (
        box_size / grid_size
    )

    assert 0.10 < rms_cells < 0.20
