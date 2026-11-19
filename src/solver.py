"""Periodic 2D Particle-Mesh assignment and FFT gravity."""

import numpy as np
from numpy.typing import NDArray


def _validate_box(box_size: float, grid_size: int) -> None:
    if box_size <= 0.0:
        raise ValueError("box_size must be positive")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")


def _cic_stencil(
    particle_positions: NDArray[np.float64],
    box_size: float,
    grid_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    positions = np.asarray(particle_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("particle_positions must have shape (particle_count, 2)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("particle_positions must be finite")

    cell_size = box_size / grid_size
    scaled = np.mod(positions, box_size) / cell_size
    lower_float = np.floor(scaled)
    lower = lower_float.astype(np.int64)
    fraction = scaled - lower_float
    return lower, (lower + 1) % grid_size, fraction


def cloud_in_cell(
    particle_positions: NDArray[np.float64],
    box_size: float,
    grid_size: int,
    particle_masses: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Deposit particle masses on a periodic 2D mesh using CIC weights."""
    _validate_box(box_size, grid_size)
    positions = np.asarray(particle_positions, dtype=np.float64)
    lower, upper, fraction = _cic_stencil(positions, box_size, grid_size)
    particle_count = positions.shape[0]
    if particle_masses is None:
        masses = np.ones(particle_count, dtype=np.float64)
    else:
        masses = np.asarray(particle_masses, dtype=np.float64)
        if masses.shape != (particle_count,):
            raise ValueError("particle_masses must have one value per particle")
        if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("particle_masses must be finite and non-negative")

    grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    wx = (1.0 - fraction[:, 0], fraction[:, 0])
    wy = (1.0 - fraction[:, 1], fraction[:, 1])
    ix = (lower[:, 0], upper[:, 0])
    iy = (lower[:, 1], upper[:, 1])
    for x_side in (0, 1):
        for y_side in (0, 1):
            np.add.at(
                grid,
                (ix[x_side], iy[y_side]),
                masses * wx[x_side] * wy[y_side],
            )
    return grid


def mass_grid_to_overdensity(mass_grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert deposited particle masses to dimensionless overdensity."""
    mass = np.asarray(mass_grid, dtype=np.float64)
    if mass.ndim != 2 or mass.shape[0] != mass.shape[1]:
        raise ValueError("mass_grid must be a square 2D array")
    mean_mass = float(np.mean(mass))
    if mean_mass <= 0.0 or not np.isfinite(mean_mass):
        raise ValueError("mass_grid must have positive finite mean mass")
    return mass / mean_mass - 1.0


def interpolate_grid_to_particles(
    grid_values: NDArray[np.float64],
    particle_positions: NDArray[np.float64],
    box_size: float,
) -> NDArray[np.float64]:
    """Interpolate scalar or vector nodal values using periodic CIC weights."""
    values = np.asarray(grid_values, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[0] != values.shape[1]:
        raise ValueError("grid_values must have shape (N, N) or (N, N, components)")
    grid_size = values.shape[0]
    _validate_box(box_size, grid_size)
    lower, upper, fraction = _cic_stencil(
        particle_positions, box_size, grid_size
    )
    result_shape = (lower.shape[0],) + values.shape[2:]
    interpolated = np.zeros(result_shape, dtype=np.float64)
    wx = (1.0 - fraction[:, 0], fraction[:, 0])
    wy = (1.0 - fraction[:, 1], fraction[:, 1])
    ix = (lower[:, 0], upper[:, 0])
    iy = (lower[:, 1], upper[:, 1])
    for x_side in (0, 1):
        for y_side in (0, 1):
            weight = wx[x_side] * wy[y_side]
            if values.ndim == 2:
                interpolated += weight * values[ix[x_side], iy[y_side]]
            else:
                interpolated += weight[:, None] * values[ix[x_side], iy[y_side]]
    return interpolated


def solve_gravitational_acceleration(
    density_grid: NDArray[np.float64],
    box_size: float,
    gravitational_constant: float = 1.0,
    *,
    scale_factor: float = 1.0,
    omega_m: float = 0.31,
    hubble_constant: float = 1.0,
) -> NDArray[np.float64]:
    """Solve the periodic cosmological Poisson equation and return nodal force.

    The Poisson source is (3/2) * Omega_m * H0**2 * delta / a. The zero
    Fourier mode is removed, as required for a periodic density contrast.
    """
    density = np.asarray(density_grid, dtype=np.float64)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density_grid must be a square 2D array")
    if not np.all(np.isfinite(density)):
        raise ValueError("density_grid must be finite")
    grid_size = density.shape[0]
    _validate_box(box_size, grid_size)
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be positive")
    if gravitational_constant < 0.0 or omega_m < 0.0 or hubble_constant < 0.0:
        raise ValueError("gravity and cosmology parameters must be non-negative")

    delta = density - np.mean(density)
    kx = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=box_size / grid_size)
    ky = 2.0 * np.pi * np.fft.rfftfreq(grid_size, d=box_size / grid_size)
    k_squared = kx[:, None] ** 2 + ky[None, :] ** 2
    source_k = np.fft.rfft2(delta) * (
        1.5 * gravitational_constant * omega_m * hubble_constant**2 / scale_factor
    )
    potential_k = np.zeros_like(source_k)
    nonzero = k_squared > 0.0
    potential_k[nonzero] = -source_k[nonzero] / k_squared[nonzero]
    potential = np.fft.irfft2(potential_k, s=density.shape)

    cell_size = box_size / grid_size
    acceleration_x = -(
        np.roll(potential, -1, axis=0) - np.roll(potential, 1, axis=0)
    ) / (2.0 * cell_size)
    acceleration_y = -(
        np.roll(potential, -1, axis=1) - np.roll(potential, 1, axis=1)
    ) / (2.0 * cell_size)
    return np.stack((acceleration_x, acceleration_y), axis=-1)
