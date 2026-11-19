"""Periodic 2D Gaussian initial conditions and Zel'dovich particle state."""

import numpy as np
from numpy.typing import NDArray

from src.cosmology import (
    Cosmology,
    CosmologyLike,
    growth_factor,
    growth_rate,
    matter_power_spectrum,
    expansion_rate,
)
from src.solver import interpolate_grid_to_particles


INITIAL_REDSHIFT = 50.0
INITIAL_SCALE_FACTOR = 1.0 / (1.0 + INITIAL_REDSHIFT)
INITIAL_RMS_DISPLACEMENT_CELLS = 0.15


def _validate_grid_and_box(grid_size: int, box_size: float) -> None:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if box_size <= 0.0:
        raise ValueError("box_size must be positive")


def _displacement_field(
    density_grid: NDArray[np.float64],
    box_size: float,
) -> NDArray[np.float64]:
    density = np.asarray(density_grid, dtype=np.float64)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density_grid must be a square 2D array")
    if not np.all(np.isfinite(density)):
        raise ValueError("density_grid must be finite")
    grid_size = density.shape[0]
    _validate_grid_and_box(grid_size, box_size)

    kx = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=box_size / grid_size)
    ky = 2.0 * np.pi * np.fft.rfftfreq(grid_size, d=box_size / grid_size)
    k_squared = kx[:, None] ** 2 + ky[None, :] ** 2
    density_k = np.fft.rfft2(density - np.mean(density))
    displacement_k = np.zeros(density_k.shape + (2,), dtype=np.complex128)
    nonzero = k_squared > 0.0
    displacement_k[..., 0][nonzero] = (
        1j * kx[:, None] * density_k / np.where(nonzero, k_squared, 1.0)
    )[nonzero]
    displacement_k[..., 1][nonzero] = (
        1j * ky[None, :] * density_k / np.where(nonzero, k_squared, 1.0)
    )[nonzero]
    return np.stack(
        (
            np.fft.irfft2(displacement_k[..., 0], s=density.shape),
            np.fft.irfft2(displacement_k[..., 1], s=density.shape),
        ),
        axis=-1,
    )


def generate_density_grid(
    grid_size: int,
    seed: int | None = None,
    *,
    box_size: float = 100.0,
    redshift: float = INITIAL_REDSHIFT,
    parameters: CosmologyLike = Cosmology(),
    dark_matter: bool = True,
    target_rms_displacement_cells: float = INITIAL_RMS_DISPLACEMENT_CELLS,
) -> NDArray[np.float64]:
    """Create a seeded 2D Gaussian contrast field with a physical P(k) shape."""
    _validate_grid_and_box(grid_size, box_size)
    if redshift < 0.0:
        raise ValueError("redshift must be non-negative")
    if target_rms_displacement_cells <= 0.0:
        raise ValueError("target_rms_displacement_cells must be positive")

    rng = np.random.default_rng(seed)
    white_noise = rng.normal(size=(grid_size, grid_size))
    noise_k = np.fft.rfft2(white_noise)
    kx = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=box_size / grid_size)
    ky = 2.0 * np.pi * np.fft.rfftfreq(grid_size, d=box_size / grid_size)
    wavenumber = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    power = np.asarray(
        matter_power_spectrum(wavenumber, parameters, dark_matter=dark_matter)
    )
    a_init = 1.0 / (1.0 + redshift)
    delta_k = noise_k * np.sqrt(power) * growth_factor(a_init, parameters)
    delta_k[0, 0] = 0.0
    density = np.fft.irfft2(delta_k, s=(grid_size, grid_size))

    displacement = _displacement_field(density, box_size)
    rms = float(np.sqrt(np.mean(np.sum(displacement**2, axis=-1))))
    if not np.isfinite(rms) or rms == 0.0:
        raise ValueError("The sampled power spectrum produced a zero displacement field")
    target_rms = target_rms_displacement_cells * box_size / grid_size
    return density * (target_rms / rms)


def generate_particle_positions(
    grid_size: int,
    box_size: float,
) -> NDArray[np.float64]:
    """Return regular particle positions at cell centers in a periodic square."""
    _validate_grid_and_box(grid_size, box_size)
    coordinate = (np.arange(grid_size, dtype=np.float64) + 0.5) * (
        box_size / grid_size
    )
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    return np.stack((x.ravel(), y.ravel()), axis=-1)


def apply_zel_dovich_displacements(
    particle_positions: NDArray[np.float64],
    density_grid: NDArray[np.float64],
    scale_factor: float,
    box_size: float,
    parameters: CosmologyLike = Cosmology(),
) -> NDArray[np.float64]:
    """Displace particles from their Lagrangian positions periodically."""
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be positive")
    density = np.asarray(density_grid, dtype=np.float64)
    displacement = _displacement_field(density, box_size)
    initial_scale = INITIAL_SCALE_FACTOR
    displacement *= growth_factor(scale_factor, parameters) / growth_factor(
        initial_scale, parameters
    )
    positions = np.asarray(particle_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("particle_positions must have shape (particle_count, 2)")
    shift = interpolate_grid_to_particles(displacement, positions, box_size)
    return np.mod(positions + shift, box_size)


def generate_initial_conditions(
    grid_size: int,
    box_size: float,
    seed: int | None = None,
    parameters: CosmologyLike = Cosmology(),
    redshift: float = INITIAL_REDSHIFT,
    dark_matter: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return initial positions, canonical momenta, and density contrast."""
    _validate_grid_and_box(grid_size, box_size)
    if redshift < 0.0:
        raise ValueError("redshift must be non-negative")
    scale_factor = 1.0 / (1.0 + redshift)
    density = generate_density_grid(
        grid_size,
        seed,
        box_size=box_size,
        redshift=redshift,
        parameters=parameters,
        dark_matter=dark_matter,
    )
    lagrangian_positions = generate_particle_positions(grid_size, box_size)
    displacement = _displacement_field(density, box_size)
    positions = np.mod(
        lagrangian_positions
        + interpolate_grid_to_particles(displacement, lagrangian_positions, box_size),
        box_size,
    )

    displacement_at_particles = interpolate_grid_to_particles(
        displacement, lagrangian_positions, box_size
    )
    canonical_momenta = (
        scale_factor**3
        * expansion_rate(scale_factor, parameters)
        * growth_rate(scale_factor, parameters)
        * displacement_at_particles
    )
    return positions, canonical_momenta, density
