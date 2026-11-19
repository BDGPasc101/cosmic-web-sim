"""Scale-factor KDK integration and snapshot storage for a 2D PM simulation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from src.cosmology import (
    Cosmology,
    CosmologyLike,
    CosmologyParameters,
    expansion_rate,
)
from src.solver import (
    cloud_in_cell,
    interpolate_grid_to_particles,
    mass_grid_to_overdensity,
    solve_gravitational_acceleration,
)


@dataclass(frozen=True)
class SimulationSnapshot:
    """Particle state and density field saved at one point in cosmic time."""

    redshift: float
    scale_factor: float
    positions: NDArray[np.float64]
    momenta: NDArray[np.float64]
    density_grid: NDArray[np.float64]
    cosmic_age_gyr: float

    @property
    def velocities(self) -> NDArray[np.float64]:
        """Compatibility alias; the integrated state is canonical momentum."""
        return self.momenta


class Simulation:
    """Evolve particles in comoving coordinates with canonical momentum."""

    def __init__(
        self,
        initial_positions: NDArray[np.float64],
        initial_velocities: NDArray[np.float64],
        box_size: float,
        grid_size: int,
        parameters: CosmologyLike = Cosmology(),
        initial_scale_factor: float = 1.0 / 51.0,
    ) -> None:
        positions = np.asarray(initial_positions, dtype=np.float64)
        momenta = np.asarray(initial_velocities, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("initial_positions must have shape (particle_count, 2)")
        if momenta.shape != positions.shape:
            raise ValueError("initial_velocities must match initial_positions shape")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(momenta)):
            raise ValueError("initial particle state must be finite")
        if box_size <= 0.0 or grid_size < 2:
            raise ValueError("box_size must be positive and grid_size at least 2")
        if initial_scale_factor <= 0.0 or initial_scale_factor > 1.0:
            raise ValueError("initial_scale_factor must be in (0, 1]")

        self.positions = np.mod(positions, box_size)
        self.momenta = momenta.copy()
        self.box_size = float(box_size)
        self.grid_size = int(grid_size)
        self.parameters = (
            parameters.as_cosmology()
            if isinstance(parameters, CosmologyParameters)
            else parameters
        )
        self.scale_factor = float(initial_scale_factor)
        self._snapshot_masses = np.ones(positions.shape[0], dtype=np.float64)

    @property
    def velocities(self) -> NDArray[np.float64]:
        """Compatibility alias for the canonical momentum state."""
        return self.momenta

    def _overdensity(self) -> NDArray[np.float64]:
        mass_grid = cloud_in_cell(
            self.positions,
            self.box_size,
            self.grid_size,
            self._snapshot_masses,
        )
        return mass_grid_to_overdensity(mass_grid)

    def _acceleration(self, scale_factor: float) -> NDArray[np.float64]:
        density = self._overdensity()
        acceleration_grid = solve_gravitational_acceleration(
            density,
            self.box_size,
            scale_factor=scale_factor,
            omega_m=self.parameters.omega_m,
        )
        return interpolate_grid_to_particles(
            acceleration_grid, self.positions, self.box_size
        )

    def step(self, scale_factor: float) -> None:
        """Advance to ``scale_factor`` with a midpoint-coefficient KDK step."""
        target = float(scale_factor)
        if target <= self.scale_factor or target > 1.0:
            raise ValueError("scale_factor must increase and be no greater than 1")
        start = self.scale_factor
        delta_a = target - start
        midpoint = 0.5 * (start + target)
        hubble = expansion_rate(midpoint, self.parameters)

        # With the requested Poisson source, dp/da = -grad(Phi)/(a H).
        # Absorb 1/a into the force so each kick coefficient is delta_a/H.
        start_force = self._acceleration(start) / midpoint
        self.momenta += 0.5 * delta_a / hubble * start_force

        drift_factor = delta_a / (midpoint**3 * hubble)
        self.positions = np.mod(
            self.positions + drift_factor * self.momenta,
            self.box_size,
        )

        end_force = self._acceleration(target) / midpoint
        self.momenta += 0.5 * delta_a / hubble * end_force
        self.scale_factor = target

    def _cosmic_age_gyr(self, scale_factor: float) -> float:
        integral, _ = quad(
            lambda a: 1.0 / (a * expansion_rate(a, self.parameters)),
            1e-8,
            scale_factor,
            epsabs=1e-9,
        )
        h = self.parameters.h
        return float(integral * 9.778 / h)

    def _make_snapshot(self) -> SimulationSnapshot:
        density = self._overdensity()
        return SimulationSnapshot(
            redshift=1.0 / self.scale_factor - 1.0,
            scale_factor=self.scale_factor,
            positions=self.positions.copy(),
            momenta=self.momenta.copy(),
            density_grid=density,
            cosmic_age_gyr=self._cosmic_age_gyr(self.scale_factor),
        )

    def run(
        self,
        start_redshift: float = 50.0,
        end_redshift: float = 0.0,
        snapshot_count: int = 51,
    ) -> list[SimulationSnapshot]:
        """Evolve and save equally spaced scale-factor snapshots."""
        if start_redshift < 0.0 or end_redshift < 0.0:
            raise ValueError("redshifts must be non-negative")
        if start_redshift <= end_redshift:
            raise ValueError("start_redshift must be greater than end_redshift")
        if snapshot_count < 2:
            raise ValueError("snapshot_count must be at least 2")
        start_scale = 1.0 / (1.0 + start_redshift)
        end_scale = 1.0 / (1.0 + end_redshift)
        if start_scale < self.scale_factor - 1e-12:
            raise ValueError("start_redshift is earlier than the current simulation state")
        if start_scale > self.scale_factor + 1e-12:
            self.step(start_scale)
        else:
            self.scale_factor = start_scale

        scales = np.linspace(start_scale, end_scale, snapshot_count)
        snapshots = [self._make_snapshot()]
        for scale in scales[1:]:
            self.step(float(scale))
            snapshots.append(self._make_snapshot())
        return snapshots
