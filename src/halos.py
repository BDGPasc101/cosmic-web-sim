"""Periodic overdensity-based halo finding for 2D simulation snapshots."""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, label


def find_halos(
    overdensity: NDArray[np.float64],
    box_size_mpc: float,
    omega_m: float = 0.31,
    threshold_delta: float = 2.0,
    smooth_sigma: float = 1.0,
) -> list[dict[str, float]]:
    """Find connected overdense regions and estimate their cluster properties.

    The 2D projected map is assigned the full box depth when converting pixels
    to mass. The scale uses the critical density for h=0.68, matching the
    dashboard cosmology, and reports masses in units of 10^12 solar masses.
    """
    density = np.asarray(overdensity, dtype=np.float64)
    if density.ndim != 2 or min(density.shape) < 2:
        raise ValueError("overdensity must be a 2D grid at least 2 cells wide")
    if not np.all(np.isfinite(density)):
        raise ValueError("overdensity must contain only finite values")
    if not np.isfinite(box_size_mpc) or box_size_mpc <= 0.0:
        raise ValueError("box_size_mpc must be finite and positive")
    if not np.isfinite(omega_m) or omega_m <= 0.0:
        raise ValueError("omega_m must be finite and positive")
    if not np.isfinite(threshold_delta):
        raise ValueError("threshold_delta must be finite")
    if not np.isfinite(smooth_sigma) or smooth_sigma < 0.0:
        raise ValueError("smooth_sigma must be finite and non-negative")

    smoothed = gaussian_filter(density, sigma=smooth_sigma, mode="wrap")
    components, component_count = label(
        smoothed > threshold_delta,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if component_count == 0:
        return []

    parents = np.arange(component_count + 1)

    def find_root(component_id: int) -> int:
        root = component_id
        while parents[root] != root:
            parents[root] = parents[parents[root]]
            root = int(parents[root])
        return root

    def merge(first: int, second: int) -> None:
        if first and second:
            first_root = find_root(first)
            second_root = find_root(second)
            if first_root != second_root:
                parents[second_root] = first_root

    rows, columns = components.shape
    for row in range(rows):
        for offset in (-1, 0, 1):
            merge(
                int(components[row, 0]),
                int(components[(row + offset) % rows, columns - 1]),
            )
    for column in range(columns):
        for offset in (-1, 0, 1):
            merge(
                int(components[0, column]),
                int(components[rows - 1, (column + offset) % columns]),
            )

    root_by_component = np.arange(component_count + 1)
    for component_id in range(1, component_count + 1):
        root_by_component[component_id] = find_root(component_id)

    coordinates = np.argwhere(components > 0)
    roots = root_by_component[components[components > 0]]
    order = np.argsort(roots, kind="stable")
    coordinates = coordinates[order]
    roots = roots[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(roots)) + 1]
    group_ends = np.r_[group_starts[1:], len(roots)]

    pixel_size_mpc = min(box_size_mpc / size for size in density.shape)
    pixel_area_mpc2 = box_size_mpc**2 / density.size
    cell_mass_scale_1e12 = (
        2.77536627e11
        / 0.68
        * box_size_mpc**3
        / density.size
        / 1e12
    )
    halos: list[dict[str, float]] = []

    for start, end in zip(group_starts, group_ends):
        cells = coordinates[start:end]
        cell_delta = density[cells[:, 0], cells[:, 1]]
        weights = np.maximum(1.0 + cell_delta, 0.0)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            continue

        centers = []
        for axis, axis_size in enumerate(density.shape):
            angles = 2.0 * np.pi * cells[:, axis] / axis_size
            sine = float(np.sum(weights * np.sin(angles)))
            cosine = float(np.sum(weights * np.cos(angles)))
            centers.append(
                (np.arctan2(sine, cosine) % (2.0 * np.pi))
                * box_size_mpc
                / (2.0 * np.pi)
            )

        halos.append(
            {
                "mass_1e12_msun": (
                    omega_m * total_weight * cell_mass_scale_1e12
                ),
                "x_mpc": float(centers[0]),
                "y_mpc": float(centers[1]),
                "radius_mpc": max(
                    float(np.sqrt(len(cells) * pixel_area_mpc2 / np.pi)),
                    0.5 * pixel_size_mpc,
                ),
                "peak_delta": float(np.max(cell_delta)),
            }
        )

    return sorted(
        halos,
        key=lambda halo: halo["mass_1e12_msun"],
        reverse=True,
    )


def compute_halo_mass_function(
    halos: list[dict], num_bins: int = 10
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return log-spaced masses and cumulative halo counts N(>M)."""
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    if not halos:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    masses = np.asarray(
        [halo["mass_1e12_msun"] for halo in halos],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("halo masses must be finite and positive")

    bin_edges = np.geomspace(
        float(np.min(masses)) / 2.0,
        float(np.max(masses)) * 2.0,
        num_bins + 1,
    )
    mass_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    cumulative_counts = np.count_nonzero(
        masses[:, np.newaxis] >= mass_centers[np.newaxis, :],
        axis=0,
    )
    return mass_centers, cumulative_counts.astype(np.int64)
