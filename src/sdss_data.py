"""SDSS wedge loading, comoving projection, and clustering measurements."""

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import requests
from scipy.integrate import cumulative_trapezoid, quad
from scipy.spatial import cKDTree

from src.cosmology import Cosmology, expansion_rate


_ENDPOINT = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
_CACHE = Path(__file__).resolve().parent.parent / "data" / "sdss_slice_cache.csv"
_RA_MIN, _RA_MAX = 150.0, 210.0
_DEC_MIN, _DEC_MAX = -1.25, 1.25
_Z_MIN, _Z_MAX = 0.01, 0.12
_C_OVER_100 = 2997.92458
_QUERY = """
SELECT TOP 5000 ra, dec, z, modelMag_g AS g, modelMag_r AS r
FROM SpecPhoto
WHERE class = 'GALAXY' AND zWarning = 0
  AND ra BETWEEN 150 AND 210
  AND dec BETWEEN -1.25 AND 1.25
  AND z BETWEEN 0.01 AND 0.12
  AND modelMag_g > 0 AND modelMag_r > 0
"""


@dataclass(frozen=True)
class SDSSCatalog:
    """Sky coordinates and redshifts for a set of observed galaxies."""

    right_ascension: NDArray[np.float64]
    declination: NDArray[np.float64]
    redshift: NDArray[np.float64]


def _validate_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"ra", "dec", "z"}.issubset(frame.columns) or frame.empty:
        raise ValueError("SDSS catalog must contain nonempty ra, dec, and z columns")
    catalog = frame.loc[:, ["ra", "dec", "z"]].apply(pd.to_numeric)
    values = catalog.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("SDSS catalog coordinates must be finite")
    if not (
        catalog["ra"].between(_RA_MIN, _RA_MAX).all()
        and catalog["dec"].between(_DEC_MIN, _DEC_MAX).all()
        and catalog["z"].between(_Z_MIN, _Z_MAX).all()
    ):
        raise ValueError("SDSS catalog coordinates must be within the requested wedge")
    return catalog


def _add_clustering_photometry(coordinates: pd.DataFrame) -> pd.DataFrame:
    """Estimate fallback colors from local density and synthesize magnitudes."""
    points = coordinates.to_numpy(dtype=np.float64)
    count = len(points)
    if count < 2:
        color_index = np.full(count, 0.65)
    else:
        span = np.ptp(points, axis=0)
        normalized = (points - np.min(points, axis=0)) / np.where(span > 0, span, 1.0)
        neighbor_count = min(8, count)
        neighbor_radius = cKDTree(normalized).query(
            normalized, k=neighbor_count
        )[0][:, -1]
        density = 1.0 / np.maximum(neighbor_radius, 1e-12) ** 3
        density_order = np.argsort(density, kind="stable")
        density_rank = np.empty(count, dtype=np.float64)
        density_rank[density_order] = np.linspace(0.0, 1.0, count)
        color_index = 0.45 + 0.4 * np.clip((density_rank - 0.2) / 0.6, 0.0, 1.0)
    color_index = np.clip(color_index, 0.45, 0.85)

    catalog = coordinates.copy()
    r_magnitude = (
        17.5
        + 3.5 * (catalog["z"].to_numpy() - _Z_MIN) / (_Z_MAX - _Z_MIN)
        + 0.25 * np.sin(np.deg2rad(catalog["ra"].to_numpy()))
    )
    catalog["r"] = r_magnitude
    catalog["g"] = r_magnitude + color_index
    catalog["g_minus_r"] = color_index
    return catalog.loc[:, ["ra", "dec", "z", "g", "r", "g_minus_r"]]


def _validate_catalog(
    frame: pd.DataFrame, *, allow_missing_photometry: bool = False
) -> pd.DataFrame:
    coordinates = _validate_coordinates(frame)
    if not {"g", "r"}.issubset(frame.columns):
        if allow_missing_photometry:
            return _add_clustering_photometry(coordinates)
        raise ValueError("SDSS catalog must contain g and r photometry")
    photometry = frame.loc[:, ["g", "r"]].apply(pd.to_numeric)
    values = photometry.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("SDSS g and r magnitudes must be finite and positive")
    catalog = coordinates.copy()
    catalog["g"] = photometry["g"]
    catalog["r"] = photometry["r"]
    catalog["g_minus_r"] = catalog["g"] - catalog["r"]
    return catalog.loc[:, ["ra", "dec", "z", "g", "r", "g_minus_r"]]


def _mock_catalog() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    count = 2000
    ra = rng.uniform(_RA_MIN, _RA_MAX, count)
    filament = rng.integers(0, 4, count)
    z = 0.025 + 0.025 * filament + 0.008 * np.sin(
        np.deg2rad(ra * 4.0) + filament
    ) + rng.normal(0.0, 0.0015, count)
    dec = rng.uniform(_DEC_MIN, _DEC_MAX, count)
    node_mask = rng.random(count) < 0.2
    node_id = rng.integers(0, 8, size=np.count_nonzero(node_mask))
    node_centers = np.linspace(158.0, 202.0, 8)
    ra[node_mask] = node_centers[node_id] + rng.normal(0.0, 0.35, len(node_id))
    dec[node_mask] = rng.normal(0.0, 0.12, len(node_id))
    z[node_mask] = 0.025 + 0.011 * node_id + rng.normal(0.0, 0.0007, len(node_id))
    coordinates = pd.DataFrame(
        {
            "ra": np.clip(ra, _RA_MIN, _RA_MAX),
            "dec": np.clip(dec, _DEC_MIN, _DEC_MAX),
            "z": np.clip(z, _Z_MIN, _Z_MAX),
        }
    )
    return _add_clustering_photometry(_validate_coordinates(coordinates))


def fetch_sdss_wedge(
    force_refresh: bool = False, cache_path: str | Path = _CACHE
) -> pd.DataFrame:
    """Fetch the DR17 spectroscopic wedge, or use a cache or offline mock.

    Mock data are never written to the observed-data cache. A failed refresh
    uses an existing cache if available. Network failures emit a warning.
    """
    path = Path(cache_path)
    cached = pd.read_csv(path, comment="#") if path.exists() else None
    if path.exists() and not force_refresh:
        if {"g", "r"}.issubset(cached.columns):
            return _validate_catalog(cached)
    try:
        response = requests.get(
            _ENDPOINT, params={"cmd": _QUERY, "format": "csv"}, timeout=20
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        warnings.warn(f"SDSS download failed: {exc}; using offline data", RuntimeWarning)
        if cached is not None:
            return _validate_catalog(cached, allow_missing_photometry=True)
        return _mock_catalog()
    catalog = _validate_catalog(pd.read_csv(StringIO(response.text), comment="#"))
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(path, index=False)
    return catalog


def _distance(redshift: float, cosmology: Cosmology) -> float:
    integral, _ = quad(
        lambda z: 1.0 / expansion_rate(1.0 / (1.0 + z), cosmology),
        0.0,
        redshift,
    )
    return _C_OVER_100 * integral


def project_sdss_to_cartesian(
    catalog: pd.DataFrame, cosmology: Cosmology = Cosmology()
) -> NDArray[np.float64]:
    """Project RA and redshift into (N, 2) radial coordinates in Mpc/h.

    The equatorial projection intentionally ignores declination. Distances
    in Mpc/h are independent of the numerical value of H0 = 100 h km/s/Mpc.
    """
    if not {"ra", "z"}.issubset(catalog.columns):
        raise ValueError("catalog must contain ra and z columns")
    ra = np.asarray(catalog["ra"], dtype=np.float64)
    z = np.asarray(catalog["z"], dtype=np.float64)
    if not np.all(np.isfinite(ra)) or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("ra and non-negative redshift must be finite")
    distance = np.array([_distance(float(value), cosmology) for value in z])
    angle = np.deg2rad(ra)
    return np.column_stack((distance * np.cos(angle), distance * np.sin(angle)))


def _pair_counts(
    first: cKDTree, second: cKDTree, bins: NDArray[np.float64], auto: bool
) -> NDArray[np.float64]:
    cumulative = np.asarray(first.count_neighbors(second, bins), dtype=np.float64)
    # Differences remove self-pairs; auto counts both orders of each pair.
    return np.diff(cumulative) / (2.0 if auto else 1.0)


def compute_two_point_correlation(
    positions: ArrayLike,
    r_bins: ArrayLike,
    boxsize: float | None = None,
    *,
    cosmology: Cosmology = Cosmology(),
    random_seed: int = 42,
    random_multiplier: int = 5,
    redshift_jitter: float = 0.001,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return bin centers and xi for periodic particles or the SDSS wedge.

    Wedge randoms have uniform RA and bootstrap the observed redshift
    distribution with Gaussian jitter. This models radial selection, not the
    angular survey mask. Pair bins are (lower, upper]; empty RR bins are NaN.
    """
    points = np.asarray(positions, dtype=np.float64)
    bins = np.asarray(r_bins, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("positions must have shape (N, 2) with N >= 2")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite")
    if (
        bins.ndim != 1
        or len(bins) < 2
        or not np.all(np.isfinite(bins))
        or bins[0] < 0
        or np.any(np.diff(bins) <= 0)
    ):
        raise ValueError("r_bins must be finite, non-negative, and strictly increasing")
    centers = 0.5 * (bins[:-1] + bins[1:])
    n = len(points)
    if boxsize is not None:
        if not np.isfinite(boxsize) or boxsize <= 0 or bins[-1] > boxsize / 2:
            raise ValueError("boxsize must be positive and bins must not exceed boxsize/2")
        tree = cKDTree(np.mod(points, boxsize), boxsize=boxsize)
        dd = _pair_counts(tree, tree, bins, auto=True)
        expected = n * (n - 1) / 2 * np.pi * np.diff(bins**2) / boxsize**2
        return centers, dd / expected - 1.0

    if not isinstance(random_multiplier, int) or random_multiplier < 1:
        raise ValueError("random_multiplier must be a positive integer")
    if not np.isfinite(redshift_jitter) or redshift_jitter < 0:
        raise ValueError("redshift_jitter must be finite and non-negative")
    radii = np.linalg.norm(points, axis=1)
    angles = np.rad2deg(np.arctan2(points[:, 1], points[:, 0])) % 360.0
    z_grid = np.linspace(0.0, _Z_MAX, 8193)
    a = 1.0 / (1.0 + z_grid)
    e2 = cosmology.omega_m / a**3 + cosmology.omega_k / a**2 + cosmology.omega_lambda
    if not np.all(np.isfinite(e2)) or np.any(e2 <= 0):
        raise ValueError("cosmology has an invalid expansion rate")
    distances = _C_OVER_100 * cumulative_trapezoid(1.0 / np.sqrt(e2), z_grid, initial=0)
    min_radius = np.interp(_Z_MIN, z_grid, distances)
    tolerance = 1e-6
    if (
        np.any(angles < _RA_MIN - tolerance)
        or np.any(angles > _RA_MAX + tolerance)
        or np.any(radii < min_radius - tolerance)
        or np.any(radii > distances[-1] + tolerance)
    ):
        raise ValueError("positions must lie within the SDSS RA and redshift wedge")
    observed_z = np.interp(radii, distances, z_grid)
    rng = np.random.default_rng(random_seed)
    random_count = n * random_multiplier
    random_z = np.clip(
        rng.choice(observed_z, size=random_count)
        + rng.normal(0.0, redshift_jitter, random_count),
        _Z_MIN,
        _Z_MAX,
    )
    random_ra = np.deg2rad(rng.uniform(_RA_MIN, _RA_MAX, random_count))
    random_radius = np.interp(random_z, z_grid, distances)
    random_points = np.column_stack(
        (random_radius * np.cos(random_ra), random_radius * np.sin(random_ra))
    )
    data_tree = cKDTree(points)
    random_tree = cKDTree(random_points)
    dd = _pair_counts(data_tree, data_tree, bins, auto=True) / (n * (n - 1) / 2)
    dr = _pair_counts(data_tree, random_tree, bins, auto=False) / (n * random_count)
    rr = _pair_counts(random_tree, random_tree, bins, auto=True) / (
        random_count * (random_count - 1) / 2
    )
    xi = np.full_like(rr, np.nan)
    np.divide(dd - 2 * dr + rr, rr, out=xi, where=rr > 0)
    if np.any(rr == 0):
        warnings.warn("Some bins have no random pairs; xi is NaN", RuntimeWarning)
    return centers, xi


def load_sdss_catalog(source: str, cache_path: str | None = None) -> SDSSCatalog:
    """Load an SDSS catalog from a path or URL, optionally using a cache file."""
    raise NotImplementedError("SDSS catalog loading is not implemented yet.")


def correlation_function(
    catalog: SDSSCatalog, separation_bins: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Estimate galaxy clustering xi(r) for the supplied separation bins."""
    raise NotImplementedError("The galaxy clustering metric is not implemented yet.")
