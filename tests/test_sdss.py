"""Tests for SDSS loading, projection, and pair-counting estimators."""

import numpy as np
import pandas as pd
import pytest
import requests
from scipy.spatial.distance import cdist, pdist

from src.cosmology import Cosmology
from src.sdss_data import (
    compute_two_point_correlation,
    fetch_sdss_wedge,
    project_sdss_to_cartesian,
)


def test_projection_distance_increases_and_has_expected_units() -> None:
    catalog = pd.DataFrame({"ra": [150.0, 180.0, 210.0], "z": [0.01, 0.06, 0.12]})
    xy = project_sdss_to_cartesian(catalog)
    radii = np.linalg.norm(xy, axis=1)

    assert np.all(np.diff(radii) > 0)
    assert radii[0] == pytest.approx(29.91, abs=0.1)
    assert np.allclose(xy, project_sdss_to_cartesian(catalog, Cosmology(h=0.9)))


def test_periodic_uniform_and_clustered_particles() -> None:
    rng = np.random.default_rng(42)
    boxsize = 100.0
    bins = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
    uniform = rng.uniform(0, boxsize, (4000, 2))
    centers, xi = compute_two_point_correlation(uniform, bins, boxsize=boxsize)
    assert np.allclose(centers, (bins[:-1] + bins[1:]) / 2)
    assert np.all(np.abs(xi) < 0.08)

    cluster_centers = rng.uniform(0, boxsize, (30, 2))
    clustered = np.mod(
        np.repeat(cluster_centers, 100, axis=0) + rng.normal(0, 0.8, (3000, 2)),
        boxsize,
    )
    _, clustered_xi = compute_two_point_correlation(clustered, bins, boxsize=boxsize)
    assert clustered_xi[0] > 0


def test_periodic_pair_counts_exclude_self_pairs_and_wrap() -> None:
    points = np.array([[0.5, 5.0], [9.5, 5.0], [5.0, 5.0]])
    bins = np.array([0.0, 1.1, 4.8])
    _, xi = compute_two_point_correlation(points, bins, boxsize=10.0)
    expected = 3 * np.pi * np.diff(bins**2) / 100
    assert np.allclose(xi, np.array([1, 2]) / expected - 1)
    with pytest.raises(ValueError, match="boxsize/2"):
        compute_two_point_correlation(points, [0, 5.01], boxsize=10.0)


def test_wedge_landy_szalay_for_random_catalog_with_nonuniform_redshifts() -> None:
    rng = np.random.default_rng(19)
    count = 1500
    catalog = pd.DataFrame(
        {
            "ra": rng.uniform(150, 210, count),
            "z": np.clip(rng.normal(0.065, 0.012, count), 0.01, 0.12),
        }
    )
    positions = project_sdss_to_cartesian(catalog)
    bins = [3, 10, 20, 35, 50]
    r, xi = compute_two_point_correlation(
        positions, bins, random_seed=11, redshift_jitter=0
    )
    assert np.allclose(r, [6.5, 15, 27.5, 42.5])
    assert np.all(np.abs(xi) < 0.12)
    _, repeated = compute_two_point_correlation(
        positions, bins, random_seed=11, redshift_jitter=0
    )
    assert np.array_equal(xi, repeated)


def test_wedge_landy_szalay_matches_brute_force_pair_counts() -> None:
    catalog = pd.DataFrame(
        {"ra": [155, 165, 175, 185, 195, 205], "z": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]}
    )
    points = project_sdss_to_cartesian(catalog)
    bins = np.array([0.0, 20.0, 50.0, 100.0])
    random_count = len(points) * 50
    rng = np.random.default_rng(7)
    random_z = rng.choice(catalog["z"].to_numpy(), random_count)
    rng.normal(0.0, 0.0, random_count)
    random_ra = rng.uniform(150, 210, random_count)
    random_points = project_sdss_to_cartesian(pd.DataFrame({"ra": random_ra, "z": random_z}))

    def counts(distances):
        return np.array(
            [np.count_nonzero((distances > low) & (distances <= high))
             for low, high in zip(bins[:-1], bins[1:])]
        )

    dd = counts(pdist(points)) / (len(points) * (len(points) - 1) / 2)
    dr = counts(cdist(points, random_points)) / (len(points) * random_count)
    rr = counts(pdist(random_points)) / (random_count * (random_count - 1) / 2)
    _, xi = compute_two_point_correlation(
        points, bins, random_seed=7, random_multiplier=50, redshift_jitter=0
    )
    assert np.all(rr > 0)
    assert np.allclose(xi, (dd - 2 * dr + rr) / rr)


def test_wedge_empty_random_bins_and_invalid_geometry() -> None:
    points = project_sdss_to_cartesian(pd.DataFrame({"ra": [150, 210], "z": [0.01, 0.12]}))
    with pytest.warns(RuntimeWarning, match="no random pairs"):
        _, xi = compute_two_point_correlation(points, [0, 1e-10])
    assert np.isnan(xi[0])
    with pytest.raises(ValueError, match="wedge"):
        compute_two_point_correlation(np.array([[1.0, 0], [2.0, 0]]), [0, 1])


def test_offline_fallback_is_deterministic_and_does_not_create_cache(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cache.csv"

    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", offline)
    with pytest.warns(RuntimeWarning, match="offline"):
        first = fetch_sdss_wedge(cache_path=path)
    with pytest.warns(RuntimeWarning, match="offline"):
        second = fetch_sdss_wedge(cache_path=path)
    pd.testing.assert_frame_equal(first, second)
    assert not path.exists()
    assert len(first) > 1
    assert first["ra"].between(150, 210).all()
    assert first["dec"].between(-1.25, 1.25).all()
    assert first["z"].between(0.01, 0.12).all()
    assert {"g", "r", "g_minus_r"}.issubset(first.columns)
    assert np.allclose(first["g_minus_r"], first["g"] - first["r"])
    assert first["g_minus_r"].between(0.45, 0.85).all()


def test_legacy_cache_refreshes_and_derives_color_from_local_density(
    tmp_path, monkeypatch
) -> None:
    rng = np.random.default_rng(4)
    dense_cluster = pd.DataFrame(
        {
            "ra": rng.normal(180.0, 0.03, 24),
            "dec": rng.normal(0.0, 0.003, 24),
            "z": rng.normal(0.05, 0.0003, 24),
        }
    )
    sparse_filament = pd.DataFrame(
        {
            "ra": np.linspace(150, 210, 12),
            "dec": np.linspace(-1.0, 1.0, 12),
            "z": np.linspace(0.02, 0.11, 12),
        }
    )
    old_cache = pd.concat([dense_cluster, sparse_filament], ignore_index=True)
    path = tmp_path / "legacy.csv"
    old_cache.to_csv(path, index=False)

    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", offline)
    with pytest.warns(RuntimeWarning, match="offline"):
        catalog = fetch_sdss_wedge(cache_path=path)

    assert catalog.iloc[: len(dense_cluster)]["g_minus_r"].median() > (
        catalog.iloc[len(dense_cluster) :]["g_minus_r"].median()
    )
    assert catalog.iloc[: len(dense_cluster)]["g_minus_r"].median() > 0.75
    assert catalog.iloc[len(dense_cluster) :]["g_minus_r"].median() < 0.55
    assert list(pd.read_csv(path).columns) == ["ra", "dec", "z"]


def test_fetch_parses_skyserver_comment_header_and_reuses_cache(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cache.csv"
    calls = []

    class Response:
        text = "#Table1\nra,dec,z,g,r\n180.0,0.2,0.06,18.1,17.5\n"

        def raise_for_status(self):
            pass

    def get(url, *, params, timeout):
        calls.append((url, params, timeout))
        return Response()

    monkeypatch.setattr(requests, "get", get)
    first = fetch_sdss_wedge(cache_path=path)
    second = fetch_sdss_wedge(cache_path=path)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "g_minus_r"] == pytest.approx(0.6)
    assert len(calls) == 1
    assert calls[0][1]["format"] == "csv"
    assert "TOP 5000" in calls[0][1]["cmd"]
    assert "FROM SpecPhoto" in calls[0][1]["cmd"]
    assert "modelMag_g > 0 AND modelMag_r > 0" in calls[0][1]["cmd"]
    assert path.exists()

    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", offline)
    with pytest.warns(RuntimeWarning, match="offline"):
        refreshed = fetch_sdss_wedge(force_refresh=True, cache_path=path)
    pd.testing.assert_frame_equal(first, refreshed)
