"""Focused tests for the dashboard's fixed physical heatmap scale."""

import numpy as np

from app import _base_zmax_from_density, _heatmap, _sdss_wedge_figure
from src.simulation import SimulationSnapshot


def test_base_density_scale_uses_99_5th_percentile_with_safe_floor() -> None:
    assert _base_zmax_from_density(np.zeros((4, 4))) == 0.4
    uniform_density = np.full((4, 4), 3.0)
    assert np.isclose(_base_zmax_from_density(uniform_density), np.log10(4.0))


def test_heatmap_uses_fixed_physical_density_limits_and_ticks() -> None:
    snapshot = SimulationSnapshot(
        redshift=0.0,
        scale_factor=1.0,
        positions=np.empty((0, 2)),
        momenta=np.empty((0, 2)),
        density_grid=np.array([[-0.5, 0.0], [1.0, 2.0]]),
        cosmic_age_gyr=13.8,
    )

    default_heatmap = _heatmap(
        snapshot, overlay_particles=False, contrast=1.0, base_zmax=1.35
    )
    assert default_heatmap.data[0].zmin == -0.5
    assert default_heatmap.data[0].zmax == 1.35
    assert default_heatmap.data[0].zauto is False
    colorscale = default_heatmap.data[0].colorscale
    assert colorscale[0][1] == "#000004"
    top_color = colorscale[-1][1].lstrip("#")
    assert int(top_color[0:2], 16) >= 240
    assert int(top_color[2:4], 16) >= 240
    assert default_heatmap.data[0].colorbar.title.text == "Matter Density"
    assert np.allclose(
        default_heatmap.data[0].colorbar.tickvals,
        np.log10([0.316, 1, 2, 4, 8, 16]),
    )
    assert list(default_heatmap.data[0].colorbar.ticktext) == [
        "0.3× (Void)",
        "1× (Avg)",
        "2×",
        "4×",
        "8×",
        "16×",
    ]

    higher_contrast_heatmap = _heatmap(
        snapshot, overlay_particles=False, contrast=1.5, base_zmax=1.35
    )
    assert higher_contrast_heatmap.data[0].zmax == 0.9
    assert list(higher_contrast_heatmap.data[0].colorbar.ticktext) == [
        "0.3× (Void)",
        "1× (Avg)",
        "2×",
        "4×",
    ]


def test_sdss_wedge_colors_markers_by_g_minus_r() -> None:
    color_index = np.array([0.4, 0.65, 0.9])
    figure = _sdss_wedge_figure(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        color_index,
    )
    marker = figure.data[0].marker

    assert np.array_equal(marker.color, color_index)
    assert marker.cmin == 0.35
    assert marker.cmax == 0.95
    assert marker.showscale is True
    assert marker.colorbar.title.text == "Galaxy Age (g - r Color)"
    assert list(marker.colorbar.tickvals) == [0.4, 0.65, 0.9]
    assert list(marker.colorbar.ticktext) == [
        "Younger (Blue)",
        "Middle-Aged",
        "Older (Red)",
    ]
