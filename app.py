"""Interactive Streamlit dashboard for the 2D Cosmic Web Simulator."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.cosmology import Cosmology
from src.initial_conditions import generate_initial_conditions
from src.sdss_data import (
    compute_two_point_correlation,
    fetch_sdss_wedge,
    project_sdss_to_cartesian,
)
from src.simulation import Simulation, SimulationSnapshot


BOX_SIZE = 100.0
SNAPSHOT_COUNT = 51
PARTICLE_DISPLAY_LIMIT = 15_000
RESOLUTIONS = {
    "128 - Fast": 128,
    "160 - Balanced": 160,
    "192 - High Detail": 192,
}
SCENARIOS = {
    "1. Our Universe (ΛCDM Standard)": {
        "omega_m": 0.31,
        "omega_lambda": 0.69,
        "dark_matter": True,
        "description": (
            "ΛCDM: dark matter builds sharp cosmic filaments around deep, "
            "quiet voids."
        ),
    },
    "2. No Dark Matter (Normal Matter Only)": {
        "omega_m": 0.05,
        "omega_lambda": 0.95,
        "dark_matter": False,
        "description": (
            "Silk damping smooths small-scale structure; weak gravity leaves "
            "a comparatively featureless fog."
        ),
    },
    "3. Heavy Matter Universe (No Dark Energy)": {
        "omega_m": 1.0,
        "omega_lambda": 0.0,
        "dark_matter": True,
        "description": (
            "With no dark energy, matter dominates the expansion and forms "
            "strong, dense clusters."
        ),
    },
    "Custom": {
        "description": "Set the matter and dark-energy densities yourself.",
    },
}


@st.cache_data(show_spinner="Simulating 13.8 billion years of gravity...")
def run_simulation(
    omega_m: float,
    omega_lambda: float,
    dark_matter: bool,
    resolution: int,
) -> list[SimulationSnapshot]:
    """Create and evolve a seeded, one-particle-per-cell universe."""
    cosmology = Cosmology(omega_m=omega_m, omega_lambda=omega_lambda)
    positions, momenta, _ = generate_initial_conditions(
        resolution,
        BOX_SIZE,
        seed=7,
        parameters=cosmology,
        dark_matter=dark_matter,
    )
    simulation = Simulation(
        positions,
        momenta,
        box_size=BOX_SIZE,
        grid_size=resolution,
        parameters=cosmology,
    )
    return simulation.run(snapshot_count=SNAPSHOT_COUNT)


@st.cache_data(show_spinner="Loading Sloan Digital Sky Survey observations...")
def load_sdss_data(
    omega_m: float, omega_lambda: float
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load cached or live SDSS observations and project them into the wedge."""
    catalog = fetch_sdss_wedge()
    cosmology = Cosmology(omega_m=omega_m, omega_lambda=omega_lambda)
    return catalog, project_sdss_to_cartesian(catalog, cosmology)


def _base_zmax_from_density(density_grid: np.ndarray) -> float:
    log_density = np.log10(np.maximum(1.0 + density_grid, 0.1))
    return max(float(np.percentile(log_density, 99.5)), 0.4)


@st.cache_data(show_spinner="Calibrating the physical density scale...")
def _calibrate_base_zmax(resolution: int) -> float:
    standard_snapshots = run_simulation(0.31, 0.69, True, resolution)
    return _base_zmax_from_density(standard_snapshots[-1].density_grid)


@st.cache_data(show_spinner=False)
def _measure_periodic_clustering(
    positions: np.ndarray, bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return compute_two_point_correlation(positions, bins, boxsize=BOX_SIZE)


@st.cache_data(show_spinner=False)
def _measure_wedge_clustering(
    positions: np.ndarray,
    bins: np.ndarray,
    omega_m: float,
    omega_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    cosmology = Cosmology(omega_m=omega_m, omega_lambda=omega_lambda)
    return compute_two_point_correlation(
        positions, bins, cosmology=cosmology, random_seed=42
    )


def _apply_preset() -> None:
    preset = st.session_state["scenario_preset"]
    values = SCENARIOS[preset]
    if preset != "Custom":
        st.session_state["omega_m"] = values["omega_m"]
        st.session_state["omega_lambda"] = values["omega_lambda"]
        st.session_state["dark_matter"] = values["dark_matter"]


def _configure_theme() -> None:
    pio.templates.default = "plotly_dark"
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp {
            background: #05050A;
            color: #E8EAF2;
        }
        [data-testid="stHeader"] { background: rgba(5, 5, 10, 0.85); }
        [data-testid="stSidebar"] {
            background: #090A12;
            border-right: 1px solid #202332;
        }
        [data-testid="stMetric"] {
            background: #0B0D16;
            border: 1px solid #25293A;
            border-radius: 0.75rem;
            padding: 1rem 1.1rem;
        }
        h1, h2, h3 { letter-spacing: -0.025em; }
        div[data-testid="stAlert"] { border: 1px solid #25293A; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _heatmap(
    snapshot: SimulationSnapshot,
    overlay_particles: bool,
    contrast: float,
    base_zmax: float,
) -> go.Figure:
    grid_size = snapshot.density_grid.shape[0]
    log_density = np.log10(np.maximum(1.0 + snapshot.density_grid.T, 0.1))
    zmax = base_zmax / contrast
    density_multipliers = np.array([0.316, 1.0, 2.0, 4.0, 8.0, 16.0])
    density_labels = ["0.3× (Void)", "1× (Avg)", "2×", "4×", "8×", "16×"]
    tick_values = np.log10(density_multipliers)
    visible_ticks = tick_values <= zmax
    figure = go.Figure(
        go.Heatmap(
            z=log_density,
            zmin=-0.5,
            zmax=zmax,
            zauto=False,
            colorscale="inferno",
            colorbar={
                "title": "Matter Density",
                "tickvals": tick_values[visible_ticks],
                "ticktext": np.asarray(density_labels)[visible_ticks],
            },
            hovertemplate="x=%{x}<br>y=%{y}<br>brightness=%{z:.2f}<extra></extra>",
        )
    )
    if overlay_particles and len(snapshot.positions):
        particle_positions = snapshot.positions
        if len(particle_positions) > PARTICLE_DISPLAY_LIMIT:
            rng = np.random.default_rng(42)
            selected = rng.choice(
                len(particle_positions), PARTICLE_DISPLAY_LIMIT, replace=False
            )
            particle_positions = particle_positions[selected]
        cell_positions = particle_positions / BOX_SIZE * grid_size
        figure.add_trace(
            go.Scattergl(
                x=cell_positions[:, 0],
                y=cell_positions[:, 1],
                mode="markers",
                marker={"size": 2, "color": "rgba(165, 235, 255, 0.35)"},
                hoverinfo="skip",
                name="Dark matter particles",
            )
        )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        height=650,
        showlegend=overlay_particles,
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "scaleanchor": "y",
            "scaleratio": 1,
        },
    )
    return figure


def _sdss_wedge_figure(
    positions: np.ndarray, g_minus_r: np.ndarray
) -> go.Figure:
    color_index = np.asarray(g_minus_r, dtype=np.float64)
    if color_index.shape != (len(positions),) or not np.all(np.isfinite(color_index)):
        raise ValueError("g_minus_r must contain one finite value per SDSS galaxy")
    figure = go.Figure(
        go.Scattergl(
            x=positions[:, 0],
            y=positions[:, 1],
            mode="markers",
            marker={
                "size": 3,
                "color": color_index,
                "colorscale": ["#38BDF8", "#F8FAFC", "#F97316", "#EF4444"],
                "cmin": 0.35,
                "cmax": 0.95,
                "opacity": 0.75,
                "showscale": True,
                "colorbar": {
                    "title": "Galaxy Age (g - r Color)",
                    "tickvals": [0.4, 0.65, 0.9],
                    "ticktext": ["Younger (Blue)", "Middle-Aged", "Older (Red)"],
                },
            },
            hovertemplate=(
                "X=%{x:.1f} Mpc/h<br>Y=%{y:.1f} Mpc/h"
                "<br>g - r=%{marker.color:.2f}<extra></extra>"
            ),
            name="SDSS galaxies",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 12, "r": 12, "t": 35, "b": 12},
        height=460,
        title=f"DR17 equatorial wedge · {len(positions):,} galaxies",
        xaxis={
            "title": "X (Mpc/h)",
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={"title": "Y (Mpc/h)", "showgrid": False, "zeroline": False},
    )
    return figure


def _clustering_figure(
    simulation_positions: np.ndarray,
    sdss_positions: np.ndarray,
    omega_m: float,
    omega_lambda: float,
) -> go.Figure:
    bins = np.geomspace(0.5, BOX_SIZE / 2.0, 17)
    sim_r, sim_xi = _measure_periodic_clustering(simulation_positions, bins)
    sdss_r, sdss_xi = _measure_wedge_clustering(
        sdss_positions, bins, omega_m, omega_lambda
    )
    figure = go.Figure()
    for label, radii, values, color in (
        ("Simulation · z = 0", sim_r, sim_xi, "#66D9EF"),
        ("SDSS DR17 wedge", sdss_r, sdss_xi, "#FFB86C"),
    ):
        positive = np.isfinite(values) & (values > 0)
        figure.add_trace(
            go.Scatter(
                x=radii[positive],
                y=values[positive],
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 6},
                name=label,
            )
        )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=430,
        margin={"l": 12, "r": 12, "t": 18, "b": 12},
        xaxis={"title": "Separation r (Mpc/h)", "type": "log"},
        yaxis={"title": "Positive ξ(r)", "type": "log"},
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def main() -> None:
    st.set_page_config(
        page_title="Cosmic Web Simulator", page_icon="🌌", layout="wide"
    )
    _configure_theme()

    st.session_state.setdefault(
        "scenario_preset", "1. Our Universe (ΛCDM Standard)"
    )
    st.session_state.setdefault("omega_m", 0.31)
    st.session_state.setdefault("omega_lambda", 0.69)
    st.session_state.setdefault("dark_matter", True)
    st.session_state.setdefault("resolution_label", "160 - Balanced")

    st.sidebar.title("Universe Controls")
    st.sidebar.selectbox(
        "Demo Scenario Preset",
        list(SCENARIOS),
        key="scenario_preset",
        on_change=_apply_preset,
    )
    scenario = st.session_state["scenario_preset"]
    st.sidebar.caption(SCENARIOS[scenario]["description"])
    custom = scenario == "Custom"
    st.sidebar.slider(
        "Matter Density (Ωₘ)",
        min_value=0.01,
        max_value=1.0,
        step=0.01,
        key="omega_m",
        disabled=not custom,
    )
    st.sidebar.slider(
        "Dark Energy (ΩΛ)",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="omega_lambda",
        disabled=not custom,
    )
    st.sidebar.checkbox(
        "Include Dark Matter",
        key="dark_matter",
        disabled=not custom,
    )
    st.sidebar.selectbox(
        "Simulation Resolution",
        list(RESOLUTIONS),
        key="resolution_label",
        disabled=not custom,
        help="One particle per grid cell; higher detail increases compute time.",
    )
    resolution = RESOLUTIONS[st.session_state["resolution_label"]]

    st.title("🌌 Cosmic Web Simulator")
    st.caption(
        "Watch gravity shape a universe, then compare its 2D structure with "
        "galaxies observed by the Sloan Digital Sky Survey."
    )

    snapshots = run_simulation(
        float(st.session_state["omega_m"]),
        float(st.session_state["omega_lambda"]),
        bool(st.session_state["dark_matter"]),
        resolution,
    )
    base_zmax = _calibrate_base_zmax(resolution)
    evolution_tab, reality_tab = st.tabs(
        ["🌌 Evolving the Cosmic Web", "🔭 Reality Check: Real SDSS Telescope Data"]
    )

    with evolution_tab:
        timeline_col, timeline_value_col = st.columns([4, 1])
        with timeline_col:
            snapshot_index = st.slider(
                "Cosmic Time Scrubber · early universe → today",
                min_value=0,
                max_value=len(snapshots) - 1,
                value=len(snapshots) - 1,
                key="timeline_index",
            )
        snapshot = snapshots[snapshot_index]
        with timeline_value_col:
            st.caption(f"Selected epoch\n\nz = {snapshot.redshift:.2f}")

        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Cosmic Age (Billion Years)", f"{snapshot.cosmic_age_gyr:.2f}"
        )
        metric_columns[1].metric("Redshift (z)", f"{snapshot.redshift:.2f}")
        metric_columns[2].metric("Universe Size (a)", f"{snapshot.scale_factor:.4f}")
        metric_columns[3].metric(
            "Peak Clumpiness (max δ)", f"{np.max(snapshot.density_grid):.2f}"
        )

        overlay_col, contrast_col = st.columns([1, 3])
        with overlay_col:
            overlay_particles = st.checkbox(
                "Overlay Dark Matter Particles", value=False
            )
        with contrast_col:
            contrast = st.slider(
                "Glow Contrast", min_value=0.5, max_value=2.5, value=1.0, step=0.1
            )
        st.plotly_chart(
            _heatmap(snapshot, overlay_particles, contrast, base_zmax),
            key="timeline-density-heatmap",
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with reality_tab:
        cosmology = Cosmology(
            omega_m=float(st.session_state["omega_m"]),
            omega_lambda=float(st.session_state["omega_lambda"]),
        )
        catalog, sdss_positions = load_sdss_data(
            cosmology.omega_m, cosmology.omega_lambda
        )
        today = snapshots[-1]
        simulation_col, sdss_col = st.columns(2)
        with simulation_col:
            st.subheader("Simulated universe · z = 0")
            st.plotly_chart(
                _heatmap(today, False, 1.0, base_zmax),
                key="comparison-simulation-heatmap",
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                f"{len(today.positions):,} particles · {resolution}² mesh · "
                f"{BOX_SIZE:g} Mpc/h periodic box"
            )
        with sdss_col:
            st.subheader("Observed galaxy wedge")
            st.plotly_chart(
                _sdss_wedge_figure(
                    sdss_positions,
                    catalog["g_minus_r"].to_numpy(dtype=np.float64),
                ),
                key="comparison-sdss-wedge",
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "Dense web intersections tend to hold ancient red galaxies, "
                "while younger, blue star-forming galaxies are more common "
                "along thinner strands. This is a 2D RA–redshift projection; "
                "declination and the full survey angular mask are not represented."
            )

        st.subheader("The Clumpiness Ruler · two-point correlation ξ(r)")
        st.plotly_chart(
            _clustering_figure(
                today.positions,
                sdss_positions,
                cosmology.omega_m,
                cosmology.omega_lambda,
            ),
            key="comparison-clustering-correlation",
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.info(
            "If the simulation curve follows the telescope curve downward "
            "from left to right, that is evidence of similar clustering "
            "across scales—not proof that the model reproduces the real "
            "universe. This is a 2D periodic simulation compared with a "
            "projected SDSS wedge, and the full survey selection is not modeled. "
            "Only positive ξ values are shown on the log-log plot."
        )


if __name__ == "__main__":
    main()
