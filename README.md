# Cosmic Web Simulator

Cosmic Web Simulator is a Python project for exploring how matter forms
large-scale structures in the universe. Its 2D Particle-Mesh engine generates
cosmological initial conditions, evolves particles under periodic gravity,
and stores density snapshots from high redshift to the present.

## Project layout

```text
cosmic-web-sim/
├── data/                   # Cached SDSS coordinates and other local data
├── src/
│   ├── cosmology.py         # Expansion, growth, and matter power spectrum
│   ├── initial_conditions.py# Density field and initial particle positions
│   ├── solver.py            # Cloud-in-Cell mass grid and FFT gravity
│   ├── simulation.py        # Time integrator and saved snapshots
│   └── sdss_data.py         # SDSS loading and galaxy clustering
├── tests/                   # Automated tests
├── app.py                   # Streamlit web app entry point
└── requirements.txt         # Python dependencies
```

`data/` holds local caches. Large datasets should remain local and should
not be committed.

## Setup

Use Python 3.10 or later. From the project directory, create and activate a
virtual environment:

```bash
python -m venv .venv
```

On Windows, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the starter app

Start the Streamlit app from the project directory:

```bash
streamlit run app.py
```

The app currently shows an introduction and implementation status.

## Run tests

Run the test suite with:

```bash
python -m pytest
```

The tests cover periodic mass conservation, force interpolation, growth of
structure, SDSS projection and offline loading, and clustering estimators.

## Engine architecture

- **Cosmology:** calculate H(a), linear growth D(a), and a sigma8-normalized
  transfer-function matter power spectrum P(k).
- **Initial conditions:** create a periodic 2D Gaussian density field and
  initialize particle positions and canonical momenta with the Zel'dovich
  approximation.
- **Solver:** deposit particle mass with periodic Cloud-in-Cell interpolation
  and calculate acceleration with an FFT Poisson solver.
- **Simulation:** integrate canonical momentum and comoving positions with a
  scale-factor Kick-Drift-Kick scheme, saving density and cosmic-age snapshots
  from z=50 to z=0.
- **SDSS data:** load observed galaxy coordinates (right ascension,
  declination, and redshift) and estimate the clustering statistic xi(r).

The SDSS data-analysis module remains a separate observational component.

## SDSS wedge and clustering

`fetch_sdss_wedge()` downloads up to 5,000 confirmed DR17 galaxies and caches
them in `data/sdss_slice_cache.csv`. Existing caches are used without a network
request; pass `force_refresh=True` to retry the download. If the network fails,
the loader warns and uses the existing cache or a deterministic mock catalog.
Mock data are never saved as observed data.

`project_sdss_to_cartesian(catalog)` returns equatorial positions in Mpc/h.
`compute_two_point_correlation(positions, r_bins, boxsize=L)` estimates
periodic-box clustering for bin edges no larger than L/2. Omitting `boxsize`
uses the SDSS wedge and a seeded Landy-Szalay estimator: random points have
uniform RA and resample observed redshifts with Gaussian jitter. This models
the radial selection but not the actual angular survey mask; results are not
a complete observational selection-function correction. Bins without random
pairs return NaN with a warning.
