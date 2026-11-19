# Engineering & Astrophysics Learnings: Cosmic Web Simulator

A living log of the physical principles, numerical methods, and architectural trade-offs behind simulating large-scale cosmic structure from $z = 50$ to the present day.

---

## Phase 1: Architecture & Computational Strategy

### 1.1 Particle-Mesh (PM) FFT Gravity vs. Direct $O(N^2)$ N-Body
* **The Physics:** Gravity is an infinite-range force—every dark matter particle in the universe pulls on every other particle simultaneously.
* **The Engineering Decision:** Calculating pairwise gravitational forces directly across $N = 50{,}000$ particles requires $N(N-1)/2 \approx 1.25 \times 10^9$ force evaluations *per time step*. Instead, we use a **Particle-Mesh (PM)** method: particles deposit mass onto a discrete grid, and we solve Poisson's equation ($\nabla^2 \Phi = 4\pi G \rho$) in Fourier space using Fast Fourier Transforms (`np.fft.rfft2`).
* **Key Takeaway:** Transforming differential equations into algebraic multiplications in Fourier space drops computational complexity from $O(N^2)$ to $O(N \log N)$, turning an hours-long cluster calculation into a sub-2-second interactive browser simulation.

### 1.2 Comoving Coordinates & Scale Factor ($a$) Stepping
* **The Physics:** As gravity pulls matter inward to form galaxies, the metric of space itself is expanding outward according to the Friedmann equation. Tracking absolute physical distances forces the integrator to resolve massive background expansion velocities just to capture tiny local gravitational motions.
* **The Engineering Decision:** Solve the equations of motion in **comoving coordinates**—a coordinate grid that stretches alongside cosmic expansion—and step the simulation forward using the cosmic scale factor $a = \frac{1}{1+z}$ (from $a \approx 0.02$ at $z = 50$ to $a = 1.0$ today) rather than raw seconds.
* **Key Takeaway:** Factoring out background expansion keeps the simulation box size constant ($L_{\text{box}}$ in $\text{Mpc}/h$) and isolates **peculiar velocities** (motions caused purely by local gravitational clustering).

### 1.3 Cloud-in-Cell (CIC) Interpolation vs. Nearest Grid Point (NGP)
* **The Physics:** Particles move through continuous space, but FFT gravity solvers operate on discrete grid cells.
* **The Engineering Decision:** Instead of assigning 100% of a particle's mass to the single closest grid node (Nearest Grid Point), we use **Cloud-in-Cell (CIC)** bilinear weighting to distribute each particle's mass smoothly across its 4 surrounding grid corners (and interpolate forces back the exact same way).
* **Key Takeaway:** CIC eliminates artificial "force jumps" when particles cross grid cell boundaries, strictly conserves total mass and momentum, and naturally produces smooth continuous density fields for logarithmic heatmap rendering.

### 1.4 The Zel'dovich Approximation for Initial Conditions
* **The Physics:** At $z = 50$ (roughly 50 million years after the Big Bang), the universe was not pure white noise; primordial quantum fluctuations stretched by inflation had already established faint density ripples and coherent velocity flows.
* **The Engineering Decision:** Generate a Gaussian random field filtered by the cosmological **Matter Power Spectrum** $P(k)$, then displace particles slightly from a uniform lattice along the initial gravitational potential gradient ($\mathbf{x} = \mathbf{q} + D(a)\mathbf{\Psi}(\mathbf{q})$).
* **Key Takeaway:** Starting particles with physically consistent initial velocities prevents artificial acoustic shockwaves at $t = 0$ and allows smooth filaments to form rapidly.

### 1.5 Observational Grounding: SDSS Sky Coordinates & $\xi(r)$
* **The Physics:** Visually convincing filaments do not prove physical accuracy. A valid cosmological model must reproduce the statistical clumpiness of real galaxies across different separation scales.
* **The Engineering Decision:** Pair the simulation with real galaxy positions from the **Sloan Digital Sky Survey (SDSS)**—converting celestial longitude (**Right Ascension**), latitude (**Declination**), and depth (**Redshift** $z$) into comoving Cartesian coordinates—and compare their **Two-Point Correlation Functions** $\xi(r)$.
* **Key Takeaway:** $\xi(r)$ measures the excess probability of finding two galaxies separated by distance $r$ compared to a random Poisson distribution, giving non-technical and technical audiences alike a direct quantitative bridge between code and telescope data.

---

## Phase 2: Initial Conditions & 2D Calibration

### 2.1 Real-Space White Noise Seeding (Avoiding Fourier Striping)
* **The Catch:** Generating random numbers directly in 2D Fourier space (`rfft2`) requires strict complex-conjugate symmetry across grid axes. Missing a single edge case produces artificial diagonal plaid stripes instead of random cosmic ripples.
* **The Fix:** Generate standard Gaussian white noise in **real space** first (`rng.normal`), transform it into Fourier space, shape it with the power spectrum $P(k)$, and transform back. Symmetry is guaranteed automatically.

### 2.2 Calibrating 2D Ripples at $z = 50$
* **The Catch:** Standard 3D cosmology formulas ($\sigma_8$) don't map cleanly onto a flat 2D toy universe—naive scaling either leaves the universe frozen or sends particles flying across the box on Step 1.
* **The Fix:** Scale the initial Zel'dovich particle nudges at $z = 50$ so the average particle moves by ~0.15 grid cells. This keeps the early universe smoothly linear while guaranteeing full cosmic web collapse by $z = 0$.

---

## Phase 3: Gravity Solver & Expanding-Space Integration

### 3.1 Hubble Drag & Canonical Momentum ($\mathbf{p} = a^2 \dot{\mathbf{x}}$)
* **The Intuition:** Imagine firing a particle across a stretching rubber sheet of galaxies. As it coasts forward, it overtakes galaxies that are receding faster than the ones it left behind—so relative to the local grid, its speed naturally drops.
* **The Mechanics:** Using standard `x += v * dt` deletes this cosmic friction, causing particles to overshoot galaxy clusters like pinballs instead of settling into sharp filaments. By evolving **canonical momentum** ($\mathbf{p} = a^2 \dot{\mathbf{x}}$) in our Kick-Drift-Kick integrator, Hubble drag is built into the math automatically.

### 3.2 Zero Self-Force & Center-of-Mass Conservation
* **The Intuition:** Because our simulation box is a closed universe with no outside forces, Newton's 3rd Law requires that every internal pull cancels out ($\sum \mathbf{F}_i = 0$). The universe's center of mass must stay locked in place.
* **The Grid Trick:** Smearing a particle across 4 grid corners (CIC) can cause an off-center particle to pull on its own smeared mass if the force math doesn't match the mass-assignment math. Computing gravitational forces at the grid corners *first* and interpolating back with the exact same CIC weights guarantees zero self-force.
* **The Canary Test:** Tracking total momentum ($\sum \mathbf{p}_i \approx 0$) in `pytest` immediately catches off-by-one array bugs or broken periodic wrapping.

## Phase 4: Observational Validation & Visualization *(Upcoming)*
* *To be documented once `src/sdss_data.py` and `app.py` are implemented:*
  * Logarithmic density contrast scaling ($\log_{10}(1 + \delta)$) for rendering high-dynamic-range filaments and voids simultaneously.