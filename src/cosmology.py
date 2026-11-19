"""Cosmological background and linear-theory calculations."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson


@dataclass(frozen=True)
class Cosmology:
    """Flat or curved matter-plus-lambda cosmology; H is in H0 units."""

    omega_m: float = 0.31
    omega_lambda: float = 0.69
    h: float = 0.68
    sigma_8: float = 0.81
    n_s: float = 0.965

    @property
    def omega_k(self) -> float:
        return 1.0 - self.omega_m - self.omega_lambda


@dataclass(frozen=True)
class CosmologyParameters:
    """Backward-compatible parameter names for the original module API."""

    h0: float = 68.0
    omega_matter: float = 0.31
    omega_lambda: float = 0.69
    sigma8: float = 0.81
    spectral_index: float = 0.965

    def as_cosmology(self) -> Cosmology:
        return Cosmology(
            omega_m=self.omega_matter,
            omega_lambda=self.omega_lambda,
            h=self.h0 / 100.0,
            sigma_8=self.sigma8,
            n_s=self.spectral_index,
        )


CosmologyLike = Cosmology | CosmologyParameters


def _as_cosmology(parameters: CosmologyLike) -> Cosmology:
    return parameters.as_cosmology() if isinstance(parameters, CosmologyParameters) else parameters


def expansion_rate(
    scale_factor: float,
    parameters: CosmologyLike = Cosmology(),
) -> float:
    """Return E(a) = H(a)/H0 for a positive scale factor."""
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be positive")
    model = _as_cosmology(parameters)
    a = float(scale_factor)
    e_squared = (
        model.omega_m / a**3
        + model.omega_k / a**2
        + model.omega_lambda
    )
    if e_squared <= 0.0:
        raise ValueError("The cosmological expansion rate is not real for this scale factor")
    return float(np.sqrt(e_squared))


def growth_factor(
    scale_factor: float,
    parameters: CosmologyLike = Cosmology(),
) -> float:
    """Return the linear growth approximation normalized to D(1) = 1."""
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be positive")
    model = _as_cosmology(parameters)

    def suppression(a: float) -> float:
        e2 = (
            model.omega_m / a**3
            + model.omega_k / a**2
            + model.omega_lambda
        )
        omega_m_a = (model.omega_m / a**3) / e2
        omega_lambda_a = model.omega_lambda / e2
        numerator = 2.5 * omega_m_a
        denominator = (
            omega_m_a ** (4.0 / 7.0)
            - omega_lambda_a
            + (1.0 + omega_m_a / 2.0) * (1.0 + omega_lambda_a / 70.0)
        )
        return numerator / denominator

    return float(scale_factor * suppression(scale_factor) / suppression(1.0))


def growth_rate(
    scale_factor: float,
    parameters: CosmologyLike = Cosmology(),
) -> float:
    """Return f(a) = d ln D / d ln a using the usual gamma approximation."""
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be positive")
    model = _as_cosmology(parameters)
    a = float(scale_factor)
    e2 = (
        model.omega_m / a**3
        + model.omega_k / a**2
        + model.omega_lambda
    )
    omega_m_a = (model.omega_m / a**3) / e2
    return float(omega_m_a**0.55)


def transfer_function(
    wavenumber: ArrayLike,
    parameters: CosmologyLike = Cosmology(),
    dark_matter: bool = True,
    silk_damping_scale: float = 10.0,
) -> NDArray[np.float64] | float:
    """Evaluate the BBKS transfer function with an optional 2D Silk cutoff.

    Wavenumbers use h/Mpc. ``silk_damping_scale`` is in the corresponding
    inverse-length units and only applies when ``dark_matter`` is false.
    """
    model = _as_cosmology(parameters)
    k = np.asarray(wavenumber, dtype=np.float64)
    if np.any(k < 0.0):
        raise ValueError("wavenumber must be non-negative")
    if not dark_matter and silk_damping_scale <= 0.0:
        raise ValueError("silk_damping_scale must be positive")

    q = k / (model.omega_m * model.h)
    with np.errstate(divide="ignore", invalid="ignore"):
        logarithm = np.log1p(2.34 * q) / (2.34 * q)
        polynomial = (
            1.0
            + 3.89 * q
            + (16.1 * q) ** 2
            + (5.46 * q) ** 3
            + (6.71 * q) ** 4
        ) ** -0.25
        transfer = logarithm * polynomial
    transfer = np.where(k == 0.0, 1.0, transfer)
    if not dark_matter:
        transfer = transfer * np.exp(-((k / silk_damping_scale) ** 1.4))
    if np.ndim(wavenumber) == 0:
        return float(transfer)
    return np.asarray(transfer, dtype=np.float64)


@lru_cache(maxsize=32)
def _power_normalization(
    model: Cosmology,
    dark_matter: bool,
    silk_damping_scale: float,
) -> float:
    k = np.logspace(-5.0, 3.0, 2048)
    transfer = np.asarray(
        transfer_function(k, model, dark_matter, silk_damping_scale)
    )
    x = 8.0 * k
    window = np.ones_like(x)
    nonzero = x > 1e-5
    window[nonzero] = (
        3.0
        * (np.sin(x[nonzero]) - x[nonzero] * np.cos(x[nonzero]))
        / x[nonzero] ** 3
    )
    integral = simpson(k ** (model.n_s + 2.0) * transfer**2 * window**2, x=k)
    return float(model.sigma_8**2 * 2.0 * np.pi**2 / integral)


def matter_power_spectrum(
    wavenumber: ArrayLike,
    parameters: CosmologyLike = Cosmology(),
    dark_matter: bool = True,
    silk_damping_scale: float = 10.0,
) -> NDArray[np.float64] | float:
    """Return sigma8-normalized P(k) proportional to k**n_s T(k)**2."""
    model = _as_cosmology(parameters)
    k = np.asarray(wavenumber, dtype=np.float64)
    if np.any(k < 0.0):
        raise ValueError("wavenumber must be non-negative")
    transfer = np.asarray(
        transfer_function(k, model, dark_matter, silk_damping_scale)
    )
    amplitude = _power_normalization(model, dark_matter, silk_damping_scale)
    power = amplitude * k**model.n_s * transfer**2
    power = np.where(k == 0.0, 0.0, power)
    if np.ndim(wavenumber) == 0:
        return float(power)
    return np.asarray(power, dtype=np.float64)


