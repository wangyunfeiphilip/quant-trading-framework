"""Numerical option pricing methods and Monte Carlo diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
import pandas as pd

from derivatives.black_scholes import OptionContract, black_scholes_price


@dataclass(frozen=True)
class MonteCarloResult:
    """Monte Carlo price estimate and variance diagnostics."""

    price: float
    standard_error: float
    estimator_variance: float
    variance_reduction_ratio: float
    n_observations: int


def _payoff(spot: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(spot - strike, 0.0)
    return np.maximum(strike - spot, 0.0)


def binomial_option_price(contract: OptionContract, steps: int = 250) -> float:
    """Price a European option with the Cox-Ross-Rubinstein tree."""

    contract.validate()
    if steps <= 0:
        raise ValueError("steps must be positive")

    dt = contract.maturity / steps
    up = exp(contract.volatility * sqrt(dt))
    down = 1.0 / up
    discount = exp(-contract.rate * dt)
    growth = exp((contract.rate - contract.dividend_yield) * dt)
    probability = (growth - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("risk-neutral probability is outside [0, 1]")

    node_index = np.arange(steps + 1)
    terminal_spots = contract.spot * (up ** (steps - node_index)) * (down**node_index)
    option_values = _payoff(terminal_spots, contract.strike, contract.option_type)

    for _ in range(steps):
        option_values = discount * (probability * option_values[:-1] + (1.0 - probability) * option_values[1:])
    return float(option_values[0])


def _simulate_terminal_spots(contract: OptionContract, shocks: np.ndarray) -> np.ndarray:
    drift = (contract.rate - contract.dividend_yield - 0.5 * contract.volatility**2) * contract.maturity
    diffusion = contract.volatility * sqrt(contract.maturity) * shocks
    return contract.spot * np.exp(drift + diffusion)


def monte_carlo_option_price(
    contract: OptionContract,
    n_paths: int = 50_000,
    seed: int | None = None,
    antithetic: bool = False,
    control_variate: bool = False,
) -> MonteCarloResult:
    """Price a European option by risk-neutral Monte Carlo.

    Antithetic variates average payoffs from `Z` and `-Z`. The control variate
    uses discounted terminal stock value, whose expectation is known.
    """

    contract.validate()
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")

    rng = np.random.default_rng(seed)
    discount = exp(-contract.rate * contract.maturity)

    if antithetic:
        half = max(1, n_paths // 2)
        shocks = rng.standard_normal(half)
        up_spots = _simulate_terminal_spots(contract, shocks)
        down_spots = _simulate_terminal_spots(contract, -shocks)
        up_payoff = discount * _payoff(up_spots, contract.strike, contract.option_type)
        down_payoff = discount * _payoff(down_spots, contract.strike, contract.option_type)
        base_observations = 0.5 * (up_payoff + down_payoff)
        terminal_control = 0.5 * discount * (up_spots + down_spots)
        raw_for_ratio = up_payoff
    else:
        shocks = rng.standard_normal(n_paths)
        terminal_spots = _simulate_terminal_spots(contract, shocks)
        base_observations = discount * _payoff(terminal_spots, contract.strike, contract.option_type)
        terminal_control = discount * terminal_spots
        raw_for_ratio = base_observations

    observations = base_observations
    if control_variate:
        known_control_mean = contract.spot * exp(-contract.dividend_yield * contract.maturity)
        control_variance = np.var(terminal_control, ddof=1)
        if control_variance > 0:
            beta = np.cov(observations, terminal_control, ddof=1)[0, 1] / control_variance
            observations = observations - beta * (terminal_control - known_control_mean)

    raw_variance = float(np.var(raw_for_ratio, ddof=1))
    estimator_variance = float(np.var(observations, ddof=1))
    standard_error = sqrt(estimator_variance / len(observations))
    variance_reduction_ratio = raw_variance / estimator_variance if estimator_variance > 0 else np.inf

    return MonteCarloResult(
        price=float(np.mean(observations)),
        standard_error=float(standard_error),
        estimator_variance=estimator_variance,
        variance_reduction_ratio=float(variance_reduction_ratio),
        n_observations=int(len(observations)),
    )


def monte_carlo_convergence(
    contract: OptionContract,
    path_grid: tuple[int, ...] = (1_000, 2_500, 5_000, 10_000, 25_000, 50_000),
    seed: int = 7,
    antithetic: bool = True,
    control_variate: bool = True,
) -> pd.DataFrame:
    """Return convergence diagnostics against the Black-Scholes benchmark."""

    analytic = black_scholes_price(contract)
    rows = []
    for index, n_paths in enumerate(path_grid):
        result = monte_carlo_option_price(
            contract,
            n_paths=n_paths,
            seed=seed + index,
            antithetic=antithetic,
            control_variate=control_variate,
        )
        absolute_error = abs(result.price - analytic)
        rows.append(
            {
                "n_paths": n_paths,
                "estimate": result.price,
                "analytic_price": analytic,
                "absolute_error": absolute_error,
                "sqrt_n_error": absolute_error * sqrt(n_paths),
                "standard_error": result.standard_error,
                "variance_reduction_ratio": result.variance_reduction_ratio,
            }
        )
    return pd.DataFrame(rows)
