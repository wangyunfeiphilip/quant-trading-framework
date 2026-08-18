"""Black-Scholes pricing and Greeks for European options."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Literal

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionContract:
    """European option contract under geometric Brownian motion."""

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0
    option_type: OptionType = "call"

    def validate(self) -> None:
        """Validate numerical inputs before pricing."""

        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.volatility <= 0:
            raise ValueError("volatility must be positive")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1_d2(contract: OptionContract) -> tuple[float, float]:
    contract.validate()
    sigma_sqrt_t = contract.volatility * sqrt(contract.maturity)
    carry = contract.rate - contract.dividend_yield
    d1 = (log(contract.spot / contract.strike) + (carry + 0.5 * contract.volatility**2) * contract.maturity) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return d1, d2


def black_scholes_price(contract: OptionContract) -> float:
    """Return the Black-Scholes European option value."""

    d1, d2 = _d1_d2(contract)
    discounted_spot = contract.spot * exp(-contract.dividend_yield * contract.maturity)
    discounted_strike = contract.strike * exp(-contract.rate * contract.maturity)

    if contract.option_type == "call":
        return discounted_spot * _norm_cdf(d1) - discounted_strike * _norm_cdf(d2)
    return discounted_strike * _norm_cdf(-d2) - discounted_spot * _norm_cdf(-d1)


def black_scholes_greeks(contract: OptionContract) -> dict[str, float]:
    """Return delta, gamma, vega, theta, and rho."""

    d1, d2 = _d1_d2(contract)
    q_discount = exp(-contract.dividend_yield * contract.maturity)
    r_discount = exp(-contract.rate * contract.maturity)
    pdf_d1 = _norm_pdf(d1)

    gamma = q_discount * pdf_d1 / (contract.spot * contract.volatility * sqrt(contract.maturity))
    vega = contract.spot * q_discount * pdf_d1 * sqrt(contract.maturity)

    common_theta = (
        -contract.spot * q_discount * pdf_d1 * contract.volatility / (2.0 * sqrt(contract.maturity))
    )
    if contract.option_type == "call":
        delta = q_discount * _norm_cdf(d1)
        theta = (
            common_theta
            - contract.rate * contract.strike * r_discount * _norm_cdf(d2)
            + contract.dividend_yield * contract.spot * q_discount * _norm_cdf(d1)
        )
        rho = contract.strike * contract.maturity * r_discount * _norm_cdf(d2)
    else:
        delta = q_discount * (_norm_cdf(d1) - 1.0)
        theta = (
            common_theta
            + contract.rate * contract.strike * r_discount * _norm_cdf(-d2)
            - contract.dividend_yield * contract.spot * q_discount * _norm_cdf(-d1)
        )
        rho = -contract.strike * contract.maturity * r_discount * _norm_cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }
