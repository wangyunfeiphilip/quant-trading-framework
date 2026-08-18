from derivatives.black_scholes import OptionContract, black_scholes_greeks, black_scholes_price
from derivatives.delta_hedging import hedging_frequency_experiment, simulate_delta_hedge_path
from derivatives.numerical_methods import binomial_option_price, monte_carlo_option_price


def test_black_scholes_price_and_greeks_are_reasonable() -> None:
    contract = OptionContract(spot=100, strike=100, maturity=1, rate=0.05, volatility=0.2)
    price = black_scholes_price(contract)
    greeks = black_scholes_greeks(contract)

    assert 9.0 < price < 12.0
    assert 0.5 < greeks["delta"] < 0.7
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0


def test_binomial_tree_converges_toward_black_scholes() -> None:
    contract = OptionContract(spot=100, strike=100, maturity=1, rate=0.05, volatility=0.2)
    analytic = black_scholes_price(contract)
    tree = binomial_option_price(contract, steps=750)

    assert abs(tree - analytic) < 0.03


def test_monte_carlo_variance_reduction_reports_ratio() -> None:
    contract = OptionContract(spot=100, strike=100, maturity=1, rate=0.05, volatility=0.2)
    plain = monte_carlo_option_price(contract, n_paths=8000, seed=3)
    reduced = monte_carlo_option_price(contract, n_paths=8000, seed=3, antithetic=True, control_variate=True)

    assert abs(reduced.price - black_scholes_price(contract)) < 0.35
    assert reduced.standard_error < plain.standard_error
    assert reduced.variance_reduction_ratio > 1.0


def test_delta_hedging_experiment_outputs_error_and_cost() -> None:
    contract = OptionContract(spot=100, strike=100, maturity=0.25, rate=0.04, volatility=0.2)
    path, summary = simulate_delta_hedge_path(contract, hedge_interval_days=5, n_steps=40, seed=10)
    experiment = hedging_frequency_experiment(contract, hedge_intervals=(2, 5), n_paths=3, n_steps=40, seed=10)

    assert not path.empty
    assert "replication_error" in summary
    assert {"rmse_replication_error", "avg_transaction_cost", "sqrt_delta_t"}.issubset(experiment.columns)
