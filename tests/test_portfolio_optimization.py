import numpy as np
import pandas as pd

from portfolio.optimization import annualized_covariance, annualized_mean_returns, portfolio_performance


def test_portfolio_performance_schema_is_finite() -> None:
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, -0.01, 0.005, 0.002],
            "MSFT": [0.002, 0.003, -0.001, 0.004],
        }
    )
    expected = annualized_mean_returns(returns)
    covariance = annualized_covariance(returns)
    ret, vol, sharpe = portfolio_performance(np.array([0.5, 0.5]), expected, covariance)

    assert np.isfinite(ret)
    assert np.isfinite(vol)
    assert np.isfinite(sharpe)
