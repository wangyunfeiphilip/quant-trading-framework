# Model Comparison

The machine learning module supports three return prediction models:

- Linear Regression
- Random Forest
- Gradient Boosting

## Evaluation Protocol

The target is future 21-trading-day return. Features include technical indicators, return momentum, volatility, and available fundamental snapshots.

Validation should use chronological train/test split and time-series cross validation. Random shuffling is avoided because it can leak future market regimes into the training set.

## Expected Output

After execution, record:

- cross-validation mean squared error
- test R-squared
- test MAE
- test RMSE
- notes about overfitting risk

## Current Status

No model performance values are documented here until the pipeline is run with actual data.
