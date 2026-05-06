# 01 — Linear Regression on California Housing

**Notebook:** [`linear_regression_california.ipynb`](linear_regression_california.ipynb)

A regression demo that fits the same model two different ways (closed-form normal
equation and gradient descent) and uses the agreement between them as a correctness
check. The analysis emphasises *what the residuals tell us* — capping bias from a
clipped target and heteroscedasticity at the top of the price range — rather than the
headline R² number.

## Why this dataset?

California Housing is large enough (20 k rows) to make gradient descent worth running,
small enough to be fast, has a clean numeric feature set with no missing values, and
has a well-known structural quirk (the target is right-censored at $500 k) that makes
the discussion of model failure modes concrete instead of hypothetical.

## Key takeaways

- Closed-form and gradient-descent fits agree to ~3 decimal places — implementation OK.
- Test R² ≈ 0.60, RMSE ≈ 0.74 ($74 k) — consistent with the canonical sklearn benchmark.
- Residuals at the top of the price range are systematically positive (target capping).
- Residual variance grows with predicted value (heteroscedasticity).
