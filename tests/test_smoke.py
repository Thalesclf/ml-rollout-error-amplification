import numpy as np
from experiments_campaign_spectral import lorenz96_rhs, rk4_step, simulate_l96

def test_lorenz96_rhs():
    x = np.ones(20) * 8.0
    y = lorenz96_rhs(x, 8.0)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))

def test_rk4_step():
    x = np.ones(20) * 8.0
    y = rk4_step(x, 0.01, 8.0)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))

def test_fixed_seed_reproducibility():
    a = simulate_l96(n=5, steps=5, burnin=2, seed=7)
    b = simulate_l96(n=5, steps=5, burnin=2, seed=7)
    np.testing.assert_allclose(a, b)
