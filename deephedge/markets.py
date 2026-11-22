"""Real index data and the two historical simulators of part 4.

A test window is a 3-month option (63 trading days) observed at the same 31 dates as the simulated
paths, rescaled to start at 1. Everything used to build a hedge comes from 1990-2009; the windows
starting in 2010-2026 are kept for testing.
"""
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

DAYS = 63                                                    # trading days in 3 months
N = 30
OFFSETS = np.round(np.arange(N + 1) * DAYS / N).astype(int)  # trading-day index of t_0, ..., t_N
END = "2026-11-19"                                           # data used: strictly before this date
TEST_START = "2010-01-01"
LOOKBACK = 20                                                # days of history for the volatility signal


def load_index(name, data_dir="../data"):
    """Daily closes and log returns of one index, cut at END so that results do not depend on the
    download date."""
    px = pd.read_csv(f"{data_dir}/{name}_daily.csv", parse_dates=["Date"], index_col="Date")["Close"]
    px = px[px.index < END]
    return px, np.log(px).diff().dropna()


def test_windows(px):
    """Every 63-day window starting in the test period, one per trading day. Shape (n, N + 1)."""
    p = px[TEST_START:].to_numpy()
    return torch.tensor(np.stack([p[s + OFFSETS] / p[s] for s in range(len(p) - DAYS)]))


def window_dates(px, n):
    return px[TEST_START:].index[:n]


def calibration_vol(returns):
    """Annualised volatility of the 1990-2009 daily log returns."""
    return returns[:"2009-12-31"].std() * np.sqrt(252)


# ---------- simulator 1: block bootstrap of historical returns ----------

def block_bootstrap_paths(returns, n_paths, seed, block=10):
    """Paths glued together from random blocks of `block` consecutive real daily returns.

    Keeps the fat tails and the short-range volatility clustering of the data. Returns are
    de-meaned, so no strategy can profit from the market's average rise.
    """
    r = returns.to_numpy() - returns.mean()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(r) - block, size=(n_paths, int(np.ceil(DAYS / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :DAYS]
    log_s = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(r[idx], axis=1)], axis=1)
    return torch.tensor(np.exp(log_s[:, OFFSETS]))


def block_bootstrap_returns(returns, n_paths, seed, block=10):
    """Same simulator, returning LOOKBACK + DAYS daily returns per path (for the volatility signal)."""
    r = returns.to_numpy() - returns.mean()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(r) - block, size=(n_paths, int(np.ceil((LOOKBACK + DAYS) / block))))
    return r[(starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :LOOKBACK + DAYS]]


# ---------- simulator 2: filtered historical simulation with a GJR-GARCH(1,1) ----------

def gjr_filter(r, params):
    """Conditional variances h_t = w + (a + g 1{r_{t-1} < 0}) r_{t-1}^2 + b h_{t-1}.
    h[t] is the variance of r[t] given r[:t]; h[-1] is the next one."""
    w, a, g, b = params
    h = np.empty(len(r) + 1)
    h[0] = r.var()
    for t in range(len(r)):
        h[t + 1] = w + (a + g * (r[t] < 0)) * r[t] ** 2 + b * h[t]
    return h


def fit_gjr(r):
    """Gaussian quasi-maximum likelihood, with the unconditional variance fixed to the sample one."""
    var = r.var()

    def nll(x):
        a, g, b = x
        if a < 0 or g < 0 or b < 0 or a + g / 2 + b >= 0.999:
            return 1e10
        h = gjr_filter(r, (var * (1 - a - g / 2 - b), a, g, b))[:-1]
        return 0.5 * np.sum(np.log(h) + r**2 / h)

    a, g, b = minimize(nll, x0=[0.03, 0.08, 0.9], method="Nelder-Mead",
                       options=dict(xatol=1e-6, fatol=1e-6)).x
    return var * (1 - a - g / 2 - b), a, g, b


def fhs_paths(returns, n_paths, seed, params):
    """Filtered historical simulation (Barone-Adesi et al., 1999).

    The GJR-GARCH carries the persistence of volatility and the leverage effect; its innovations are
    resampled from the real standardised residuals, which keeps the fat tails. Each path starts in a
    randomly drawn historical volatility regime and has LOOKBACK days of history before t_0.
    Returns the prices at the 31 dates (S_{t_0} = 1) and the daily returns.
    """
    r = returns.to_numpy() - returns.mean()
    h_hist = gjr_filter(r, params)
    z = r / np.sqrt(h_hist[:-1])
    z = (z - z.mean()) / z.std()
    z = np.concatenate([z, -z])        # symmetrised: E[z^2 1{z < 0}] = 1/2 exactly, as assumed in the fit
    w, a, g, b = params
    rng = np.random.default_rng(seed)
    h = h_hist[rng.integers(0, len(r), size=n_paths)]
    R = np.empty((n_paths, LOOKBACK + DAYS))
    for t in range(LOOKBACK + DAYS):
        R[:, t] = np.sqrt(h) * z[rng.integers(0, len(z), size=n_paths)]
        h = w + (a + g * (R[:, t] < 0)) * R[:, t] ** 2 + b * h
    log_s = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(R[:, LOOKBACK:], axis=1)], axis=1)
    return torch.tensor(np.exp(log_s[:, OFFSETS])), R


# ---------- volatility signal ----------

def vol_signal(R):
    """Annualised realised volatility of the LOOKBACK daily returns before each trading date.
    R holds LOOKBACK days of history followed by the window. Shape (n, N)."""
    return torch.tensor(np.stack([np.sqrt(252 * np.mean(R[:, o:LOOKBACK + o] ** 2, axis=1))
                                  for o in OFFSETS[:-1]], axis=1))


def history_windows(returns, S):
    """Daily returns of each real test window with its LOOKBACK days of history before t_0."""
    r = returns.to_numpy()
    first = int((returns.index < TEST_START).sum())          # the return that ends on the first test day
    R = np.stack([r[first + s + 1 - LOOKBACK:first + s + 1 + DAYS] for s in range(len(S))])
    assert np.allclose(np.exp(np.cumsum(R[:, LOOKBACK:], axis=1))[:, OFFSETS[1:] - 1], S[:, 1:].numpy())
    return R


def vol_persistence(R):
    """Correlation between the log volatility of the LOOKBACK days before t_0 and inside the window."""
    pre = np.sqrt(252 * np.mean(R[:, :LOOKBACK] ** 2, axis=1))
    inside = np.sqrt(252 * np.mean(R[:, LOOKBACK:] ** 2, axis=1))
    return np.corrcoef(np.log(pre), np.log(inside))[0, 1]
