"""Price paths under the two models of parts 1-3.

All paths live on the uniform grid t_j = j T / N, j = 0, ..., N, and have shape (n_paths, N + 1).
The interest rate is zero throughout. GBM paths are exact martingales; the Heston stock scheme is a
martingale only up to a small discretisation bias.
"""
import numpy as np
import torch
from scipy.stats import ncx2


def gbm_paths(n_paths, seed, sigma=0.35, T=0.25, n_steps=30, s0=1.0):
    """Exact Black-Scholes paths: S_{t+h} = S_t exp(-sigma^2 h / 2 + sigma sqrt(h) Z)."""
    gen = torch.Generator().manual_seed(seed)
    h = T / n_steps
    z = torch.randn(n_paths, n_steps, generator=gen, dtype=torch.float64)
    log_increments = -0.5 * sigma**2 * h + sigma * h**0.5 * z
    log_s = torch.cat([torch.zeros(n_paths, 1, dtype=torch.float64), log_increments.cumsum(dim=1)], dim=1)
    return s0 * torch.exp(log_s)


def heston_variance(n_paths, seed, v0=0.5, a=4.0, b=0.5, sigma_v=1.0, T=0.25, n_steps=30):
    """Exact CIR variance paths: V_{t+h} is c times a noncentral chi-square given V_t.

    dV = a (b - V) dt + sigma_v sqrt(V) dW. Returns a float64 array of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    h = T / n_steps
    c = sigma_v**2 / (4 * a) * (1 - np.exp(-a * h))
    dof = 4 * a * b / sigma_v**2
    V = np.empty((n_paths, n_steps + 1))
    V[:, 0] = v0
    for j in range(n_steps):
        noncentrality = np.exp(-a * h) * V[:, j] / c
        V[:, j + 1] = c * ncx2.rvs(dof, noncentrality, size=n_paths, random_state=rng)
    return V


def heston_stock(V, seed, rho=-0.3, a=4.0, b=0.5, sigma_v=1.0, T=0.25, s0=1.0):
    """Stock paths given the variance paths, with the simplified Broadie-Kaya scheme.

    The integrated variance over [t, t+h] is approximated by V_t h, and the stochastic integral of
    sqrt(V) against the variance Brownian motion is recovered from the variance increment.
    """
    rng = np.random.default_rng(seed)
    n_paths, n_steps = V.shape[0], V.shape[1] - 1
    h = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    S = np.empty_like(V)
    S[:, 0] = s0
    for j in range(n_steps):
        log_return = (rho / sigma_v * (V[:, j + 1] - V[:, j] - a * b * h)
                      + (a * rho / sigma_v - 0.5) * V[:, j] * h
                      + np.sqrt((1 - rho**2) * V[:, j] * h) * z[:, j])
        S[:, j + 1] = S[:, j] * np.exp(log_return)
    return S


def contract_paths(contracts, paths_per_contract, seed, sigma=0.35, n_steps=30, s0=1.0):
    """Black-Scholes paths for several contracts, each with its own maturity and the same number of
    trading dates. A contract is (K, T, kind) or (K, T, kind, sigma).

    Returns S of shape (m, n_steps + 1) and a dict of per-path tensors K, T, kind, sigma and group,
    in random order.
    """
    gen = torch.Generator().manual_seed(seed)
    groups = torch.arange(len(contracts)).repeat_interleave(paths_per_contract)
    terms = torch.tensor([tuple(c) + ((sigma,) if len(c) == 3 else ()) for c in contracts], dtype=torch.float64)[groups]
    h = (terms[:, 1] / n_steps)[:, None]
    vol = terms[:, 3:4]
    z = torch.randn(len(groups), n_steps, generator=gen, dtype=torch.float64)
    log_s = torch.cat([torch.zeros(len(groups), 1, dtype=torch.float64),
                       (-0.5 * vol**2 * h + vol * h.sqrt() * z).cumsum(dim=1)], dim=1)
    order = torch.randperm(len(groups), generator=gen)
    info = dict(K=terms[:, 0], T=terms[:, 1], kind=terms[:, 2], sigma=terms[:, 3], group=groups)
    return s0 * torch.exp(log_s)[order], {k: v[order] for k, v in info.items()}

