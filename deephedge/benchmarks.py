"""Classical hedges used as benchmarks: the Black-Scholes delta and a no-trade band around it."""
import numpy as np
import torch

from . import blackscholes as bs
from .losses import avar, hedging_loss


def _by_path(x):
    """A per-path tensor of shape (m,) becomes (m, 1), to broadcast against the dates."""
    x = torch.as_tensor(x, dtype=torch.float64)
    return x[:, None] if x.ndim == 1 else x


def time_to_maturity(T, n_steps):
    """tau_j = T - t_j at the trading dates t_0, ..., t_{N-1}; T may differ by path."""
    T = _by_path(T)
    return T - torch.arange(n_steps, dtype=torch.float64) * (T / n_steps)


def delta_holdings(S, sigma, K=1.0, T=0.25, kind=1):
    """Black-Scholes delta at every trading date, shape (m, N).
    sigma may be a scalar, one value per path, or one per path and date; K, T and kind may vary by path."""
    tau = time_to_maturity(T, S.shape[1] - 1)
    return bs.delta(S[:, :-1], _by_path(K), tau, _by_path(sigma), _by_path(kind))


def band_holdings(S, eps, c, sigma, K=1.0, T=0.25, kind=1):
    """No-trade band around delta (Whalley & Wilmott, 1997).

    Buy delta at t_0. Afterwards keep the current holding while it stays within B of delta,
    otherwise trade only up to the nearest edge. Their asymptotic half-width is
    B = (3 eps S Gamma^2 / (2 lambda))^(1/3) for a seller with risk aversion lambda; the unknown
    constant is absorbed into the factor c, which is tuned on simulated paths.
    """
    tau = time_to_maturity(T, S.shape[1] - 1)
    K, sigma, c = _by_path(K), _by_path(sigma), _by_path(c)
    D = bs.delta(S[:, :-1], K, tau, sigma, _by_path(kind))
    B = c * (eps * S[:, :-1] * bs.gamma(S[:, :-1], K, tau, sigma) ** 2) ** (1 / 3)
    h = D[:, 0]
    H = [h]
    for j in range(1, D.shape[1]):
        h = torch.clamp(h, D[:, j] - B[:, j], D[:, j] + B[:, j])
        H.append(h)
    return torch.stack(H, dim=1)


C_GRID = [round(0.05 * k, 2) for k in range(13)]       # c = 0 is the plain delta hedge


def tune_band(S, eps, sigma, K=1.0, T=0.25, kind=1, alpha=0.9, c_grid=C_GRID):
    """Band factor c with the lowest AVaR_alpha on the given (training or validation) paths."""
    scores = [avar(hedging_loss(band_holdings(S, eps, c, sigma, K, T, kind), S, K, kind, eps)[0], alpha)
              for c in c_grid]
    return c_grid[int(np.argmin(scores))]
