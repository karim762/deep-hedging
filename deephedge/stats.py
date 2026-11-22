"""Uncertainty of an AVaR comparison."""
import numpy as np


def _tail_mean(x, k):
    return np.sort(x)[-k:].mean()


def avar_ratio_interval(X_ref, X_new, alpha=0.9, n_rep=2000, block=63, seed=0):
    """Moving-block bootstrap over time of AVaR_alpha(X_new) / AVaR_alpha(X_ref) - 1.

    Real test windows start on consecutive days and overlap, so they are resampled in blocks of
    `block` consecutive windows (one quarter). Both strategies are resampled with the same indices.
    Returns the 95% interval and the share of resamples where X_new has the lower AVaR.
    """
    X_ref, X_new = np.asarray(X_ref), np.asarray(X_new)
    rng = np.random.default_rng(seed)
    n = len(X_ref)
    k = int(round((1 - alpha) * n))
    reps = np.empty(n_rep)
    for i in range(n_rep):
        starts = rng.integers(0, n - block, size=int(np.ceil(n / block)))
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        reps[i] = _tail_mean(X_new[idx], k) / _tail_mean(X_ref[idx], k) - 1
    return np.percentile(reps, [2.5, 97.5]), np.mean(reps < 0)

