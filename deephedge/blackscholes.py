"""Black-Scholes formulas with zero interest rate, written in terms of the time to maturity tau.

`kind` is +1 for a call and -1 for a put. All functions broadcast over tensors.
"""
import torch

_normal = torch.distributions.Normal(0.0, 1.0)


def _as_tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


def d_plus(s, K, tau, sigma):
    s, K, tau, sigma = map(_as_tensor, (s, K, tau, sigma))
    return (torch.log(s / K) + 0.5 * sigma**2 * tau) / (sigma * torch.sqrt(tau))


def price(s, K, tau, sigma, kind=1):
    s, K, sigma = map(_as_tensor, (s, K, sigma))
    dp = d_plus(s, K, tau, sigma)
    dm = dp - sigma * torch.sqrt(_as_tensor(tau))
    call = s * _normal.cdf(dp) - K * _normal.cdf(dm)
    return call if kind == 1 else call - s + K          # put-call parity with r = 0


def delta(s, K, tau, sigma, kind=1):
    """dC/ds = Phi(d+) for a call, Phi(d+) - 1 for a put."""
    kind = _as_tensor(kind)
    return _normal.cdf(d_plus(s, K, tau, sigma)) - (1 - kind) / 2


def gamma(s, K, tau, sigma):
    """Same for calls and puts."""
    s, tau, sigma = map(_as_tensor, (s, tau, sigma))
    return torch.exp(_normal.log_prob(d_plus(s, K, tau, sigma))) / (s * sigma * torch.sqrt(tau))
