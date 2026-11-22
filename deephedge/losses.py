"""Seller's hedging loss and the Average Value-at-Risk.

Sign convention: X is the loss of the seller who pays the option payoff at T and hedges with the stock.
The price charged for the option under a risk measure rho is rho(X), so lower is better.
"""
import torch


def payoff(S_T, K=1.0, kind=1):
    """European payoff, kind = +1 for a call and -1 for a put."""
    return torch.relu(kind * (S_T - K))


def trading_costs(H, S, eps):
    """Proportional costs eps * |trade| * price, paid on every trade: the initial purchase at t_0
    (from zero shares), each rebalancing, and the final liquidation at T (back to zero shares).

    H has shape (m, N): H[:, j] is the holding on [t_j, t_{j+1}). S has shape (m, N + 1).
    """
    H_full = torch.cat([torch.zeros_like(H[:, :1]), H, torch.zeros_like(H[:, :1])], dim=1)
    return eps * ((H_full[:, 1:] - H_full[:, :-1]).abs() * S).sum(dim=1)


def hedging_loss(H, S, K=1.0, kind=1, eps=0.0):
    """X = g(S_T) - sum_j H_j (S_{t_{j+1}} - S_{t_j}) + costs. Returns (X, costs)."""
    gains = (H * (S[:, 1:] - S[:, :-1])).sum(dim=1)
    costs = trading_costs(H, S, eps)
    return payoff(S[:, -1], K, kind) - gains + costs, costs


def avar(X, alpha):
    """Empirical AVaR_alpha: the average of the worst (1 - alpha) share of the losses."""
    k = int(round((1 - alpha) * len(X)))
    return X.sort().values[-k:].mean().item()


def avar_objective(X, w, alpha):
    """Rockafellar-Uryasev form w + E[(X - w)^+] / (1 - alpha), minimised jointly over w and the hedge.
    At the optimum, w is the alpha-quantile of X and the value is AVaR_alpha(X)."""
    return w + torch.relu(X - w).mean() / (1 - alpha)


def turnover(H):
    """Total number of shares traded along each path, including the first and the last trade."""
    H_full = torch.cat([torch.zeros_like(H[:, :1]), H, torch.zeros_like(H[:, :1])], dim=1)
    return (H_full[:, 1:] - H_full[:, :-1]).abs().sum(dim=1)
