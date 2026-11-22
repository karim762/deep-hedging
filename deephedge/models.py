"""Deep hedging policies: one small network per trading date, all trained jointly.

Network j decides the holding on [t_j, t_{j+1}) from information available at t_j only.
Inputs are standardised with fixed training-set statistics, computed date by date.
"""
import torch
import torch.nn as nn


class DateNet(nn.Module):
    """Two hidden layers of 32 ReLU units, one output."""

    def __init__(self, n_inputs, n_hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_inputs, n_hidden), nn.ReLU(),
                                 nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                                 nn.Linear(n_hidden, 1))

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def _date_stats(X):
    """Mean and standard deviation of X[:, j] for every date j. X has shape (m, N, d)."""
    loc, scale = [], []
    for j in range(X.shape[1]):
        m, s = X[:, j].mean(0), X[:, j].std(0)
        loc.append(m)
        scale.append(torch.where(s > 1e-8, s, torch.ones_like(s)))     # S_{t_0} = s0 on every path
    return torch.stack(loc), torch.stack(scale)


class StateHedge(nn.Module):
    """Frictionless hedge (parts 1-2): H_j = f_j(X_{t_j}), where X is the observed state.

    Without transaction costs the best holding does not depend on the position already held,
    so each network sees the current state only. n_thresholds > 0 adds trainable AVaR thresholds w.
    """

    def __init__(self, X_train, n_hidden=32, n_thresholds=0):
        super().__init__()
        n_steps, n_inputs = X_train.shape[1], X_train.shape[2]
        self.networks = nn.ModuleList([DateNet(n_inputs, n_hidden) for _ in range(n_steps)])
        self.w = nn.Parameter(torch.zeros(n_thresholds)) if n_thresholds else None
        loc, scale = _date_stats(X_train)
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def decision(self, j, x):
        """Holding chosen at t_j for states x of shape (m, d)."""
        return self.networks[j]((x - self.loc[j]) / self.scale[j])

    def forward(self, X):
        """X: (m, N, d) -> holdings (m, N)."""
        return torch.stack([self.decision(j, X[:, j]) for j in range(len(self.networks))], dim=1)


class PositionHedge(nn.Module):
    """Hedge under transaction costs (parts 3-4): H_j = H_{j-1} + f_j(S_{t_j}, H_{j-1}, Z_{t_j}).

    With costs, the best trade depends on the position inherited from t_{j-1}: when it is close
    to the target, adjusting it costs more than it protects. Each network therefore receives the
    current price and the current holding (plus optional extra features Z, such as a volatility
    signal) and outputs the trade; doing nothing is an output of zero.
    """

    def __init__(self, S_train, Z_train=None, n_hidden=32, n_thresholds=1):
        super().__init__()
        n_steps = S_train.shape[1] - 1
        n_extra = 0 if Z_train is None else Z_train.shape[2]
        self.networks = nn.ModuleList([DateNet(2 + n_extra, n_hidden) for _ in range(n_steps)])
        self.w = nn.Parameter(torch.zeros(n_thresholds))
        loc, scale = _date_stats(S_train[:, :-1, None])
        self.register_buffer("s_loc", loc[:, 0])
        self.register_buffer("s_scale", scale[:, 0])
        if n_extra:
            loc, scale = _date_stats(Z_train)
            self.register_buffer("z_loc", loc)
            self.register_buffer("z_scale", scale)

    def forward(self, S, Z=None):
        """S: (m, N + 1), Z: (m, N, k) or None -> holdings (m, N)."""
        h = torch.zeros(S.shape[0], dtype=S.dtype)            # no shares before t_0
        H = []
        for j, net in enumerate(self.networks):
            x = [(S[:, j] - self.s_loc[j]) / self.s_scale[j], h]
            if Z is not None:
                x += list(((Z[:, j] - self.z_loc[j]) / self.z_scale[j]).unbind(-1))
            h = h + net(torch.stack(x, dim=-1))
            H.append(h)
        return torch.stack(H, dim=1)
