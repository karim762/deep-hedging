import numpy as np
import torch

from deephedge import benchmarks as bm
from deephedge import blackscholes as bs
from deephedge import simulate
from deephedge.losses import avar, avar_objective, grouped_avar_objective, hedging_loss, trading_costs
from deephedge.models import PositionHedge, StateHedge


def test_avar_on_ten_losses():
    # AVaR_0.5 averages the worst 5 losses, AVaR_0.9 is the single worst one
    X = torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 14.])
    assert avar(X, 0.5) == 8.0
    assert avar(X, 0.9) == 14.0
    # the Rockafellar-Uryasev objective reaches the same values at w = the alpha-quantile
    assert abs(avar_objective(X, 4.0, 0.5).item() - 8.0) < 1e-5
    assert abs(avar_objective(X, 8.0, 0.9).item() - 14.0) < 1e-5


def test_black_scholes_at_the_money():
    assert abs(bs.price(1.0, 1.0, 0.25, 0.35).item() - 0.069726) < 1e-6
    assert abs(bs.delta(1.0, 1.0, 0.25, 0.35).item() - 0.5349) < 1e-4


def test_delta_is_the_derivative_of_the_price():
    s = torch.linspace(0.6, 1.5, 50, dtype=torch.float64)
    h = 1e-5
    for kind in (1, -1):
        fd = (bs.price(s + h, 1.0, 0.1, 0.35, kind) - bs.price(s - h, 1.0, 0.1, 0.35, kind)) / (2 * h)
        assert torch.allclose(fd, bs.delta(s, 1.0, 0.1, 0.35, kind), atol=1e-7)


def test_put_call_parity():
    s = torch.linspace(0.6, 1.5, 50, dtype=torch.float64)
    call, put = bs.price(s, 1.05, 0.2, 0.3, 1), bs.price(s, 1.05, 0.2, 0.3, -1)
    assert torch.allclose(call - put, s - 1.05)


def test_gbm_is_a_martingale_with_the_right_volatility():
    S = simulate.gbm_paths(200_000, seed=0)
    S_T = S[:, -1]
    assert abs(S_T.mean().item() - 1.0) < 4 * S_T.std().item() / len(S_T) ** 0.5
    assert abs(S_T.log().std().item() - 0.35 * 0.25**0.5) < 2e-3


def test_heston_variance_matches_its_first_two_moments():
    a, b, v0, sigma_v, T = 4.0, 0.5, 0.5, 1.0, 0.25
    V_T = simulate.heston_variance(200_000, seed=0)[:, -1]
    mean = b + (v0 - b) * np.exp(-a * T)
    var = (v0 * sigma_v**2 / a * np.exp(-a * T) * (1 - np.exp(-a * T))
           + b * sigma_v**2 / (2 * a) * (1 - np.exp(-a * T)) ** 2)
    assert abs(V_T.mean() - mean) < 4 * var**0.5 / len(V_T) ** 0.5
    assert abs(V_T.std() / var**0.5 - 1) < 0.01


def test_costs_include_first_and_last_trade():
    S = torch.ones(1, 4, dtype=torch.float64)
    H = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
    assert trading_costs(H, S, 0.01).item() == 0.01 * (0.5 + 0.5)       # buy 0.5, then sell 0.5 at T


def test_band_with_zero_width_is_the_delta_hedge():
    S = simulate.gbm_paths(1000, seed=3)
    assert torch.allclose(bm.band_holdings(S, 0.01, 0.0, 0.35), bm.delta_holdings(S, 0.35))


def test_wider_band_trades_less():
    S = simulate.gbm_paths(5000, seed=4)
    _, c_narrow = hedging_loss(bm.band_holdings(S, 0.01, 0.1, 0.35), S, eps=0.01)
    _, c_wide = hedging_loss(bm.band_holdings(S, 0.01, 0.4, 0.35), S, eps=0.01)
    assert c_wide.mean() < c_narrow.mean()


def test_policies_only_use_past_information():
    S = simulate.gbm_paths(64, seed=5).float()
    torch.manual_seed(0)
    for model, run in [(StateHedge(S[:, :-1, None]), lambda m, s: m(s[:, :-1, None])),
                       (PositionHedge(S), lambda m, s: m(s))]:
        S_future = S.clone()
        S_future[:, 16:] *= 1.3                          # change the prices from t_16 on
        with torch.no_grad():
            H, H_future = run(model, S), run(model, S_future)
        assert torch.equal(H[:, :16], H_future[:, :16])   # decisions up to t_15 are unchanged
        assert not torch.equal(H[:, 16:], H_future[:, 16:])


def test_grouped_objective_with_one_group_is_the_plain_objective():
    X = torch.randn(1000)
    w = torch.tensor([0.3])
    groups = torch.zeros(1000, dtype=torch.long)
    assert torch.allclose(grouped_avar_objective(X, w, groups, 0.9), avar_objective(X, w[0], 0.9))


def test_contract_paths_follow_their_own_maturity_and_volatility():
    S, info = simulate.contract_paths([(1.0, 0.125, 1, 0.25), (1.0, 0.5, -1, 0.7)], 50_000, seed=0)
    for g, (T, sigma) in enumerate([(0.125, 0.25), (0.5, 0.7)]):
        log_ST = S[info["group"] == g, -1].log()
        assert abs(log_ST.std().item() / (sigma * T**0.5) - 1) < 0.02


def test_benchmarks_accept_one_contract_per_path():
    S, info = simulate.contract_paths([(0.95, 0.25, -1)], 2000, seed=1)
    H_scalar = bm.band_holdings(S, 0.01, 0.15, 0.35, K=0.95, T=0.25, kind=-1)
    H_paths = bm.band_holdings(S, 0.01, torch.full((2000,), 0.15), 0.35, info["K"], info["T"], info["kind"])
    assert torch.allclose(H_scalar, H_paths)

