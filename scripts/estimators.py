"""
estimators.py: panel OLS and IV with county fixed effects, written to match Stata.

Why our own code instead of linearmodels?
The paper uses two different variance estimators that we need to reproduce exactly:
- Tables 2-4: `xtreg ..., fe cl(state_n)`: standard errors clustered by state.
- Table 5:    `xtivreg2 ..., fe r bw(3)`: HAC standard errors with a Bartlett kernel
               (bandwidth 3) within each county (heteroskedasticity- and autocorrelation-robust).
linearmodels has no panel-aware Bartlett kernel, so we compute the sandwich ourselves.
Both functions only need numpy and pandas, so every line can be checked.

Fixed effects are removed with the within transformation (demeaning by county).
The weights w1 are constant within a county, so weighted and unweighted county means
are identical. Year effects enter as dummies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from config import PANEL_ID, STATE_ID, TIME_ID, WEIGHT


@dataclass
class Result:
    coef: pd.Series
    se: pd.Series
    nobs: int
    n_counties: int
    n_states: int
    r2_within: float | None = None
    first_stage_F: float | None = None
    extra: dict = field(default_factory=dict)
    df_t: int | None = None   # clustered SEs: Stata uses a t distribution with (clusters - 1) df

    def stars(self, var: str) -> str:
        t = abs(self.coef[var] / self.se[var])
        crit = (lambda q: stats.t.ppf(q, self.df_t)) if self.df_t else stats.norm.ppf
        return "***" if t > crit(0.995) else "**" if t > crit(0.975) else "*" if t > crit(0.95) else ""


def _prepare(df, cols, drop_singletons):
    d = df.dropna(subset=cols + [WEIGHT]).copy()
    if drop_singletons:  # xtivreg2 drops counties with a single observation, xtreg keeps them
        d = d[d.groupby(PANEL_ID)[PANEL_ID].transform("size") > 1]
    d = d.reset_index(drop=True)
    years = pd.get_dummies(d[TIME_ID], prefix="yr", drop_first=True, dtype=float)
    return pd.concat([d, years], axis=1), list(years.columns)


def _within(d, cols):
    return d[cols] - d.groupby(PANEL_ID)[cols].transform("mean")


def _cluster_meat(scores, groups):
    g = pd.DataFrame(scores).groupby(np.asarray(groups)).sum().values
    return g.T @ g


def _bartlett_meat(scores, panel, time, bw=3):
    """Sum over counties of sum_{s,t} k(|s-t|) g_s g_t', k(j) = 1 - j/bw for j < bw."""
    k = scores.shape[1]
    meat = np.zeros((k, k))
    order = np.lexsort((time, panel))
    s, p, t = scores[order], panel[order], time[order]
    starts = np.flatnonzero(np.r_[True, p[1:] != p[:-1]])
    ends = np.r_[starts[1:], len(p)]
    for a, b in zip(starts, ends):
        G, tt = s[a:b], t[a:b]
        lag = np.abs(tt[:, None] - tt[None, :])
        K = np.where(lag < bw, 1 - lag / bw, 0.0)
        meat += G.T @ K @ G
    return meat


def fe_ols(df: pd.DataFrame, y: str, x: list[str]) -> Result:
    """Stata: xtreg y x yr* [aw=w1], fe cl(state_n)."""
    d, yrs = _prepare(df, [y] + x, drop_singletons=False)
    regs = x + yrs
    t = _within(d, [y] + regs)
    sw = np.sqrt(d[WEIGHT].values)
    X, Y = t[regs].values * sw[:, None], t[y].values * sw
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    e = Y - X @ b
    n, G = len(d), d[STATE_ID].nunique()
    K = np.linalg.matrix_rank(X) + 1                       # + constant, as reported by Stata
    q = G / (G - 1) * (n - 1) / (n - K)                    # Stata's small-sample factor for cluster VCE
    V = q * XtX_inv @ _cluster_meat(X * e[:, None], d[STATE_ID]) @ XtX_inv
    r2 = 1 - (e @ e) / (Y @ Y)
    return Result(pd.Series(b[: len(x)], x), pd.Series(np.sqrt(np.diag(V))[: len(x)], x),
                  n, d[PANEL_ID].nunique(), G, r2_within=r2, df_t=G - 1)


def fe_iv(df: pd.DataFrame, y: str, endog: str, instrument: str, exog: list[str], bw: int = 3,
          cluster: str | None = None) -> Result:
    """Stata: xtivreg2 y (endog = instrument) exog yr* [aw=w1], fe r bw(3) partial(yr*).
    cluster="state_n" instead: SEs clustered by state (xtivreg2 ..., fe cluster(state_n)), not in the paper."""
    d, yrs = _prepare(df, [y, endog, instrument] + exog, drop_singletons=True)
    w_ = exog + yrs
    t = _within(d, [y, endog, instrument] + w_)
    sw = np.sqrt(d[WEIGHT].values)
    Y = t[y].values * sw
    X = t[[endog] + w_].values * sw[:, None]
    Z = t[[instrument] + w_].values * sw[:, None]
    n, n_c = len(d), d[PANEL_ID].nunique()
    dof = n / (n - n_c)                    # xtivreg2 counts the absorbed county effects
    panel, time = d[PANEL_ID].values, d[TIME_ID].values

    # 2SLS
    Xhat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    A = np.linalg.pinv(Xhat.T @ Xhat)
    b = A @ Xhat.T @ Y
    e = Y - X @ b
    if cluster:
        G = d[cluster].nunique()
        V = G / (G - 1) * A @ _cluster_meat(Xhat * e[:, None], d[cluster]) @ A
    else:
        V = dof * A @ _bartlett_meat(Xhat * e[:, None], panel, time, bw) @ A

    # First stage: robust (HAC) Wald test of the excluded instrument
    xe = X[:, 0]
    Az = np.linalg.pinv(Z.T @ Z)
    g = Az @ Z.T @ xe
    u = xe - Z @ g
    if cluster:
        Vz = G / (G - 1) * Az @ _cluster_meat(Z * u[:, None], d[cluster]) @ Az
    else:
        Vz = dof * Az @ _bartlett_meat(Z * u[:, None], panel, time, bw) @ Az
    F = g[0] ** 2 / Vz[0, 0]

    names = [endog] + exog
    k = len(names)
    return Result(pd.Series(b[:k], names), pd.Series(np.sqrt(np.diag(V))[:k], names),
                  n, n_c, d[STATE_ID].nunique(), first_stage_F=F, df_t=(G - 1) if cluster else None,
                  extra={"first_stage_coef": g[0], "first_stage_se": np.sqrt(Vz[0, 0])})
