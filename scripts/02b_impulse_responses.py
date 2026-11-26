"""
02b_impulse_responses.py: Figures 1 and 3 of Favara & Imbs (2015), local projections (Jorda 2005).

Idea: the regressions of Tables 2 and 5 measure the effect in the same year. To see how the
effect evolves, the outcome is shifted h = 0, 1, ..., 8 years into the future and the same
regression is run once per horizon. Each coefficient becomes one point of the impulse response.

Figure 1 (credit), original_code/dofiles/irf_fig_1-3.do, first block:
    xtreg F{h}.Dl_L  L.Dl_L $D_control yr* Linter_bra, fe cluster(state_n)
Figure 3 (house prices), second block:
    xtivreg2 F{h}.Dl_hpi  L.Dl_hpi $D_control_hp yr* (Dl_L = Linter_bra) [aw=w1], fe r partial(yr*)
    note: "r" without bw() = heteroskedasticity-robust SEs only (no autocorrelation correction),
          unlike Table 5. Figure 1 is unweighted, Figure 3 uses w1, exactly as in the Stata code.
Bands: 90 percent (coefficient +/- 1.64 standard errors), as in the paper.

Outputs
    paper/figures/figure3_irf_house_prices.pdf   (main figure of the report)
    paper/figures/figure1_irf_credit.pdf
    paper/tables/irf_estimates.csv               (all coefficients, for checking and the text)
    dashboard/data/irf_house_prices.csv          (for the website chart)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DASHBOARD_DATA, FIGURES, INSTRUMENT, PANEL_FILE, PANEL_ID, TABLES, TIME_ID, WEIGHT
from estimators import fe_iv, fe_ols

HORIZONS = range(0, 9)
Z90 = 1.64
CREDIT = {"Dl_nloans_b": "Number of loans", "Dl_vloans_b": "Loan volume", "Dl_lir_b": "Loan to income ratio"}
D_CONTROL = ["Dl_inc", "LDl_inc", "Dl_pop", "LDl_pop", "Dl_hpi", "LDl_hpi", "Dl_her_v", "LDl_her_v"]
D_CONTROL_HP = ["Dl_inc", "LDl_inc", "Dl_pop", "LDl_pop", "Dl_her_v", "LDl_her_v"]


def lead(df: pd.DataFrame, var: str, h: int) -> pd.Series:
    """Stata's F{h}.var with tsset county year: value of var in year t+h of the same county (missing if absent)."""
    if h == 0:
        return df[var]
    future = df[[PANEL_ID, TIME_ID, var]].assign(**{TIME_ID: df[TIME_ID] - h})
    merged = df[[PANEL_ID, TIME_ID]].merge(future, on=[PANEL_ID, TIME_ID], how="left")
    return merged[var].values


def figure1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df.copy()
    d["_unit_w"] = 1.0  # Figure 1 is estimated without weights
    for var, label in CREDIT.items():
        for h in HORIZONS:
            d["_y"] = lead(d, var, h)
            r = fe_ols(d.assign(**{WEIGHT: d["_unit_w"]}), "_y", [INSTRUMENT, "L" + var] + D_CONTROL)
            rows.append({"figure": 1, "series": label, "horizon": h, "coef": r.coef[INSTRUMENT],
                         "se": r.se[INSTRUMENT], "nobs": r.nobs})
    return pd.DataFrame(rows)


def figure3(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df.copy()
    for var, label in CREDIT.items():
        for h in HORIZONS:
            d["_y"] = lead(d, "Dl_hpi", h)
            r = fe_iv(d, "_y", var, INSTRUMENT, ["LDl_hpi"] + D_CONTROL_HP, bw=1)  # bw=1: robust, no HAC
            rows.append({"figure": 3, "series": label, "horizon": h, "coef": r.coef[var],
                         "se": r.se[var], "nobs": r.nobs, "first_stage_F": r.first_stage_F})
    return pd.DataFrame(rows)


def plot(est: pd.DataFrame, title: str, ylabel: str, path) -> None:
    """Three panels side by side (fits the full text width of the report)."""
    series = list(CREDIT.values())
    fig, axes = plt.subplots(1, len(series), figsize=(10, 3.0), sharey=True)
    for ax, (panel, s) in zip(axes, zip("ABC", series)):
        e = est[est.series == s].sort_values("horizon")
        lo, hi = e.coef - Z90 * e.se, e.coef + Z90 * e.se
        ax.fill_between(e.horizon, lo, hi, color="0.85", lw=0)
        ax.plot(e.horizon, e.coef, color="black", lw=1.6, marker="o", ms=3)
        ax.plot(e.horizon, lo, color="0.4", lw=0.8, ls="--")
        ax.plot(e.horizon, hi, color="0.4", lw=0.8, ls="--")
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_title(f"Panel {panel}. {s}", fontsize=10, loc="left")
        ax.set_xlabel("Years after the shock", fontsize=9)
        ax.set_xticks(list(HORIZONS))
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel, fontsize=9)
    # title and notes are in the LaTeX caption; `title` is kept for the function signature
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(PANEL_FILE)
    f3 = figure3(df)
    f1 = figure1(df)
    est = pd.concat([f3, f1], ignore_index=True)
    est.to_csv(TABLES / "irf_estimates.csv", index=False)
    f3.to_csv(DASHBOARD_DATA / "irf_house_prices.csv", index=False)

    plot(f3, "House prices: response to an instrumented credit shock",
         "Elasticity of house price growth", FIGURES / "figure3_irf_house_prices.pdf")
    plot(f1, "Mortgage credit: response to a branching deregulation shock",
         "Effect on credit growth", FIGURES / "figure1_irf_credit.pdf")

    with pd.option_context("display.precision", 3, "display.width", 160):
        print(f3.pivot(index="horizon", columns="series", values="coef").round(3).to_string())
    print(f"\nWrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
