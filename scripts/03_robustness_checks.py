"""
03_robustness_checks.py: critical evaluation of Favara & Imbs (2015).

Each check targets one threat to the IV estimate in Table 5 (equations 3 and 4 of the paper):

A. Placebo (Table 2): deregulation should raise lending by commercial banks (covered by branching
   rules) but not by independent mortgage companies, thrifts and credit unions (not covered).
   If credit DEMAND drove the result, both groups would lend more. -> replication of Table 2.
B. Anticipation / reverse timing: does FUTURE deregulation "predict" today's credit or house price
   growth? A significant lead suggests states deregulated in response to local booms
   (violates exogeneity of the instrument).
C. Nickell bias: regressions with county FE AND a lagged dependent variable are biased when T is
   short (here up to 12 years). Check: drop the lagged dependent variable. (The authors' own answer,
   GMM estimators, is in literature/Favara_Imbs_2015_Addendum.pdf.)
J. Clustering: Table 5 SEs are county HAC; the instrument varies by state -> cluster by state.
I. Weights: Table 5 weights states equally (w1); what if every county counts equally?
H. OLS vs. IV: how far does instrumenting move the estimate (simultaneity, measurement error, LATE)?
D. Control sensitivity (OVB intuition): how much does the IV estimate move when controls are
   dropped? Large movements = the estimate leans on the controls; small = less worry about
   omitted variables correlated with the instrument.
E. Weak instruments: first-stage F is about 10 (at the Stock-Yogo rule of thumb). Anderson-Rubin
   confidence intervals stay valid with weak instruments.
F. Mechanism: the effect on prices should be larger where housing supply is inelastic (Saiz 2010).
   2SLS separately for counties below/above the median elasticity.
G. Extension: same IV with the public FHFA county house price index instead of Moody's.
   Runs only once data/raw/hpi_at_bdl_county.xlsx exists (download link in section G).

Outputs: paper/tables/robustness.tex, paper/tables/placebo_table2.tex,
         paper/tables/robustness_all.csv (all numbers), paper/tables/extension.tex (once G runs)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DASHBOARD_DATA, INSTRUMENT, PANEL_FILE, PANEL_ID, RAW, TABLES, TIME_ID, WEIGHT
from estimators import _bartlett_meat, _cluster_meat, _prepare, _within, fe_iv, fe_ols

D_CONTROL = ["Dl_inc", "LDl_inc", "Dl_pop", "LDl_pop", "Dl_hpi", "LDl_hpi", "Dl_her_v", "LDl_her_v"]
D_CONTROL_HP = ["Dl_inc", "LDl_inc", "Dl_pop", "LDl_pop", "Dl_her_v", "LDl_her_v"]
CREDIT = {"Dl_nloans_b": "Number of loans", "Dl_vloans_b": "Loan volume", "Dl_lir_b": "Loan to income"}
RESULTS: list[dict] = []


def record(check, spec, var, res, extra=None):
    row = {"check": check, "spec": spec, "variable": var, "coef": res.coef[var], "se": res.se[var],
           "stars": res.stars(var), "nobs": res.nobs, "counties": res.n_counties,
           "first_stage_F": res.first_stage_F}
    row.update(extra or {})
    RESULTS.append(row)
    return row


def unweighted(df):
    return df.assign(**{WEIGHT: 1.0})


def add_lead(df, var, h, name):
    future = df[[PANEL_ID, TIME_ID, var]].assign(**{TIME_ID: df[TIME_ID] - h}).rename(columns={var: name})
    return df.merge(future, on=[PANEL_ID, TIME_ID], how="left")


# --- A. Placebo: Table 2 ---------------------------------------------------------------------
PAPER_T2 = {"Dl_nloans_b": (0.028, 0.010), "Dl_vloans_b": (0.029, 0.012), "Dl_lir_b": (0.029, 0.011),
            "Dl_nloans_pl": (0.002, 0.008), "Dl_vloans_pl": (0.001, 0.008), "Dl_lir_pl": (0.001, 0.008)}


def placebo_table2(df):
    d = unweighted(df)  # Table 2 is unweighted in master.do
    for group, suffix in (("Commercial banks", "_b"), ("Placebo lenders", "_pl")):
        for base, label in (("Dl_nloans", "Number of loans"), ("Dl_vloans", "Loan volume"), ("Dl_lir", "Loan to income")):
            y = base + suffix
            r = fe_ols(d, y, [INSTRUMENT] + D_CONTROL + ["L" + y])
            pc, ps = PAPER_T2[y]
            record("A placebo (Table 2)", f"{group}: {label}", INSTRUMENT, r, {"paper_coef": pc, "paper_se": ps})


# --- B. Anticipation: leads of deregulation ------------------------------------------------------
def anticipation(df):
    # inter_bra_t = deregulation index in t; the instrument is its lag. Leads = deregulation in t+1, t+2.
    d = add_lead(df, "inter_bra", 1, "F1_dereg")
    d = add_lead(d, "inter_bra", 2, "F2_dereg")
    d = unweighted(d)
    r = fe_ols(d, "Dl_nloans_b", [INSTRUMENT, "F1_dereg", "F2_dereg"] + D_CONTROL + ["LDl_nloans_b"])
    for v in (INSTRUMENT, "F1_dereg", "F2_dereg"):
        record("B anticipation", "Credit (number of loans) on deregulation lags and leads", v, r)
    d = add_lead(df, "inter_bra", 1, "F1_dereg")
    d = add_lead(d, "inter_bra", 2, "F2_dereg")
    r = fe_ols(d, "Dl_hpi", [INSTRUMENT, "F1_dereg", "F2_dereg", "LDl_hpi"] + D_CONTROL_HP)
    for v in (INSTRUMENT, "F1_dereg", "F2_dereg"):
        record("B anticipation", "House prices on deregulation lags and leads", v, r)


# --- C + D. Specification checks on Table 5, column 1 ---------------------------------------------
def spec_checks(df):
    x = "Dl_nloans_b"
    specs = {
        "Baseline (Table 5, col. 1)": ["LDl_hpi"] + D_CONTROL_HP,
        "C: without lagged house price (Nickell)": D_CONTROL_HP,
        "D: only lagged house price, no other controls": ["LDl_hpi"],
        "D: no controls at all": [],
    }
    for name, exog in specs.items():
        r = fe_iv(df, "Dl_hpi", x, INSTRUMENT, exog)
        check = "C nickell" if name.startswith("C") else "D controls" if name.startswith("D") else "baseline"
        record(check, name, x, r)


# --- I. Weights -------------------------------------------------------------------------------------
def weighting(df):
    """Table 5 weights every STATE equally (w1 = 1 / counties in the state), so a county in Wyoming
    counts far more than one in California. Unweighted = every COUNTY counts equally."""
    exog = ["LDl_hpi"] + D_CONTROL_HP
    for x, label in CREDIT.items():
        r = fe_iv(unweighted(df), "Dl_hpi", x, INSTRUMENT, exog)
        record("I weights", f"Unweighted IV: {label}", x, r)
        r = fe_ols(unweighted(df), x, [INSTRUMENT] + exog)
        record("I weights", f"Unweighted first stage: {label}", INSTRUMENT, r)
        r = fe_ols(df, x, [INSTRUMENT] + exog)
        record("I weights", f"Weighted first stage: {label}", INSTRUMENT, r)
        r = fe_ols(unweighted(df), "Dl_hpi", [INSTRUMENT] + exog)
    record("I weights", "Unweighted reduced form (house prices on deregulation)", INSTRUMENT, r)
    r = fe_ols(df, "Dl_hpi", [INSTRUMENT] + exog)
    record("I weights", "Weighted reduced form (house prices on deregulation)", INSTRUMENT, r)


# --- H. OLS vs. IV ---------------------------------------------------------------------------------
def ols_vs_iv(df):
    """OLS counterpart of Table 5 on the same sample (county FE, years, controls, weights, county-HAC
    is not needed for the point estimate; SEs clustered by state). Compare with the IV estimate."""
    exog = ["LDl_hpi"] + D_CONTROL_HP
    for x, label in CREDIT.items():
        d = df.dropna(subset=["Dl_hpi", x, INSTRUMENT] + exog)
        d = d[d.groupby(PANEL_ID)[PANEL_ID].transform("size") > 1]
        record("H OLS vs IV", f"OLS (no instrument): {label}", x, fe_ols(d, "Dl_hpi", [x] + exog))


# --- E. Weak instruments: Anderson-Rubin confidence intervals -----------------------------------
def anderson_rubin(df, x, exog, grid=np.linspace(-1.0, 3.0, 401), bw=3, cluster=None):
    """AR test: for each beta0, regress (y - beta0*x) on the instrument + exog (county FE, years),
    HAC Wald test that the instrument's coefficient is zero. CI = all beta0 not rejected."""
    d, yrs = _prepare(df, ["Dl_hpi", x, INSTRUMENT] + exog, drop_singletons=True)
    regs = [INSTRUMENT] + exog + yrs
    t = _within(d, ["Dl_hpi", x] + regs)
    sw = np.sqrt(d[WEIGHT].values)
    Z = t[regs].values * sw[:, None]
    A = np.linalg.pinv(Z.T @ Z)
    n, n_c = len(d), d[PANEL_ID].nunique()
    stats = []
    for b0 in grid:
        ystar = (t["Dl_hpi"].values - b0 * t[x].values) * sw
        g = A @ Z.T @ ystar
        u = ystar - Z @ g
        if cluster:
            G = d[cluster].nunique()
            V = G / (G - 1) * A @ _cluster_meat(Z * u[:, None], d[cluster]) @ A
        else:
            V = n / (n - n_c) * A @ _bartlett_meat(Z * u[:, None], d[PANEL_ID].values, d[TIME_ID].values, bw) @ A
        stats.append(g[0] ** 2 / V[0, 0])
    stats = np.array(stats)
    out = {}
    for level, crit in (("90", 2.706), ("95", 3.841)):
        acc = grid[stats < crit]
        bounded = acc.size and acc.min() > grid.min() and acc.max() < grid.max()
        out[f"AR_{level}_lo"] = acc.min() if acc.size else np.nan
        out[f"AR_{level}_hi"] = acc.max() if acc.size else np.nan
        out[f"AR_{level}_bounded"] = bool(bounded)
    return out


def weak_iv(df):
    for x, label in CREDIT.items():
        r = fe_iv(df, "Dl_hpi", x, INSTRUMENT, ["LDl_hpi"] + D_CONTROL_HP)
        ar = anderson_rubin(df, x, ["LDl_hpi"] + D_CONTROL_HP)
        wald95 = (r.coef[x] - 1.96 * r.se[x], r.coef[x] + 1.96 * r.se[x])
        record("E weak IV", f"{label}: Anderson-Rubin vs. Wald CI", x, r,
               {**ar, "Wald_95_lo": wald95[0], "Wald_95_hi": wald95[1]})


# --- J. Clustering by state ----------------------------------------------------------------------
def state_clustering(df):
    """The instrument varies only by state x year, so errors of counties in the same state are likely
    correlated. Table 5 uses county-level HAC errors. Same IV, SEs clustered by state (50 clusters)."""
    exog = ["LDl_hpi"] + D_CONTROL_HP
    for x, label in CREDIT.items():
        r = fe_iv(df, "Dl_hpi", x, INSTRUMENT, exog, cluster="state_n")
        ar = anderson_rubin(df, x, exog, cluster="state_n")
        record("J state clusters", f"IV, SEs clustered by state: {label}", x, r, ar)


# --- F. Mechanism: housing supply elasticity -------------------------------------------------------
def elasticity_split(df):
    x = "Dl_nloans_b"
    d = df.dropna(subset=["elasticity"])
    med = d.drop_duplicates(PANEL_ID)["elasticity"].median()
    for name, sub in (("Inelastic supply (elasticity below median)", d[d.elasticity < med]),
                      ("Elastic supply (elasticity above median)", d[d.elasticity >= med])):
        r = fe_iv(sub, "Dl_hpi", x, INSTRUMENT, ["LDl_hpi"] + D_CONTROL_HP)
        record("F elasticity", name, x, r, {"median_elasticity": med})


# --- G. Extension: FHFA house prices -------------------------------------------------------------
# Download (free, no login) and save in data/raw/ as hpi_at_bdl_county.xlsx:
#   https://www.fhfa.gov/hpi/download/annual/hpi_at_county.xlsx   (file name on the FHFA site since 2025)
# FHFA "Annual House Price Indexes, Counties (Developmental Index)", repeat-sales, all transactions.
FHFA_FILE = RAW / "hpi_at_bdl_county.xlsx"


def load_fhfa() -> pd.DataFrame:
    """Read the FHFA county file. The header row is found automatically (the sheet has title rows on top)."""
    raw = pd.read_excel(FHFA_FILE, header=None)
    # header = the first row with a cell that is exactly "Year" (the description rows above mention "FIPS" too)
    header_row = next(i for i in range(min(30, len(raw)))
                      if raw.iloc[i].astype(str).str.strip().str.lower().eq("year").any())
    raw.columns = [str(c).strip() for c in raw.iloc[header_row]]
    raw = raw.iloc[header_row + 1:]
    fips_col = next(c for c in raw.columns if "fips" in c.lower())
    year_col = next(c for c in raw.columns if c.lower() == "year")
    hpi_col = next(c for c in raw.columns if c.strip().upper() == "HPI")   # base: first year of each county
    f = pd.DataFrame({PANEL_ID: pd.to_numeric(raw[fips_col], errors="coerce"),
                      TIME_ID: pd.to_numeric(raw[year_col], errors="coerce"),
                      "hpi_fhfa": pd.to_numeric(raw[hpi_col], errors="coerce")}).dropna()
    f = f[f.hpi_fhfa > 0].sort_values([PANEL_ID, TIME_ID])
    consecutive = f.groupby(PANEL_ID)[TIME_ID].diff() == 1
    f["Dl_hpi_fhfa"] = np.log(f.hpi_fhfa).groupby(f[PANEL_ID]).diff().where(consecutive)
    f["LDl_hpi_fhfa"] = f.groupby(PANEL_ID)["Dl_hpi_fhfa"].shift(1).where(consecutive)
    return f[[PANEL_ID, TIME_ID, "Dl_hpi_fhfa", "LDl_hpi_fhfa"]]


def extension(df):
    if not FHFA_FILE.exists():
        print(f"  extension: {FHFA_FILE.name} not found in data/raw/, skipped "
              "(download link in 03_robustness_checks.py, section G)")
        return
    d = df.merge(load_fhfa(), on=[PANEL_ID, TIME_ID], how="left")
    both = d.dropna(subset=["Dl_hpi", "Dl_hpi_fhfa"])
    print(f"  extension: FHFA growth for {d.Dl_hpi_fhfa.notna().sum():,} of {len(d):,} county-years; "
          f"correlation with Moody's growth: {both.Dl_hpi.corr(both.Dl_hpi_fhfa):.2f}")
    for x, label in CREDIT.items():
        r = fe_iv(d, "Dl_hpi_fhfa", x, INSTRUMENT, ["LDl_hpi_fhfa"] + D_CONTROL_HP)
        record("G extension FHFA", f"FHFA house prices: {label}", x, r)
        # Moody's on exactly the same county-years, so differences come from the data, not the sample
        same = d.dropna(subset=["Dl_hpi_fhfa", "LDl_hpi_fhfa"])
        r2 = fe_iv(same, "Dl_hpi", x, INSTRUMENT, ["LDl_hpi"] + D_CONTROL_HP)
        record("G extension FHFA", f"Moody's, FHFA sample: {label}", x, r2)


# --- Output -------------------------------------------------------------------------------------
def fmt(r):
    return f"{r['coef']:.3f}{r['stars']}", f"({r['se']:.3f})"


def write_dashboard(res: pd.DataFrame):
    """Small file for the website: the IV effect (number of loans) under each check, plain-language labels."""
    pick = {
        "Baseline (Table 5, col. 1)": ("Paper's specification", "Original"),
        "IV, SEs clustered by state: Number of loans": ("Errors grouped by state", "How uncertainty is measured"),
        "Unweighted IV: Number of loans": ("Every county counts equally", "How counties are weighted"),
        "C: without lagged house price (Nickell)": ("Without last year's price growth", "Which controls are used"),
        "D: no controls at all": ("Without any controls", "Which controls are used"),
        "FHFA house prices: Number of loans": ("Public FHFA house prices", "Which house price data"),
        "Moody's, FHFA sample: Number of loans": ("Moody's, same counties as FHFA", "Which house price data"),
    }
    out = []
    for spec, (label, group) in pick.items():
        r = res[res.spec == spec]
        if len(r):
            r = r.iloc[0]
            out.append({"label": label, "group": group, "coef": r.coef, "se": r.se,
                        "first_stage_F": r.first_stage_F, "nobs": r.nobs})
    pd.DataFrame(out).to_csv(DASHBOARD_DATA / "robustness_results.csv", index=False)


def write_tables(res: pd.DataFrame):
    res.to_csv(TABLES / "robustness_all.csv", index=False)
    write_dashboard(res)

    a = res[res.check.str.startswith("A")]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r" & Number of loans & Loan volume & Loan to income \\", r"\midrule"]
    for group in ("Commercial banks", "Placebo lenders"):
        g = a[a.spec.str.startswith(group)]
        cells = [fmt(r) for _, r in g.iterrows()]
        lines += [f"{group}: deregulation & " + " & ".join(c for c, _ in cells) + r" \\",
                  " & " + " & ".join(s for _, s in cells) + r" \\",
                  "\\quad paper & " + " & ".join(f"{r.paper_coef:.3f} ({r.paper_se:.3f})" for _, r in g.iterrows()) + r" \\",
                  "\\quad observations & " + " & ".join(f"{int(r.nobs):,}" for _, r in g.iterrows()) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "placebo_table2.tex").write_text("\n".join(lines))

    rows = res[res.check.isin(["baseline", "C nickell", "D controls", "F elasticity"])
               | (res.check.isin(["H OLS vs IV"]) & res.spec.str.contains("Number of loans"))
               | res.spec.eq("Unweighted IV: Number of loans")
               | res.spec.eq("IV, SEs clustered by state: Number of loans")]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"\multicolumn{4}{l}{\textit{2SLS, number of loans}} \\",
             r"Specification & Coefficient (SE) & First-stage $F$ & Obs. \\", r"\midrule"]
    short = {"Baseline (Table 5, col. 1)": "Baseline (Table 5, col. 1)",
             "C: without lagged house price (Nickell)": "Without lagged house price",
             "D: only lagged house price, no other controls": "Only lagged house price as control",
             "D: no controls at all": "No controls",
             "OLS (no instrument): Number of loans": "OLS (no instrument)",
             "Unweighted IV: Number of loans": "Unweighted (no $w_1$)",
             "IV, SEs clustered by state: Number of loans": "SEs clustered by state",
             "Inelastic supply (elasticity below median)": "Inelastic supply (below median)",
             "Elastic supply (elasticity above median)": "Elastic supply (above median)"}
    for _, r in rows.iterrows():
        c, s = fmt(r)
        r = r.copy(); r["spec"] = short.get(r.spec, r.spec)
        fs = "--" if pd.isna(r.first_stage_F) else f"{r.first_stage_F:.1f}"
        lines.append(f"{r.spec} & {c} {s} & {fs} & {int(r.nobs):,}" + r" \\")
    # Panel B: 95% confidence sets, one row per credit measure
    e = res[res.check == "E weak IV"].reset_index(drop=True)
    j = res[res.check == "J state clusters"].reset_index(drop=True)

    def ci(lo, hi, bounded=True):
        return f"[{lo:.2f}, {hi:.2f}]" + ("" if bounded else "$^{a}$")

    lines += [r"\midrule", r"\multicolumn{4}{l}{\textit{95\% confidence sets}} \\",
              r"Credit measure & Wald (HAC) & AR (HAC) & AR (state clusters) \\", r"\midrule"]
    for k in range(len(e)):
        r, rj = e.loc[k], j.loc[k] if k < len(j) else None
        name = r.spec.split(":")[0]
        aj = ci(rj.AR_95_lo, rj.AR_95_hi, bool(rj.AR_95_bounded)) if rj is not None else "--"
        lines.append(f"{name} & {ci(r.Wald_95_lo, r.Wald_95_hi)} & {ci(r.AR_95_lo, r.AR_95_hi)} & {aj}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "robustness.tex").write_text("\n".join(lines))

    g = res[res.check.str.startswith("G")]
    if len(g):
        lines = [r"\begin{tabular}{lccc}", r"\toprule", r" & Number of loans & Loan volume & Loan to income \\", r"\midrule"]
        for prefix in ("FHFA house prices", "Moody's, FHFA sample"):
            gg = g[g.spec.str.startswith(prefix)]
            cells = [fmt(r) for _, r in gg.iterrows()]
            lines += [f"{prefix} & " + " & ".join(c for c, _ in cells) + r" \\", " & " + " & ".join(s for _, s in cells) + r" \\",
                      "\\quad first-stage $F$ / obs. & " + " & ".join(f"{r.first_stage_F:.1f} / {int(r.nobs):,}" for _, r in gg.iterrows()) + r" \\"]
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TABLES / "extension.tex").write_text("\n".join(lines))


def main() -> None:
    df = pd.read_parquet(PANEL_FILE)
    for step in (placebo_table2, anticipation, spec_checks, ols_vs_iv, weighting, state_clustering, weak_iv,
                 elasticity_split, extension):
        step(df)
    res = pd.DataFrame(RESULTS)
    write_tables(res)
    show = res[["check", "spec", "variable", "coef", "se", "stars", "nobs", "first_stage_F"]]
    with pd.option_context("display.width", 220, "display.max_colwidth", 60, "display.precision", 4):
        print(show.to_string(index=False))
        e = res[res.check == "E weak IV"]
        if len(e):
            print(e[["spec", "AR_90_lo", "AR_90_hi", "AR_95_lo", "AR_95_hi", "AR_95_bounded", "Wald_95_lo", "Wald_95_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
