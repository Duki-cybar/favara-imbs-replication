"""
02_iv_regression.py: main results of Favara & Imbs (2015).

Table 4 (reduced form), master.do lines 100-104:
    xtreg Dl_hpi Linter_bra [Linter_ela] [LDl_hpi controls] yr* [aw=w1], fe cl(state_n)
Table 5 (IV, main result), master.do lines 116-120:
    xtivreg2 Dl_hpi (Dl_L = Linter_bra) LDl_hpi controls yr* [aw=w1], fe r bw(3) partial(yr*)
    with Dl_L = Dl_nloans_b (number of loans), Dl_vloans_b (volume), Dl_lir_b (loan to income)

Model of Table 5 (equations 3 and 4 in the paper):
    first stage : dlog L_ct = beta1 * D_s,t-1 + beta2 X_ct + a_c + g_t + e_ct
    second stage: dlog P_ct = delta1 * dlog L_ct(hat) + delta2 X_ct + a_c + g_t + e_ct
    c = county, s = state, t = year; D = lagged deregulation index (Linter_bra)

Outputs
    paper/tables/table4_reduced_form.tex
    paper/tables/table5_iv.tex
    paper/tables/comparison_with_paper.csv    (numbers for Worksheet A.3)
    dashboard/data/regression_results.csv     (for the website)
"""
from __future__ import annotations

import pandas as pd

from config import DASHBOARD_DATA, INSTRUMENT, PANEL_FILE, TABLES
from estimators import fe_iv, fe_ols
from paper_values import TABLE4, TABLE5

CONTROLS = ["Dl_inc", "LDl_inc", "Dl_pop", "LDl_pop", "Dl_her_v", "LDl_her_v"]  # $D_control_hp
Y = "Dl_hpi"

TABLE4_SPECS = {
    "(1)": [INSTRUMENT],
    "(2)": [INSTRUMENT, "Linter_ela"],
    "(3)": [INSTRUMENT, "Linter_ela", "LDl_hpi"] + CONTROLS,
}
TABLE5_SPECS = {
    "(1) Number of loans": "Dl_nloans_b",
    "(2) Loan volume": "Dl_vloans_b",
    "(3) Loan to income": "Dl_lir_b",
}


def run_table4(df):
    return {col: fe_ols(df, Y, x) for col, x in TABLE4_SPECS.items()}


def run_table5(df):
    return {col: fe_iv(df, Y, endog, INSTRUMENT, ["LDl_hpi"] + CONTROLS)
            for col, endog in TABLE5_SPECS.items()}


def comparison(t4, t5) -> pd.DataFrame:
    rows = []
    for table, res, paper, var in (("Table 4", t4, TABLE4, lambda c: INSTRUMENT),
                                   ("Table 5", t5, TABLE5, lambda c: TABLE5_SPECS[c])):
        for col, r in res.items():
            p, v = paper[col], var(col)
            rows.append({
                "table": table, "column": col, "variable": v,
                "coef_ours": r.coef[v], "coef_paper": p["coef"],
                "se_ours": r.se[v], "se_paper": p["se"],
                "N_ours": r.nobs, "N_paper": p["N"],
                "counties_ours": r.n_counties, "counties_paper": p["counties"],
                "F_ours": r.first_stage_F, "F_paper": p.get("F"),
            })
    return pd.DataFrame(rows)


def _cell(r, v, digits):
    return f"{r.coef[v]:.{digits}f}{r.stars(v)}", f"({r.se[v]:.{digits}f})"


def tex_table4(t4) -> str:
    cols = list(t4)
    labels = {INSTRUMENT: "Deregulation index", "Linter_ela": "Deregulation $\\times$ supply elasticity",
              "LDl_hpi": "Lagged house price"}
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(cols) + r" \\", r"\midrule"]
    for v, lab in labels.items():
        coefs, ses = [], []
        for c in cols:
            if v in t4[c].coef:
                a, b = _cell(t4[c], v, 4 if v != "LDl_hpi" else 3)
            else:
                a, b = "", ""
            coefs.append(a); ses.append(b)
        lines += [f"{lab} & " + " & ".join(coefs) + r" \\", " & " + " & ".join(ses) + r" \\"]
    lines += [r"\midrule",
              "Paper: deregulation index & " + " & ".join(f"{TABLE4[c]['coef']:.4f}" for c in cols) + r" \\",
              " & " + " & ".join(f"({TABLE4[c]['se']:.3f})" for c in cols) + r" \\",
              r"\midrule",
              "Controls & No & No & Yes" + r" \\",
              "Observations & " + " & ".join(f"{t4[c].nobs:,}" for c in cols) + r" \\",
              "Counties & " + " & ".join(f"{t4[c].n_counties:,}" for c in cols) + r" \\",
              "$R^2$ (within) & " + " & ".join(f"{t4[c].r2_within:.3f}" for c in cols) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tex_table5(t5) -> str:
    cols = list(t5)
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(c.split(" ", 1)[0] for c in cols) + r" \\",
             " & " + " & ".join(c.split(" ", 1)[1] for c in cols) + r" \\", r"\midrule"]
    coefs, ses = zip(*(_cell(t5[c], TABLE5_SPECS[c], 3) for c in cols))
    lines += ["Instrumented credit growth & " + " & ".join(coefs) + r" \\", " & " + " & ".join(ses) + r" \\"]
    lag = [_cell(t5[c], "LDl_hpi", 3) for c in cols]
    lines += ["Lagged house price & " + " & ".join(a for a, _ in lag) + r" \\",
              " & " + " & ".join(b for _, b in lag) + r" \\", r"\midrule",
              "Paper: credit growth & " + " & ".join(f"{TABLE5[c]['coef']:.3f}" for c in cols) + r" \\",
              " & " + " & ".join(f"({TABLE5[c]['se']:.3f})" for c in cols) + r" \\", r"\midrule",
              "First-stage $F$ & " + " & ".join(f"{t5[c].first_stage_F:.2f}" for c in cols) + r" \\",
              "First-stage $F$ (paper) & " + " & ".join(f"{TABLE5[c]['F']:.2f}" for c in cols) + r" \\",
              "Observations & " + " & ".join(f"{t5[c].nobs:,}" for c in cols) + r" \\",
              "Counties & " + " & ".join(f"{t5[c].n_counties:,}" for c in cols) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    df = pd.read_parquet(PANEL_FILE)
    t4, t5 = run_table4(df), run_table5(df)

    comp = comparison(t4, t5)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print(comp[["table", "column", "coef_ours", "coef_paper", "se_ours", "se_paper",
                    "N_ours", "N_paper", "F_ours", "F_paper"]].to_string(index=False))

    (TABLES / "table4_reduced_form.tex").write_text(tex_table4(t4))
    (TABLES / "table5_iv.tex").write_text(tex_table5(t5))
    comp.to_csv(TABLES / "comparison_with_paper.csv", index=False)
    comp.to_csv(DASHBOARD_DATA / "regression_results.csv", index=False)
    print(f"\nWrote tables to {TABLES}")


if __name__ == "__main__":
    main()
