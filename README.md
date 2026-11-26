# Replication: Favara & Imbs (2015), "Credit Supply and the Price of Housing"

Replication and extension for *Empirical Methods* (UZH, HS 2026), Problem Sets 3 & 4.

Favara, G., & Imbs, J. (2015). Credit Supply and the Price of Housing.
*American Economic Review*, 105(3), 958–992. https://doi.org/10.1257/aer.20121416

## Structure
```
data/raw/          FHFA file (in Git) + replication data from the openICPSR package (not in Git, never modified)
data/processed/    county-year panel produced by scripts/01 (not in Git)
scripts/
  config.py               paths, variable names, constants (relative paths only)
  01_data_cleaning.py     merge hmda + hp_dereg_controls + call -> county-year panel (as in master.do)
  02_iv_regression.py     Table 4 (reduced form) and Table 5 (IV), compared with the paper
  02b_impulse_responses.py Figures 1 and 3 (local projections), PDFs in paper/figures/
  03_robustness_checks.py placebo (Table 2), anticipation, Nickell, controls, OLS vs. IV, weights,
                          state clustering, Anderson-Rubin, supply elasticity, FHFA extension
  estimators.py           panel OLS / IV with county FE; state-clustered and HAC (Bartlett) SEs as in Stata
  paper_values.py         published coefficients, used for the automatic comparison
run_all.py         master script: runs 01 -> 02 -> 02b -> 03 from the raw data
paper/             Overleaf project (main.tex; tables/ and figures/ are written by the scripts)
dashboard/         Streamlit app for PS4 (app.py + small aggregated data in dashboard/data/)
```

## Setup (once per person)
1. Download the replication package from openICPSR: https://doi.org/10.3886/E112894V1
   (free account needed) and unzip it.
2. Copy all 7 `.dta` files from `20121416_1data/data/` into `data/raw/`.
3. Install the packages and run everything:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py          # all tables/figures -> paper/tables, paper/figures
streamlit run dashboard/app.py
```

## Replication status
| | Paper | Ours |
|---|---|---|
| Table 4, col. 1 (reduced form) | 0.0103 (0.004), N = 12,214 | 0.0103 (0.004), N = 12,214 |
| Table 5, col. 1 (IV, number of loans) | 0.141 (0.069), F = 10.13 | 0.141 (0.069), F = 10.15 |
| Table 5, col. 2 (IV, loan volume) | 0.134 (0.062), F = 10.31 | 0.135 (0.063), F = 10.33 |
| Table 5, col. 3 (IV, loan to income) | 0.120 (0.057), F = 8.59 | 0.120 (0.058), F = 8.61 |

| Figure 3 (impulse response of house prices) | peak about 0.2 after 2 years | peak 0.19-0.26 after 2 years, same shape and bands |
| Figure 1 (impulse response of credit) | peak on impact, insignificant after 4 years | same |

| Table 2 (placebo: banks vs. other lenders) | 0.028 (0.010) vs. 0.002 (0.008) | identical |

Full comparison: `paper/tables/comparison_with_paper.csv` and `paper/tables/irf_estimates.csv` (created locally by `run_all.py`).

## Data sources
All data come from the authors' replication package (accessed 11 Nov 2026). The authors
built the county-year files from the following sources; we do not process these sources ourselves.

| Variable(s) | Original source | In file |
|---|---|---|
| Mortgage originations, denials, loan-to-income | HMDA (FFIEC), IRS | `hmda.dta` |
| Number of branches | FDIC Summary of Deposits | `hmda.dta` |
| Bank balance sheets | Call Reports (FFIEC) | `call.dta` |
| House price index, housing stock | Moody's Economy.com (proprietary) | `hp_dereg_controls.dta` |
| Income per capita, population | BEA | `hp_dereg_controls.dta` |
| Interstate branching deregulation index | Rice & Strahan (2010) | `hp_dereg_controls.dta` |
| Housing supply elasticity | Saiz (2010) | `hp_dereg_controls.dta` |

The FHFA county house price index used in the extension is included in `data/raw/`. All other data come from the
authors' replication package on openICPSR (https://doi.org/10.3886/E112894V1) and include proprietary Moody's house
price data, so they are not included here; download them and place the `.dta` files in `data/raw/` as described in Setup.

Extension: FHFA annual county house price index, downloaded without login from
https://www.fhfa.gov/hpi/download/annual/hpi_at_county.xlsx (accessed 12 Nov 2026) and saved as `data/raw/hpi_at_bdl_county.xlsx`.

Software: Python 3.13, package versions in `requirements.txt`.

## Website (PS4): deploy on Streamlit Community Cloud
1. Run `python run_all.py` locally and commit the CSV files in `dashboard/data/` (small aggregates only; no raw data).
2. Sign in at https://share.streamlit.io with GitHub (use the team pseudonym account if you want the URL anonymous).
3. "Create app" → repository `favara-imbs-replication`, branch `main`, main file path `dashboard/app.py`.
4. Under "Advanced settings" choose Python 3.13; Streamlit installs `requirements.txt` from the repo root.
5. Check the public URL on laptop and phone, then hand in repo link + URL on OLAT.

