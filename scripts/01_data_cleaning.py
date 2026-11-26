"""
01_data_cleaning.py: build the county-year analysis panel for Favara & Imbs (2015).

The replication package already contains county x year data that the authors built
from HMDA, Call Reports, FDIC SOD, BEA, IRS and Moody's Economy.com. So this script
only does what master.do does before Tables 2-5:

    use hmda.dta
    merge 1:1 county year using hp_dereg_controls.dta, keep(1 3)
    merge 1:1 county year using call.dta, keep(1 3)

Input : data/raw/hmda.dta, data/raw/hp_dereg_controls.dta, data/raw/call.dta
        (copy of 20121416_1data/data/ from the openICPSR package, never modified)
Output: data/processed/county_panel.parquet
        dashboard/data/deregulation_state_year.csv (small aggregate for the website)

The row counts printed here are needed for Worksheet A.3 ("Sample restrictions").
"""
from __future__ import annotations

import pandas as pd

from config import (DASHBOARD_DATA, PANEL_FILE, PANEL_ID, RAW, STATE_ID,
                    TIME_ID, YEAR_MAX, YEAR_MIN)

KEYS = [PANEL_ID, TIME_ID]


def load(name: str) -> pd.DataFrame:
    """Read one .dta file from data/raw and check that county x year is a unique key."""
    path = RAW / f"{name}.dta"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy the contents of 20121416_1data/data/ from the "
            "openICPSR package (https://doi.org/10.3886/E112894V1) into data/raw/."
        )
    df = pd.read_stata(path, convert_categoricals=False)
    assert not df.duplicated(KEYS).any(), f"{name}: county x year is not unique"
    print(f"  {name:20s} {len(df):>7,} rows, {df[PANEL_ID].nunique():>5,} counties")
    return df


def build_panel() -> pd.DataFrame:
    """Left-merge as in master.do (keep(1 3) = keep master and matched rows)."""
    hmda = load("hmda")
    panel = hmda
    for name in ("hp_dereg_controls", "call"):
        using = load(name)
        # Like Stata's merge: variables already in the master data keep the master values
        using = using.drop(columns=[c for c in using.columns if c in panel.columns and c not in KEYS])
        panel = panel.merge(using, on=KEYS, how="left", validate="1:1")
    assert len(panel) == len(hmda), "merge changed the number of rows"
    panel = panel[panel[TIME_ID].between(YEAR_MIN, YEAR_MAX)]
    for col in KEYS:
        panel[col] = panel[col].astype(int)
    return panel


def describe(panel: pd.DataFrame) -> None:
    """Print the numbers to compare with the paper (Table 4 col. 1: 12,214 obs., 1,021 counties)."""
    print(f"\nPanel: {len(panel):,} county-years, {panel[PANEL_ID].nunique():,} counties, "
          f"{panel[STATE_ID].nunique()} states, {panel[TIME_ID].min()}-{panel[TIME_ID].max()}")
    hp = panel.dropna(subset=["Dl_hpi"])
    print(f"With house price growth (Dl_hpi): {len(hp):,} obs., {hp[PANEL_ID].nunique():,} counties")


def main() -> None:
    print("Reading raw data:")
    panel = build_panel()
    describe(panel)
    panel.to_parquet(PANEL_FILE, index=False)
    print(f"Saved {PANEL_FILE.relative_to(PANEL_FILE.parents[2])}")

    # Small state x year file for the dashboard map.
    # state_n is the 2-digit state FIPS code (= county FIPS // 1000);
    # inter_bra = 4 - Rice-Strahan index: 0 = most restricted, 4 = fully open to out-of-state branching.
    dereg = (panel[[STATE_ID, TIME_ID, "inter_bra"]]
             .drop_duplicates([STATE_ID, TIME_ID]).sort_values([STATE_ID, TIME_ID]))
    dereg = pd.DataFrame({
        "state_fips": dereg[STATE_ID].astype(int).astype(str).str.zfill(2),
        "year": dereg[TIME_ID].astype(int),
        "dereg_index": dereg["inter_bra"],
    })
    dereg.to_csv(DASHBOARD_DATA / "deregulation_state_year.csv", index=False)


if __name__ == "__main__":
    main()
