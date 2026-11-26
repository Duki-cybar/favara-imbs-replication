"""Central paths and constants. Every script imports from here -> only relative paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "paper" / "tables"
FIGURES = ROOT / "paper" / "figures"
DASHBOARD_DATA = ROOT / "dashboard" / "data"

PANEL_FILE = PROCESSED / "county_panel.parquet"

YEAR_MIN, YEAR_MAX = 1994, 2005   # sample period of the paper

# Variable names as they appear in the replication package (see literature/Replication_Package_README.pdf)
PANEL_ID = "county"     # county FIPS code -> county fixed effects
TIME_ID = "year"        # -> year dummies
STATE_ID = "state_n"    # state identifier; cluster level of Tables 2-4
WEIGHT = "w1"           # inverse number of counties per state ([aw=w1] in master.do)
INSTRUMENT = "Linter_bra"   # lagged deregulation index, 4 - Rice-Strahan index (0 = most restricted, 4 = least)

SEED = 20261112

for _p in (PROCESSED, TABLES, FIGURES, DASHBOARD_DATA):
    _p.mkdir(parents=True, exist_ok=True)
