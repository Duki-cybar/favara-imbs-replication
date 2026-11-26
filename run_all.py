"""Master script: runs the full analysis from raw data (Reproducibility checklist, Appendix A.5)."""
import runpy
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

for name in ("01_data_cleaning.py", "02_iv_regression.py", "02b_impulse_responses.py",
             "03_robustness_checks.py"):
    print(f"\n=== {name} ===")
    runpy.run_path(str(SCRIPTS / name), run_name="__main__")
