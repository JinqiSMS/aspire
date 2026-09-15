import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache" / "matplotlib")
os.environ["ASPIRE_ROOT"] = str(ROOT)

