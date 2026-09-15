"""Portable entry point for the complete ASPIRE reproduction pipeline."""
from pathlib import Path
import os
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

from aspire.experiments.reproduction import main

if __name__ == "__main__":
    main()
