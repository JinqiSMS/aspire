"""Run without installing; all project caches stay inside this directory."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for key, subdir in {"MPLCONFIGDIR": "cache/matplotlib", "TMP": "tmp", "TEMP": "tmp",
                    "PYTHONPYCACHEPREFIX": "cache/bytecode", "PIP_CACHE_DIR": "cache/pip"}.items():
    path = ROOT / subdir
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ["ASPIRE_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from aspire.run import main
    main()

