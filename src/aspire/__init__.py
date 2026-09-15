"""ASPIRE. Mathematical provenance is recorded in module/function docstrings."""
import os
from pathlib import Path

__version__ = "0.1.0"
_project = next((p for p in Path(__file__).resolve().parents if (p / "run_experiments.py").exists()), Path.cwd())
ROOT = Path(os.environ.get("ASPIRE_ROOT", _project)).resolve()
for _key, _subdir in {"MPLCONFIGDIR":"cache/matplotlib", "TMP":"tmp", "TEMP":"tmp", "PIP_CACHE_DIR":"cache/pip"}.items():
    _path=ROOT / _subdir
    _path.mkdir(parents=True,exist_ok=True)
    os.environ[_key]=str(_path)
