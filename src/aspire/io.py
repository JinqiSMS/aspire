"""Bounded paths and finite, atomic JSON output."""
from pathlib import Path
from dataclasses import asdict, is_dataclass
import json
import math
import mpmath as mp
import numpy as np
from . import ROOT

def inside(path):
    # Historical records used Windows separators; accept them on every OS.
    p = Path(str(path).replace("\\", "/"))
    p = (ROOT/p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT): raise ValueError(f"Output must stay inside {ROOT}")
    return p

def clean(value):
    if is_dataclass(value): value = asdict(value)
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, (mp.mpf,mp.mpc)): return clean(float(mp.re(value)))
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,Path): return str(value)
    return value

def write_json(path, obj):
    p = inside(path); p.parent.mkdir(parents=True,exist_ok=True)
    temp = p.with_suffix(p.suffix+".tmp")
    temp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(p)

def save_npz(path, **arrays):
    p=inside(path); p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p, **{k:np.asarray(v) for k,v in arrays.items()})
