"""Configuration defaults are explicit experimental choices, not certificates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import yaml
from .architecture import Architecture

DEFAULTS = {
    "experiment_name": "strict_smoke", "kind": "recovery", "seed": 0, "seeds": [], "sweep": [],
    "randomness": {"teacher_seed": None, "algorithm_seed": None, "test_seed": None},
    "architecture": {"d": 8, "hidden_widths": [3,3], "k": 4},
    "teacher": {"family": "positive_partition", "first_layer": "orthonormal", "first_perturbation": .15,
        "kappa0_max": 10., "mu_min": .01, "max_generation_attempts": 100, "beta_range": [.05,.35],
        "symmetric_beta": .2, "t_scale": .01, "gamma_bound": None,
        "instance_path": None, "instance_sha256": None},
    "oracle": {"mode": "strict_real", "backend": "float64", "dps": 80,
        "cache_size": 50000, "max_real_learning_queries": 2000000, "max_degree_strict": 128,
        "wall_seconds": 3600,
        "precision_retry": {"enabled": True, "decimal_digits": [80,110], "relative_tolerance": 1e-8, "absolute_tolerance": 1e-10}},
    "recovery": {"first_span_points": 16, "intermediate_span_points": 64,
        "moment_samples_by_layer": [256], "original_direction_step": .25,
        "reduced_gradient_step_factor": .5, "final_hessian_step": .1,
        "moment_ridge": 0., "gap_floor": 1e-10},
    "sampler": {"mode": "persistent_chain", "chains": 2, "burn_in": 256, "thin": 4,
        "batch_size": 0,
        "endpoint_steps": 256, "bisection_absolute_tolerance": 1e-7, "max_bisection_iterations": 100,
        "max_zero_fraction": .2, "xi_by_layer": [0.], "xi_source": "exact_first_suffix_only"},
    "final_layer": {"tau_fin": .2, "perturbation_candidates": 1,
        "method": "two_hessian", "hessian_directions": 12, "combination_candidates": 64},
    "evaluation": {"parameter_delta": .1, "gaussian_test_points": 10000, "gaussian_batches": 4,
        "angular_risk_directions": 10000, "angular_risk_batches": 4, "rank_tolerance": 1e-10,
        "population_gap_directions": 0, "population_gap_batches": 2},
    "diagnostic": {"layer": 1, "points": 64, "prefix_perturbation": .001,
        "reference_directions": 4096, "degrees": [4,16,64], "precisions": [0,80,110]},
    "baseline": {"methods": ["supervised", "polynomial_krr"], "label_budget": 512,
        "validation_fraction": .2, "lambdas": [1e-8,1e-5,.01], "learning_rates": [.001,.01],
        "restarts": 2, "epochs": 200, "batch_size": 128, "max_train_points": 8192,
        "max_working_memory_mb": 1024, "ntk_width": 64, "kernel_trace_scaling": True},
    "output": {"directory": "results", "save_artifacts": True, "make_figures": True}
}

def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if key not in base: raise ValueError(f"Unknown configuration key: {key}")
        result[key] = merge(base[key], value) if isinstance(base[key], dict) else deepcopy(value)
    return result

def set_dotted(cfg, key, value):
    target = cfg
    parts = key.split(".")
    for part in parts[:-1]: target = target[part]
    if parts[-1] not in target: raise ValueError(f"Unknown sweep key: {key}")
    target[parts[-1]] = deepcopy(value)

def resolve(raw):
    cfg = merge(DEFAULTS, raw)
    arch = Architecture(**cfg["architecture"])
    if cfg["kind"] not in ("recovery","single_layer","prefix_error","oracle_accuracy","sampler_check","baselines"):
        raise ValueError("Unknown experiment kind")
    if cfg["oracle"]["mode"] not in ("strict_real","complex_debug","true_suffix_debug","exact_prefix_debug"):
        raise ValueError("Unknown oracle mode")
    if cfg["oracle"]["backend"] not in ("float64","mpmath"): raise ValueError("Unknown backend")
    if cfg["sampler"]["mode"] not in ("persistent_chain","independent_endpoints"): raise ValueError("Unknown sampler mode")
    if not 0 < cfg["final_layer"]["tau_fin"] < 1: raise ValueError("tau_fin must be in (0,1)")
    final = cfg["final_layer"]
    if final["method"] not in ("two_hessian", "random_combination"):
        raise ValueError("Unknown final_layer.method")
    if final["method"] == "random_combination":
        if final["hessian_directions"] <= 0 or final["hessian_directions"] % arch.widths[-2]:
            raise ValueError("Final Hessian directions must comprise complete orthogonal bases")
        if final["combination_candidates"] <= 0:
            raise ValueError("Positive combination_candidates required")
        if final["tau_fin"] + cfg["recovery"]["final_hessian_step"] >= 1:
            raise ValueError("Final Hessian nodes must remain positive")
    moments = cfg["recovery"]["moment_samples_by_layer"]
    if len(moments)==1: cfg["recovery"]["moment_samples_by_layer"] = moments*(arch.L-2)
    if len(cfg["recovery"]["moment_samples_by_layer"]) != arch.L-2: raise ValueError("One moment sample count per non-final layer")
    xi = cfg["sampler"]["xi_by_layer"]
    if len(xi) != arch.L-2 or any(not 0<=v<=.5 for v in xi):
        raise ValueError("Specify xi separately for every non-final layer; higher-layer zero is not assumed")
    if min(cfg["recovery"]["moment_samples_by_layer"]) < 2: raise ValueError("Too few samples")
    for field in ("chains","thin","endpoint_steps","max_bisection_iterations"):
        if cfg["sampler"][field] <= 0: raise ValueError(f"Positive sampler.{field} required")
    if cfg["sampler"]["burn_in"] < 0: raise ValueError("Negative burn-in")
    if cfg["sampler"]["batch_size"] < 0: raise ValueError("Negative batch size")
    if cfg["oracle"]["max_real_learning_queries"] < 1: raise ValueError("Positive query budget required")
    if cfg["oracle"]["dps"] < 30: raise ValueError("Use at least 30 decimal digits for mp backend")
    return cfg

def read_config(path):
    with open(path, encoding="utf-8-sig") as f: return yaml.safe_load(f) or {}

def fingerprint(cfg):
    data = deepcopy(cfg)
    data.pop("output", None); data.pop("seeds", None); data.pop("sweep", None)
    for key in ("max_real_learning_queries","wall_seconds"): data["oracle"].pop(key, None)
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:12]
