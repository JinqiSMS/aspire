"""convex_body_sampling.tex lem:noisy-hit-and-run; §6 of experiment spec."""
import math
import numpy as np
from ..numeric import finite
from ..status import NumericalFailure

def radius_bounds(architecture, layer, public_bounds):
    kappa, mu = public_bounds["kappa0"], public_bounds["mu"]
    n = architecture.hidden_widths[0] if layer == 1 else architecture.widths[layer-1]
    return 1 / (2*kappa), 2*kappa*mu ** (-1/(architecture.k-1))*math.sqrt(n)

def hr_step(z, value, *, xi, outer_radius, tolerance, max_iterations, rng):
    if not 0 <= xi <= .5 or outer_radius <= 0 or tolerance <= 0:
        raise ValueError("Invalid HR parameters")
    direction = rng.normal(size=len(z)); direction /= np.linalg.norm(direction)
    endpoints = []
    for sign in (-1, 1):
        lo, hi = 0., 3*outer_radius
        for _ in range(max_iterations):
            if hi-lo <= tolerance: break
            mid = (hi+lo)/2
            y = value(z + sign*mid*direction)
            if not finite(y): raise NumericalFailure("nonfinite_oracle_value")
            if y <= 1-xi: lo = mid
            else: hi = mid
        if hi-lo > tolerance: raise NumericalFailure("bisection_limit_exceeded")
        endpoints.append(lo)
    minus, plus = endpoints
    if minus+plus == 0: return z.copy(), True
    # No rejection at the sampled interior point: proof uses TRUE-body convexity.
    return z + rng.uniform(-minus, plus)*direction, False

def effective_sample_size(values):
    """Initial positive pair autocorrelation sum, for one stationary chain."""
    v = np.asarray(values, float)
    if v.ndim == 1: v = v[:, None]
    n = len(v)
    if n < 8: return np.full(v.shape[1], np.nan)
    centered = v-v.mean(axis=0)
    size = 1 << (2*n-1).bit_length()
    f = np.fft.rfft(centered, n=size, axis=0)
    ac = np.fft.irfft(f.conj()*f, n=size, axis=0)[:n]
    variance = ac[0].copy()
    ac /= np.where(variance > 0, variance, 1)
    tau = np.ones(v.shape[1]); active = variance > 0
    for lag in range(1, min(n-1, n//2), 2):
        pair = ac[lag] + ac[lag+1]
        active &= pair > 0
        tau += 2*np.where(active, pair, 0)
    return np.where(variance > 0, np.clip(n/tau, 1, n), 0)

def hr_step_batch(z, batch_value, *, xi, outer_radius, tolerance, max_iterations,
                  directions, fractions):
    """Independent HR transitions; supplied randomness permits scalar replay."""
    if not 0 <= xi <= .5 or outer_radius <= 0 or tolerance <= 0:
        raise ValueError("Invalid HR parameters")
    directions = directions / np.linalg.norm(directions,axis=1)[:,None]
    signed = np.concatenate([-directions,directions])
    origins = np.concatenate([z,z])
    lo = np.zeros(2*len(z)); hi = np.full(2*len(z),3*outer_radius)
    for _ in range(max_iterations):
        if np.max(hi-lo) <= tolerance: break
        mid = (hi+lo)/2
        values = np.asarray(batch_value(origins+mid[:,None]*signed))
        if values.shape != (len(lo),) or not np.isfinite(values).all():
            raise NumericalFailure("nonfinite_oracle_value")
        inside = values <= 1-xi
        lo = np.where(inside,mid,lo); hi = np.where(inside,hi,mid)
    if np.max(hi-lo) > tolerance: raise NumericalFailure("bisection_limit_exceeded")
    minus,plus = lo[:len(z)],lo[len(z):]
    return z+(-minus+fractions*(minus+plus))[:,None]*directions, int(np.sum(minus+plus == 0))


def sample_independent_batch(batch_value, n, count, *, config, xi, outer_radius, rng, ledger):
    import time
    size=config["batch_size"]; steps=config["endpoint_steps"]
    points=np.empty((count,n)); zeros=transitions=0; last_log=time.perf_counter()
    with ledger.scope("membership"):
        for start in range(0,count,size):
            used=min(size,count-start); z=np.zeros((used,n))
            for _ in range(steps):
                z,zero=hr_step_batch(z,batch_value,xi=xi,outer_radius=outer_radius,
                    tolerance=config["bisection_absolute_tolerance"],
                    max_iterations=config["max_bisection_iterations"],
                    directions=rng.normal(size=(used,n)),fractions=rng.uniform(size=used))
                zeros+=zero; transitions+=used; ledger.tick("n_hr_transitions",used)
                if transitions>=100 and zeros/transitions>config["max_zero_fraction"]:
                    raise NumericalFailure("zero_chord_excess",zeros=zeros,transitions=transitions)
            points[start:start+used]=z; ledger.tick("n_kept_samples",used)
            if time.perf_counter()-last_log>20:
                print(f"  independent endpoints {start+used}/{count}; queries={ledger.counts['n_real_calls_total']}",flush=True)
                last_log=time.perf_counter()
    return points,np.arange(count),{"zero_chords":zeros,"transitions":transitions,
        "sampler_mode":"independent_endpoints","xi":xi,"outer_radius":outer_radius,
        "chains":count,"endpoint_steps":steps,"batch_size":size,
        "randomness_layout":"independent normal directions and uniform fractions per endpoint/step"}


def sample_body(value, n, count, *, config, xi, outer_radius, rng, ledger, batch_value=None):
    mode = config["mode"]
    if count < n+1: raise NumericalFailure("moment_not_spd", explanation="Too few samples")
    if mode == "independent_endpoints" and config.get("batch_size",0) and batch_value is not None:
        return sample_independent_batch(batch_value,n,count,config=config,xi=xi,
            outer_radius=outer_radius,rng=rng,ledger=ledger)
    chains = count if mode == "independent_endpoints" else config["chains"]
    allocations = [1]*count if mode == "independent_endpoints" else [count//chains + (i<count%chains) for i in range(chains)]
    points, ids, zeros, transitions = [], [], 0, 0
    with ledger.scope("membership"):
        for chain, keep in enumerate(allocations):
            if keep == 0: continue
            z = np.zeros(n)
            burn = config["endpoint_steps"] if mode == "independent_endpoints" else config["burn_in"]
            steps = burn if mode == "independent_endpoints" else burn + keep*config["thin"]
            for t in range(steps):
                z, zero = hr_step(z, value, xi=xi, outer_radius=outer_radius,
                    tolerance=config["bisection_absolute_tolerance"], max_iterations=config["max_bisection_iterations"], rng=rng)
                transitions += 1; zeros += zero; ledger.tick("n_hr_transitions")
                keep_now = t == steps-1 if mode == "independent_endpoints" else t>=burn and (t-burn+1)%config["thin"]==0
                if keep_now:
                    points.append(z.copy()); ids.append(chain); ledger.tick("n_kept_samples")
                if transitions >= 100 and zeros/transitions > config["max_zero_fraction"]:
                    raise NumericalFailure("zero_chord_excess", zeros=zeros, transitions=transitions)
    return np.asarray(points), np.asarray(ids), {"zero_chords": zeros, "transitions": transitions,
           "sampler_mode": mode, "xi": xi, "outer_radius": outer_radius, "chains": chains}
