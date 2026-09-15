"""construction.tex eq:wc-explicit-weights + spec §10 experimental families."""
import numpy as np
import hashlib
import json
from .teacher import Teacher
from .status import NumericalFailure
from .io import inside


def _frozen_instance(architecture, config):
    """Evaluator-owned, content-pinned latent parameters; never learner input."""
    path=inside(config['instance_path'])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=config['instance_sha256']:
        raise ValueError('Frozen instance checksum mismatch')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data['k']!=architecture.k or tuple(data['hidden_widths'])!=architecture.hidden_widths:
        raise ValueError('Frozen instance architecture mismatch')
    weights=[np.asarray(w,float) for w in data['higher_weights']]
    a=np.asarray(data['output_weights'],float)
    shapes=list(zip(architecture.hidden_widths,architecture.hidden_widths[1:]))
    if len(weights)!=len(shapes) or any(w.shape!=shape for w,shape in zip(weights,shapes)):
        raise ValueError('Frozen instance weight shape mismatch')
    if a.shape!=(architecture.hidden_widths[-1],):raise ValueError('Frozen output shape mismatch')
    for w in weights+[a[:,None]]:
        if not np.isfinite(w).all() or w.min()<=0 or not np.allclose(w.sum(axis=0),1,rtol=0,atol=1e-12):
            raise ValueError('Frozen instance violates positivity or normalization')
    return weights,a,{'instance_id':data['instance_id'],'instance_sha256':digest,
        'instance_path':str(path),'verified_population_gaps':data['verified_population_gaps'],
        'gap_certificate':data['gap_certificate']}

def diagnose(weights, a):
    matrices = weights + [a[:, None]]
    conds = [float(np.linalg.cond(w)) for w in weights]
    mus = [w.shape[0] / w.shape[1] * float(np.min(w.sum(axis=1))) for w in matrices[1:]]
    return {"condition_numbers": conds, "mu_actual": min(mus), "row_balance": mus,
            "minimum_elements": [float(w.min()) for w in matrices],
            "column_normalization_error": [float(np.max(np.abs(np.linalg.norm(weights[0], axis=0) - 1)))]
                 + [float(np.max(np.abs(w.sum(axis=0) - 1))) for w in matrices[1:]],
            "kappa_actual": max(conds), "gap_certificate": "not_certified"}

def generate(architecture, config, rng):
    rejects = []
    frozen=_frozen_instance(architecture,config) if config['family']=='frozen' else None
    # Keep the latent network fixed when d changes: embedding draws are separate.
    embedding_rng=np.random.default_rng(int(rng.integers(2**32)))
    rng=np.random.default_rng(int(rng.integers(2**32)))
    for attempt in range(config["max_generation_attempts"]):
        first, _ = np.linalg.qr(embedding_rng.normal(size=(architecture.d, architecture.hidden_widths[0])))
        if config["first_layer"] == "nonorthogonal":
            r = first.shape[1]
            first = first @ (np.eye(r) + config["first_perturbation"] * embedding_rng.normal(size=(r, r)))
            first /= np.linalg.norm(first, axis=0)
        weights = [first]
        family = config["family"]
        for index,(n, s) in enumerate(zip(architecture.hidden_widths, architecture.hidden_widths[1:])):
            if family == 'frozen':
                w=frozen[0][index].copy()
            elif family == "positive_partition":
                parts = np.array_split(rng.permutation(n), s)
                c = np.zeros((n, s))
                for j, indices in enumerate(parts): c[indices, j] = 1 / len(indices)
                v = rng.uniform(.2, 1.2, (n, s)); v /= v.sum(axis=0)
                beta = rng.uniform(*config["beta_range"], s)
                w = c * (1 - beta) + v * beta
            elif family == "symmetric":
                if n != s: raise ValueError("Symmetric family requires equal widths")
                beta = config["symmetric_beta"]
                w = (1 - beta) * np.eye(n) + beta * np.ones((n, n)) / n
            elif family == "weak_coupling":
                if n != s or n < 3: raise ValueError("Weak coupling requires equal widths>=3")
                t = config["t_scale"] / n ** 3
                idx = np.arange(1, n + 1)
                w = t * (idx[:, None] + idx[None, :]) / n ** 2
                np.fill_diagonal(w, 0)
                np.fill_diagonal(w, 1 - w.sum(axis=0))
            else: raise ValueError(f"Unknown teacher family: {family}")
            weights.append(w)
        a = frozen[1].copy() if frozen is not None else (np.ones(architecture.hidden_widths[-1]) if family in ("weak_coupling", "symmetric") else rng.uniform(.5, 1.5, architecture.hidden_widths[-1]))
        a /= a.sum()
        diag = diagnose(weights, a)
        if (diag["kappa_actual"] <= config["kappa0_max"] and diag["mu_actual"] >= config["mu_min"]
                and all(np.min(w) > 0 for w in weights[1:])):
            diag.update(rejection_count=attempt, rejection_reasons=rejects, family=family)
            if frozen is not None:diag.update(frozen[2])
            return Teacher(weights, a, architecture.k), diag
        rejects.append({"kappa": diag["kappa_actual"], "mu": diag["mu_actual"]})
    raise NumericalFailure("instance_generation_exhausted", attempts=len(rejects))
