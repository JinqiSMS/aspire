"""S2 alg:aspire / final-hessian-recovery. Strict wrapper has no truth inputs."""
import numpy as np
from ..numeric import matmul
from ..oracles.suffix import SuffixOracle
from ..oracles.interpolation import gradient, gradient_batch
from ..status import NumericalFailure
from .subspace import recover_subspace
from .sampling import sample_body, radius_bounds
from .moments import raw_moments, moment_directions, normalize_directions, moment_ess
from .final_layer import recover_final, simplex_projection

def recover_network(real_oracle, *, architecture, public_bounds, config, rng,
                    checkpoint=None, restored=None, stop_after_layer=None):
    """Formal learner: ONLY real values, public bounds and recovered prefix."""
    if config["oracle"]["mode"] != "strict_real": raise ValueError("Use explicit debug driver for non-strict runs")
    def builder(layer,prefix):
        return SuffixOracle(real_oracle, architecture, layer, prefix, config["oracle"], real_oracle.ledger)
    if stop_after_layer is not None and not 1 <= stop_after_layer <= architecture.L - 2:
        raise ValueError("stop_after_layer must identify a non-final hidden layer")
    return _recover(real_oracle, architecture, public_bounds, config, rng, builder,
                    checkpoint, restored, stop_after_layer)

def _recover(real, arch, bounds, cfg, rng, suffix_builder, checkpoint=None, restored=None,
             stop_after_layer=None):
    """Internal engine; debug adapters are supplied only by diagnostics driver."""
    ledger = real.ledger
    weights = [np.array(w,float) for w in (restored or {}).get("weights",[])]
    stages = list((restored or {}).get("stages",[]))
    artifacts = {}
    result = {"hidden_weights": weights, "output_weights_raw": None, "output_weights_projected": None,
              "stages": stages, "status": "running", "failure_reason": None,
              "oracle_mode": cfg["oracle"]["mode"], "sampler_mode":cfg["sampler"]["mode"],
              "guarantee_status": "empirical_only", "artifacts": artifacts}
    layer = len(weights)+1
    before=ledger.snapshot()
    if stop_after_layer is not None and len(weights) >= stop_after_layer:
        result.update(status="prefix_complete", query_counts=ledger.snapshot())
        return result
    try:
        if arch.Q > cfg["oracle"]["max_degree_strict"] and cfg["oracle"]["mode"]=="strict_real":
            raise NumericalFailure("degree_limit_exceeded", degree=arch.Q)
        for layer in range(len(weights)+1, arch.L-1):
            before = ledger.snapshot()
            basis, diag = recover_subspace(real, arch, layer, weights, cfg, rng, ledger)
            suffix = suffix_builder(layer, weights)
            def reduced(z): return suffix(matmul(basis, z))
            batch_reduced=None
            # Layer one needs no prefix inversion; batch queries remain queries
            # to the original real value oracle, in the estimated basis.
            if layer==1 and cfg["oracle"]["backend"]=="float64" and cfg["sampler"].get("batch_size",0):
                def batch_reduced(z):
                    ledger.tick("n_suffix_calls",len(z))
                    suffix.diagnostics["queries"]+=len(z)
                    return real.batch(np.asarray(z)@basis.T)
            inner, outer = radius_bounds(arch,layer,bounds)
            count = cfg["recovery"]["moment_samples_by_layer"][layer-1]
            z, ids, sampling_diag = sample_body(reduced, arch.widths[layer], count,
                 config=cfg["sampler"], xi=cfg["sampler"]["xi_by_layer"][layer-1], outer_radius=outer, rng=rng, ledger=ledger,batch_value=batch_reduced)
            artifacts.update({f"layer{layer}_basis":basis,f"layer{layer}_samples":z,f"layer{layer}_chain_ids":ids})
            dps = cfg["oracle"]["dps"] if cfg["oracle"]["backend"]=="mpmath" else 0
            with ledger.scope("moment_gradient"):
                if batch_reduced is not None:
                    g=gradient_batch(batch_reduced,z,degree=arch.suffix_degree(layer),
                        step=inner*cfg["recovery"]["reduced_gradient_step_factor"],ledger=ledger,
                        batch_size=cfg["sampler"]["batch_size"])
                else:
                    g = np.array([gradient(reduced,p,degree=arch.suffix_degree(layer),
                        step=inner*cfg["recovery"]["reduced_gradient_step_factor"], dps=dps,ledger=ledger) for p in z],float)
            sx,sg = raw_moments(z,g)
            artifacts.update({f"layer{layer}_gradients":g,f"layer{layer}_Sx":sx,f"layer{layer}_Sg":sg})
            theta,v,spectral = moment_directions(sx,sg,cfg["recovery"]["moment_ridge"],cfg["recovery"]["gap_floor"])
            w = normalize_directions(basis@v,layer)
            weights.append(w)
            stage = {"layer":layer,"status":"complete",**diag,**sampling_diag,**spectral,
                     **moment_ess(z,g,ids,cfg["sampler"]["mode"]), "inner_radius":inner,
                     "radius_source":"public_bounds_conditional_on_accurate_basis", "xi_source":cfg["sampler"]["xi_source"],
                     "suffix_diagnostics":getattr(suffix,"diagnostics",{}),
                     "query_counts":{k:v-before.get(k,0) for k,v in ledger.snapshot().items()}}
            stages.append(stage)
            artifacts.update({f"layer{layer}_"+name:value for name,value in
                             {"basis":basis,"samples":z,"gradients":g,"chain_ids":ids,"Sx":sx,"Sg":sg,"eigenvalues":theta,"directions":v}.items()})
            if checkpoint: checkpoint(weights,stages,rng.bit_generator.state,ledger.snapshot(),artifacts)
            if layer == stop_after_layer:
                result.update(status="prefix_complete", query_counts=ledger.snapshot())
                return result
        layer=arch.L-1
        suffix=suffix_builder(layer,weights)
        before=ledger.snapshot()
        w,a,diag,final_artifacts=recover_final(suffix,arch,cfg,rng,ledger)
        weights.append(w)
        result["output_weights_raw"],result["output_weights_projected"]=a,simplex_projection(a)
        artifacts.update(final_artifacts)
        stages.append({"layer":layer,"status":"complete",**diag,"suffix_diagnostics":getattr(suffix,"diagnostics",{}),
                       "query_counts":{k:v-before.get(k,0) for k,v in ledger.snapshot().items()}})
        result["status"]="complete"
    except NumericalFailure as error:
        result.update(status="failed",failure_reason=error.reason,failure_details=error.details)
        stages.append({"layer":layer,"status":"failed","failure_reason":error.reason,"details":error.details,
                       "query_counts":{k:v-before.get(k,0) for k,v in ledger.snapshot().items()}})
    except np.linalg.LinAlgError as error:
        result.update(status="failed",failure_reason="linear_algebra_failure",failure_details=str(error))
        stages.append({"layer":layer,"status":"failed","failure_reason":"linear_algebra_failure"})
    result["query_counts"]=ledger.snapshot()
    return result
