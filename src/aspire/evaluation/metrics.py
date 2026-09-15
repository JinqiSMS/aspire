"""Parameter error and tail-aware Gaussian risk; spec §11."""
import math
import numpy as np
from scipy.special import logsumexp, gammaln
from .alignment import align
from ..teacher import Teacher

def log_square_mean(values):
    absolute=np.abs(np.asarray(values,float))
    if np.any(~np.isfinite(absolute)): return float("nan")
    with np.errstate(divide="ignore"):
        return float(logsumexp(2*np.log(absolute))-np.log(absolute.size))

def exp_or_none(x): return float(np.exp(x)) if np.isfinite(x) and x<709 else (0. if x==-np.inf else None)

def risk_batch(prediction, target):
    error=np.asarray(prediction)-np.asarray(target)
    logerr,logtarget=log_square_mean(error),log_square_mean(target)
    with np.errstate(divide="ignore",invalid="ignore"):
        losses=2*np.log(abs(error))
    top=max(1,len(error)//100)
    total=logsumexp(losses)
    return {"mse":exp_or_none(logerr),"log_mse":logerr,"nmse":exp_or_none(logerr-logtarget),
            "risk_unavailable_reason":'nonfinite_prediction_or_target' if np.isnan(logerr) or np.isnan(logtarget) else ('zero_target_energy' if logtarget==-np.inf else None),
            "max_abs_target":float(np.max(abs(target))), "max_abs_error":float(np.max(abs(error))),
            "top_one_percent_loss_share":float(np.exp(logsumexp(np.sort(losses)[-top:])-total)) if np.isfinite(total) else None}

def parameter_metrics(teacher,result,delta):
    estimates=result["hidden_weights"]
    aligned,a,orders=align(teacher.weights,estimates,result.get("output_weights_raw"))
    errors=[float(np.linalg.norm(w-t,2)) for w,t in zip(aligned,teacher.weights)]
    fro=[float(np.linalg.norm(w-t)) for w,t in zip(aligned,teacher.weights)]
    output=float(np.linalg.norm(a-teacher.a,1)) if a is not None else None
    full=len(estimates)==len(teacher.weights) and a is not None
    maximum=max(errors) if full else None
    span=None
    if estimates:
        b=np.linalg.qr(estimates[0])[0]; bt=np.linalg.qr(teacher.weights[0])[0]
        # Low-rank principal-angle identity avoids d by d projection matrices.
        cosine=np.linalg.svd(bt.T@b,compute_uv=False)[-1]
        span=float(np.sqrt(max(0.,1-min(1.,cosine)**2)))
    return {"per_layer_errors":errors,"per_layer_frobenius_errors":fro,"output_l1_error":output,
        "max_weight_operator_error":maximum,"first_subspace_error":span,
        "alignment_method":"sequential_hungarian","alignment_permutations":orders,
        "parameter_success":bool(full and result["status"]=="complete" and maximum<=delta and output<=delta)}

def evaluate_predictor(teacher,predictor,architecture,settings,rng,ledger,student_weights=None):
    batches=[]
    for _ in range(settings["gaussian_batches"]):
        x=rng.normal(size=(settings["gaussian_test_points"],architecture.d))
        y=teacher.forward(x)
        ledger.tick("n_real_calls_total",len(x)); ledger.tick("n_real_calls_evaluation",len(x))
        batches.append(risk_batch(predictor(x),y))
    result={"gaussian_batches":batches,
            "empirical_gaussian_nmse":float(np.median([b["nmse"] for b in batches if b["nmse"] is not None])) if any(b["nmse"] is not None for b in batches) else None,
            "empirical_gaussian_mse":float(np.median([b["mse"] for b in batches if b["mse"] is not None])) if any(b["mse"] is not None for b in batches) else None}
    if student_weights is not None and settings["angular_risk_directions"]>0:
        u,s,_=np.linalg.svd(np.concatenate([teacher.weights[0],student_weights[0]],axis=1),full_matrices=False)
        rank=int(np.count_nonzero(s>settings["rank_tolerance"]*s[0])); u=u[:,:rank]
        radial=architecture.Q*np.log(2)+gammaln(architecture.Q+rank/2)-gammaln(rank/2)
        angular=[]
        for _ in range(settings["angular_risk_batches"]):
            omega=rng.normal(size=(settings["angular_risk_directions"],rank)); omega/=np.linalg.norm(omega,axis=1)[:,None]
            x=omega@u.T; y=teacher.forward(x)
            ledger.tick("n_real_calls_total",len(x));ledger.tick("n_real_calls_evaluation",len(x))
            batch=risk_batch(predictor(x),y); batch["log_gaussian_mse"]=batch["log_mse"]+radial
            angular.append(batch)
        result.update(angular_batches=angular,angular_gaussian_nmse=float(np.median([b["nmse"] for b in angular if b["nmse"] is not None])) if any(b["nmse"] is not None for b in angular) else None,
            angular_gaussian_log_mse=float(np.median([b["log_gaussian_mse"] for b in angular])),
            angular_rank=rank,angular_rank_singular_values=s.tolist(),angular_rank_tolerance=settings["rank_tolerance"])
    return result

def evaluate_recovery(teacher,result,architecture,settings,rng,ledger):
    metrics=parameter_metrics(teacher,result,settings["parameter_delta"])
    if result["output_weights_raw"] is not None:
        student=Teacher(result["hidden_weights"],result["output_weights_raw"],architecture.k)
        state=rng.bit_generator.state
        metrics.update(evaluate_predictor(teacher,student.forward,architecture,settings,rng,ledger,student.weights))
        projected=Teacher(student.weights,result["output_weights_projected"],architecture.k)
        rng.bit_generator.state=state
        metrics["projected_output_metrics"]=evaluate_predictor(teacher,projected.forward,architecture,settings,rng,ledger,student.weights)
        _,aligned,_=align(teacher.weights,student.weights,result["output_weights_projected"])
        metrics["projected_output_l1_error"]=float(np.linalg.norm(aligned-teacher.a,1))
    else: metrics["prediction_metrics_unavailable_reason"]="incomplete_parameter_recovery"
    return metrics
