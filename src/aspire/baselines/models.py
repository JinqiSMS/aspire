"""Precisely specified supervised, homogeneous KRR, and finite empirical NTK."""
import math
import time
import numpy as np
from ..status import NumericalFailure

def check_deadline(cfg):
    if time.perf_counter()>cfg.get('_deadline',float('inf')):raise NumericalFailure('wall_time_exhausted')

def memory_guard(n, p, cfg, matrices=4):
    estimated=8*(matrices*n*n+2*n*p)
    if n>cfg['max_train_points'] or estimated>cfg['max_working_memory_mb']*1024**2:
        raise NumericalFailure('baseline_resource_limit',estimated_bytes=estimated,points=n,features=p)
    return estimated

def krr_fit(x,y,xv,yv,architecture,cfg,rng):
    estimated=memory_guard(len(x),0,cfg)
    q,d=architecture.Q,architecture.d
    with np.errstate(over='ignore',invalid='ignore'):
        gram=(x@x.T/d)**q; validation=(xv@x.T/d)**q
    if not np.isfinite(gram).all():raise NumericalFailure('nonfinite_kernel')
    scale=float(np.trace(gram)/len(x)) if cfg['kernel_trace_scaling'] else 1.
    if not scale>0:raise NumericalFailure('invalid_kernel_scale')
    gram/=scale; validation/=scale
    scores=[];solutions=[]
    yscale=max(np.max(abs(y)),np.finfo(float).tiny)
    for reg in cfg['lambdas']:
        check_deadline(cfg)
        alpha=np.linalg.solve(gram+len(x)*reg*np.eye(len(x)),y/yscale)
        score=float(np.mean((validation@alpha-yv/yscale)**2))
        scores.append(score);solutions.append(alpha)
    choice=int(np.argmin(scores));alpha=solutions[choice]*yscale
    def predict(points):
        # Bound prediction workspace even for large Gaussian test batches.
        return np.concatenate([((block@x.T/d)**q/scale)@alpha for block in np.array_split(points,max(1,math.ceil(len(points)/256)))])
    return predict,{'selected_lambda':cfg['lambdas'][choice],'validation_scores':scores,'kernel_scale':scale,
        'regularization_convention':'K_scaled + n*lambda*I','estimated_working_bytes':estimated}, {'train_inputs':x,'alpha':alpha},None

def supervised_fit(x,y,xv,yv,architecture,cfg,rng):
    import torch
    torch.set_num_threads(1)
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw=torch.nn.ParameterList([torch.nn.Parameter(torch.randn(n,s,dtype=torch.float64)/np.sqrt(n)) for n,s in zip(architecture.widths,architecture.widths[1:])])
            self.output=torch.nn.Parameter(torch.zeros(architecture.hidden_widths[-1],dtype=torch.float64))
        def normalized(self):
            return [self.raw[0]/torch.linalg.vector_norm(self.raw[0],dim=0)]+[torch.softmax(w,dim=0) for w in self.raw[1:]],torch.softmax(self.output,dim=0)
        def forward(self,points):
            ws,a=self.normalized()
            for w in ws:points=(points@w)**architecture.k
            return points@a
    tx,ty,tv,tvy=[torch.tensor(v,dtype=torch.float64) for v in (x,y,xv,yv)]
    scale=max(float(np.max(abs(y))),np.finfo(float).tiny)
    best=None;scores=[];histories=[];selection=[]
    for lr in cfg['learning_rates']:
        for restart in range(cfg['restarts']):
            seed=int(rng.integers(2**31));torch.manual_seed(seed)
            model=Student();optimizer=torch.optim.Adam(model.parameters(),lr=lr)
            history=[];local_best=float('inf');state=None
            for epoch in range(cfg['epochs']):
                check_deadline(cfg)
                order=torch.randperm(len(x))
                stable=True
                for indices in order.split(cfg['batch_size']):
                    optimizer.zero_grad()
                    loss=torch.mean(((model(tx[indices])-ty[indices])/scale)**2)
                    if not torch.isfinite(loss):stable=False;break
                    loss.backward();optimizer.step()
                with torch.no_grad():val=float(torch.mean(((model(tv)-tvy)/scale)**2))
                history.append(val)
                if stable and np.isfinite(val) and val<local_best:
                    local_best=val;state={k:v.detach().clone() for k,v in model.state_dict().items()}
                if not stable:break
            scores.append(local_best);histories.append(history);selection.append({'learning_rate':lr,'restart':restart,'seed':seed})
            if state is not None and (best is None or local_best<best[0]):
                best=(local_best,state,selection[-1],model)
    if best is None:raise NumericalFailure('baseline_optimization_failed')
    model=best[3];model.load_state_dict(best[1]);model.eval()
    with torch.no_grad():ws,a=model.normalized();weights=[w.numpy().copy() for w in ws];coef=a.numpy().copy()
    def predict(points):
        with torch.no_grad():return model(torch.tensor(points,dtype=torch.float64)).numpy()
    artifacts={f'W{i+1}':w for i,w in enumerate(weights)};artifacts['a']=coef
    maxlen=max(map(len,histories));hist=np.full((len(histories),maxlen),np.nan)
    for i,h in enumerate(histories):hist[i,:len(h)]=h
    artifacts['validation_history']=hist
    return predict,{'training_scale':scale,'validation_scores':scores,'selected':best[2],'all_candidates':selection},artifacts,(weights,coef)

def empirical_ntk_fit(x,y,xv,yv,architecture,cfg,rng):
    import torch
    torch.set_num_threads(1);torch.manual_seed(int(rng.integers(2**31)))
    width=cfg['ntk_width'];sizes=[architecture.d]+[width]*(architecture.L-1)
    params=[torch.randn(n,s,dtype=torch.float64,requires_grad=True) for n,s in zip(sizes,sizes[1:])]
    output=torch.randn(width,dtype=torch.float64,requires_grad=True)
    params.append(output);count=sum(v.numel() for v in params)
    estimated=memory_guard(len(x)+len(xv),count,cfg)
    ck=math.sqrt(math.prod(range(1,2*architecture.k,2)))
    def forward(row):
        h=row
        for n,w in zip(sizes,params[:-1]):h=(h@w/math.sqrt(n))**architecture.k/ck
        return h@output/math.sqrt(width)
    def features(points):
        features_,initial=[],[]
        for row in points:
            check_deadline(cfg)
            f=forward(torch.tensor(row,dtype=torch.float64))
            grad=torch.autograd.grad(f,params)
            features_.append(torch.cat([v.reshape(-1) for v in grad]).detach().numpy());initial.append(float(f.detach()))
        phi=np.asarray(features_);init=np.asarray(initial)
        if not np.isfinite(phi).all():raise NumericalFailure('nonfinite_kernel')
        return phi,init
    phi,initial=features(x);phiv,initialv=features(xv)
    gram=phi@phi.T;validation=phiv@phi.T
    scale=float(np.trace(gram)/len(x)) if cfg['kernel_trace_scaling'] else 1.
    if not scale>0:raise NumericalFailure('invalid_kernel_scale')
    gram/=scale;validation/=scale
    scores=[];alphas=[]
    target_scale=max(float(np.max(abs(y))),np.finfo(float).tiny)
    for reg in cfg['lambdas']:
        alpha=np.linalg.solve(gram+len(x)*reg*np.eye(len(x)),(y-initial)/target_scale)
        score=float(np.mean((initialv/target_scale+validation@alpha-yv/target_scale)**2))
        scores.append(score);alphas.append(alpha*target_scale)
    choice=int(np.argmin(scores));alpha=alphas[choice]
    feature_weight=phi.T@alpha/scale
    def predict(points):
        predictions=[]
        for block in np.array_split(points,max(1,math.ceil(len(points)/128))):
            f,init=features(block);predictions.append(init+f@feature_weight)
        return np.concatenate(predictions)
    artifacts={'train_inputs':x,'alpha':alpha,'feature_weight':feature_weight}
    artifacts.update({f'initial_parameter_{i}':v.detach().numpy() for i,v in enumerate(params)})
    return predict,{'selected_lambda':cfg['lambdas'][choice],'validation_scores':scores,'kernel_scale':scale,
        'feature_count':count,'estimated_working_bytes':estimated,'bias_convention':'initial_function_plus_residual_kernel_regression',
        'width':width},artifacts,None

FITTERS={'supervised':supervised_fit,'polynomial_krr':krr_fit,'empirical_ntk':empirical_ntk_fit}
