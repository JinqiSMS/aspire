"""Spec §11.1: consistent sequential relabeling; signs only at first layer."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def align(true_weights, estimates, output=None):
    est=[w.copy() for w in estimates]; a=None if output is None else output.copy()
    permutations=[]
    for layer,(truth,w) in enumerate(zip(true_weights,est)):
        minus=np.sum((truth[:,:,None]-w[:,None,:])**2,axis=0)
        plus=np.sum((truth[:,:,None]+w[:,None,:])**2,axis=0)
        _,order=linear_sum_assignment(np.minimum(minus,plus) if layer==0 else minus)
        w=w[:,order]
        if layer==0: w*=np.where(np.sum(truth*w,axis=0)<0,-1.,1.)
        est[layer]=w
        if layer+1<len(est): est[layer+1]=est[layer+1][order,:]
        elif a is not None: a=a[order]
        permutations.append(order.tolist())
    return est,a,permutations

