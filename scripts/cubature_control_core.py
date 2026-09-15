"""Low-dimensional control: real values only, deterministic angular moments.

This is NOT the paper's Hit-and-Run sampling implementation. It approximates
the same body moments by homogeneity and sphere quadrature, for width 3 and
two hidden layers only. No teacher parameters or analytic derivatives enter.
"""
import numpy as np
from scipy.special import roots_legendre
from aspire.recovery.subspace import recover_subspace
from aspire.oracles.interpolation import gradient
from aspire.oracles.suffix import SuffixOracle
from aspire.recovery.moments import moment_directions, normalize_directions
from aspire.recovery.final_layer import recover_final, simplex_projection
from aspire.recovery.sampling import radius_bounds
from aspire.status import NumericalFailure


def recover_cubature_control(real, *, architecture, public_bounds, config, order, rng):
    arch=architecture;ledger=real.ledger
    if arch.hidden_widths!=(3,3) or arch.k!=4:
        raise ValueError('Control restricted to width [3,3], k=4')
    result={'hidden_weights':[],'output_weights_raw':None,'output_weights_projected':None,
        'stages':[],'status':'running','failure_reason':None,'artifacts':{}}
    try:
        basis,diag=recover_subspace(real,arch,1,[],config,rng,ledger)
        def reduced(z):return real(basis@z)
        t,weights=roots_legendre(order);phi=np.arange(2*order)*np.pi/order
        omega=np.stack(np.broadcast_arrays(np.sqrt(1-t*t)[:,None]*np.cos(phi),
            np.sqrt(1-t*t)[:,None]*np.sin(phi),t[:,None]),axis=-1).reshape(-1,3)
        rotation=np.linalg.qr(np.random.default_rng(112358).normal(size=(3,3)))[0]
        omega=omega@rotation;weights=np.repeat(weights,2*order)/(4*order)
        with ledger.scope('membership'):
            values=np.array([reduced(point) for point in omega])
        inner,_=radius_bounds(arch,1,public_bounds)
        with ledger.scope('moment_gradient'):
            grads=np.array([gradient(reduced,point,degree=arch.Q,
                step=inner*config['recovery']['reduced_gradient_step_factor'],ledger=ledger) for point in omega],float)
        if np.any(values<=0):raise NumericalFailure('nonpositive_radial_value')
        radii=values**(-1/arch.Q);den=weights@radii**3
        sx=(omega.T*(weights*radii**5))@omega*(3/5)/den
        exponent=3+2*arch.Q-2
        sg=(grads.T*(weights*radii**exponent))@grads*(3/exponent)/den
        theta,directions,spectral=moment_directions(sx,sg,config['recovery']['moment_ridge'],config['recovery']['gap_floor'])
        result['hidden_weights'].append(normalize_directions(basis@directions,1))
        result['stages'].append({'layer':1,'status':'complete',**diag,**spectral,
            'integration_order':order,'angular_directions':len(omega),'rotation_seed':112358})
        result['artifacts'].update(basis=basis,omega=omega,angular_weights=weights,values=values,
            gradients=grads,Sx=sx,Sg=sg,eigenvalues=theta)
        suffix=SuffixOracle(real,arch,2,result['hidden_weights'],config['oracle'],ledger)
        w,a,diag,arrays=recover_final(suffix,arch,config,rng,ledger)
        result['hidden_weights'].append(w)
        result['output_weights_raw']=a;result['output_weights_projected']=simplex_projection(a)
        result['artifacts'].update(arrays)
        result['stages'].append({'layer':2,'status':'complete',**diag,'suffix_diagnostics':suffix.diagnostics})
        result['status']='complete'
    except NumericalFailure as error:
        result.update(status='failed',failure_reason=error.reason,failure_details=error.details)
    return result
