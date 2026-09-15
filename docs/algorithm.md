# Mathematical algorithm

The bias-free target is

$$
f(x)=a^T\sigma\!\left(W_2^T\sigma(W_1^Tx)\right),\qquad \sigma(t)=t^4.
$$

The main instance has $W_1\in\mathbb R^{8\times3}$, $W_2\in\mathbb R^{3\times3}$, and $a\in\mathbb R^3$. Its total degree is 16. Stored weights have input coordinates in rows and output units in columns.

## First layer

Original real-value queries are interpolated along directions to recover gradients and a column-space basis. Analytic target gradients are never supplied to the learner. Independent Hit-and-Run chains sample the reduced sublevel set, giving uncentered moments

$$
\widehat\Sigma_x=\frac1m\sum_i z_i z_i^T,\qquad
\widehat\Sigma_g=\frac1m\sum_i g_i g_i^T.
$$

For $\widehat\Sigma_x=L_xL_x^T$, the implementation diagonalizes $L_x^T\widehat\Sigma_gL_x$, then maps directions through $L_x^{-T}$ and the column-space basis. The best setting has $m=16{,}777{,}216$ endpoints and 32 steps each. Batch size is part of the fixed RNG layout.

## Measured suffix Hessians

The next stage uses the **estimated** first layer to construct a suffix oracle through prefix inversion and original real queries. It does not receive the true hidden-layer oracle. Prefix errors therefore perturb the expected simultaneous-diagonalization structure.

For an exact prefix, with $W=W_2$,

$$
H(y)=12W\operatorname{diag}\!\left(a\odot(W^Ty)^2\right)W^T.
$$

Use anchor $y_0=\mathbf1$, probes $y_i=\mathbf1+0.2u_i$ from random orthogonal bases, and interpolation radius 0.1. In three coordinates, each degree-four symmetric Hessian needs $6\times5=30$ real queries.

## Generalized eigendecomposition

With positive-definite anchor $H_0$, form a random combination and solve

$$
H_\beta=\sum_{i=1}^M\beta_iH_i,\qquad
H_\beta C=H_0C\Lambda,\qquad C^TH_0C=I.
$$

The implementation uses symmetric generalized `scipy.linalg.eigh`, projecting onto the resolved rank subspace when rectangular. Whitening diagnostics use Cholesky factors and triangular solves, without explicitly inverting the Cholesky factor.

The eigenvectors are dual directions. Recover weights using

$$
V=H_0C,\qquad \widehat W_{:j}=\frac{V_{:j}}{\mathbf1^TV_{:j}},\qquad
\widehat a_H=\frac1{12}\operatorname{diag}\!\left(\widehat W^\dagger H_0\widehat W^{\dagger T}\right).
$$

Using $C$ directly as weights is incorrect. The main solver evaluates 64 normalized Gaussian combinations, minimizes training off-diagonal residual relative to the non-isotropic signal, and breaks residual ties within `1e-12` using the larger absolute eigengap.

Unresolved rank, nonpositive anchor rank, and unresolved eigengaps are explicit failures. Approximate-prefix probe matrices need not all be positive definite. Held-out residuals and whitened commutators are diagnostics only. The separate absolute-normalization control applies elementwise absolute values to the same raw directions before normalizing columns.

## Gaussian output regression

Draw $x_i\sim N(0,I_8)$, query $y_i=f(x_i)$, and freeze both hidden estimates. Form

$$
\Phi_{ij}=\left[\sigma\!\left(\widehat W_2^T\sigma(\widehat W_1^Tx_i)\right)\right]_j,\qquad
\widehat a=\arg\min_b\|\Phi b-y\|_2^2.
$$

There is no intercept, ridge, positivity constraint, or simplex projection. Column and target scaling improve numerical conditioning without changing the unweighted least-squares objective. The SVD solution is checked against `numpy.linalg.lstsq`.

## Evaluation

Hidden permutations and admissible first-layer signs are aligned before computing errors. The numerical target is

$$
\max\left\{\|\widehat W_1-W_1\|_2,\|\widehat W_2-W_2\|_2,\|\widehat a-a\|_1\right\}\le0.1.
$$

The norm $\|\widehat a\|_1$ is reported separately. Prediction evaluation uses directions in the union of true and estimated first-layer spans; that construction belongs to the evaluator alone. The successful setting is a local empirical result, not a theorem certificate.
