# Experiment: layerwise parameter recovery

The target is the homogeneous polynomial network

$$
h_1(x)=(W_1^\top x)^{\odot 4},\qquad
h_2(x)=(W_2^\top h_1(x))^{\odot 4},\qquad
f(x)=a^\top h_2(x).
$$

Here $W_1\in\mathbb R^{8\times3}$, $W_2\in\mathbb R^{3\times3}$, and $a\in\mathbb R^3$. The experiment uses a fixed target and noiseless real-valued function queries. All random seeds are explicit in `configs/experiment_k4.yaml`.

## First hidden layer

Column-space recovery evaluates first-layer gradients through polynomial interpolation of real function values. It obtains a basis $B$ for the estimated column space and works with the reduced function $g(z)=f(Bz)$. Independent Hit-and-Run chains provide endpoints in the reduced sublevel body. Each chain starts from the configured initial point and takes 32 transitions; the sample count is 16,777,216 and the batch size is 4,096.

The algorithm estimates position and gradient moments,

$$
S_z=\frac1N\sum_{r=1}^N z_rz_r^\top,\qquad
S_\nabla=\frac1N\sum_{r=1}^N\nabla g(z_r)\nabla g(z_r)^\top.
$$

The ASPIRE moment eigensystem gives hidden directions in the reduced space, which are mapped back with $B$ and normalized. The gradients are obtained from the real-value oracle by interpolation; their exact analytic values are not supplied to the learner.

## Second hidden layer

The suffix oracle is constructed from the recovered $\widehat W_1$ and real queries to $f$. Its Hessians are measured at the anchor $y_0=\mathbf1$ and at 12 probe points $y_i=\mathbf1+\tau u_i$, where $\tau=0.2$. The directions consist of four independent orthogonal bases. Six additional probes are reserved for evaluating the joint residual.

In exact suffix coordinates the Hessians have the common structure

$$
H(y)=12W_2\,\operatorname{diag}\!\left(a\odot(W_2^\top y)^{\odot2}\right)W_2^\top.
$$

For a positive-definite anchor $H_0$, whitening gives

$$
H_0=LL^\top,\qquad A_i=L^{-1}H_iL^{-\top}.
$$

For each of 64 random coefficient vectors, the code forms $H_\beta=\sum_i\beta_iH_i$ and solves the symmetric generalized eigenproblem

$$
H_\beta C=H_0C\Lambda,\qquad C^\top H_0C=I.
$$

Directions are recovered from $V=H_0C$. Following the paper, take coordinatewise absolute values and then normalize each column:

$$
\widehat w_{2,j}=\frac{|V_{:,j}|}{\mathbf 1^\top|V_{:,j}|}.
$$

For a nonzero direction this guarantees nonnegative entries and a unit column sum; it does not impose strict positivity on zero entries. The mixture is selected using the joint off-diagonal residual of measured training Hessians, with a spectral-gap tie break. Ground-truth parameter error does not enter this within-run mixture selection. Joint-diagonalization residuals refer to the spectral directions before normalization; the anchor reconstruction residual is evaluated again using the normalized weights. The output coefficients are subsequently fitted using the normalized hidden-layer features.

## Output coefficients

Draw $N_G=1,048,576$ input points $x_r\sim\mathcal N(0,I_8)$ and query $f(x_r)$. Holding both estimated hidden layers fixed, define

$$
\Phi_{rj}=\left[\widehat W_2^\top
      (\widehat W_1^\top x_r)^{\odot4}\right]_j^4,
\qquad
\widehat a=\mathop{\arg\min}_{b\in\mathbb R^3}\|\Phi b-f(X)\|_2^2.
$$

The solver uses scaled SVD least squares, without an intercept or regularization. The saved coefficients are the fitted coefficients; they are not projected onto a simplex.

## Parameter comparison

Evaluation aligns hidden-unit permutations sequentially and allows sign changes in the first layer, consistent with the even activation. The following quantities are reported:

$$
\|\widehat W_{1,\mathrm{aligned}}-W_1\|_2,\qquad
\|\widehat W_{2,\mathrm{aligned}}-W_2\|_2,\qquad
\|\widehat a_{\mathrm{aligned}}-a\|_1.
$$

`weights.json` and `weights.npz` retain ground truth, raw estimates, aligned estimates, and signed differences. `weights.csv` records every aligned coordinate separately. This alignment changes labels, not the network function.

## Other even activation exponents

For the activation study the same parameter matrices are used with $k=6$ and $k=8$. Replace fourth powers by $k$th powers in the network and feature formulas. The total polynomial degree is $Q=k^2$, and the last-suffix Hessian is

$$
H(y)=k(k-1)W_2\operatorname{diag}\!\left(a\odot(W_2^\top y)^{\odot(k-2)}\right)W_2^\top.
$$

First-layer directional derivatives therefore use $k^2+1$ interpolation nodes, while each Hessian directional derivative uses $k+1$ nodes. The positive-coordinate prefix inversion uses $k$th roots. The generalized eigenproblem, coordinatewise absolute-value normalization, and unregularized OLS retain the same form. Each exponent has its own first-layer estimate; the sample count, number of transitions, probe count, mixture count, Gaussian sample count, seeds, and tolerances are fixed.
