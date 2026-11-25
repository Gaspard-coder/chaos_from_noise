#import "@preview/diverential:0.3.0": *

#set page(
  paper: "a4",
  margin: 2.5cm,
)

#set text(
  font: "New Computer Modern",
  size: 12pt,
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.65em,
  first-line-indent: 0pt,
)

#set page(numbering: "1")

#set heading(numbering: "1.")

#let nb(body) = block(
  fill: rgb("#eaf3ff"),
  stroke: (paint: rgb("#4a90e2"), thickness: 1.2pt),
  radius: 6pt,
  inset: 12pt,
  width: 100%,
  [
    #text(weight: "bold")[NB :] #body
  ],
)

#let rslt(body) = block(
  fill: rgb("#fff8af"),
  stroke: (paint: rgb("#ffdc00"), thickness: 1.2pt),
  radius: 6pt,
  inset: 12pt,
  width: 100%,
  [
    #text(weight: "bold")[Result :] #body
  ],
)

#let impt(body) = block(
  fill: rgb("#fc9494"),
  stroke: (paint: rgb("#ff1900"), thickness: 1.2pt),
  radius: 6pt,
  inset: 12pt,
  width: 100%,
  [
    #text(weight: "bold")[Important :] #body
  ],
)




#align(right)[
  #datetime.today().display("Last version : [day]/[month]/[year]")
]
#align(center)[
  #text(size: 19pt, weight: "bold")[
    Computing the Lyapunov Exponent: \
    the Jacobian approach
  ]
]

#outline()

= Origin
This short report aims to explain and explore different approach to compute the Lyapunov exponent using the well-knownn Jacobian matrix as well as the Oseledet's theorem. It heavily relies on the work of Ahmed BenSaïda with its very interesting paper : #link("https://aijcr.thebrpi.org/journals/Vol_2_No_8_August_2012/6.pdf")[#underline(text(fill: blue)[Are Financial Markets Stochastic: A Test for Noisy Chaos])].

The idea is not to be as extensive as BenSaïda's work but to see how his approach can be customed to detect chaos.

= The method

Let's take a given timeseries ${x_t}^T#sub[t=1]$. The first step and in fine the core of this approach is to find the chaotic mapping $f$ to deduce $x_t$ from a sub group of its predecessors ${x#sub[t-L],x#sub[t-2L],...,x#sub[t-mL]}$:

$ x_t = f(x#sub[t-L],x#sub[t-2L],...,x#sub[t-mL]) + epsilon_t, $
where $epsilon_t$ is added white noise.

#nb[It introduces two parameters: $m$ the embedding time delay and $L$ the lag or embedding dimension. It is reminiscent of permutation entropy sampling's parameters but don't have to be confused]

Given this mapping, we denote J the Jacobian matrix defined as :

$
  J_t = mat(
    dvp(f, x#sub[t-L]), dvp(f, x#sub[t-2L]), ..., ..., dvp(f, x#sub[t-mL]);
    0, 1, ..., 0, 0;
    dots.v, dots.v, dots.down, dots.v, dots.v;
    0, 0, ..., 1, 0;
  ).
$
We then note T the Jacobian product of the M first Jacobian matrix
$
  T_M = product_(t=1)^M J_t.
$

Then, we introduce $U_0 = mat(1; 0; dots.v; 0)$ an unitary vector leading to
$
  V_M = (T_M U_0)'(T_M U_0),
$
where the (.)' is the matrix transpose operator.

Given the fact that $V_M$ is diagonalizable, we note $v_1 = max("Sp("V_M")")$ the largest eigenvalue of $V_M$.

Finally, the #link("https://www.people.maths.bris.ac.uk/~macpd/ads/oseledets.pdf")[#underline(text(fill: blue)[Oseledet's theorem])] states that
$
  lambda = lim_(M arrow.r infinity) 1/(2M) ln(v_1).
$

Litterature suggests that the approxmiation
$
  hat(lambda) = 1/(2M) ln(v_1),
$
is raisonnable for $T#super[2/3] lt.eq M lt.eq T$.

#nb[Alternative approachs and extensive explanations can be found in BenSaïda's work, we retain here the most efficient method according to his study.]

= Finding the best chaotic mapping
The main challenge here is to find this chaotic mapping $f$. Indeed, "we need to
approximate the unknown chaotic map with a known function that can _learn_ the process by reading the relation $y
= f(x)$".

Multiple candidate approach exist such as _spline_, _nearest neighbors_ or _neural network_, this last option appearing to be the most promising according to the litterature.

BenSaïda showed that with a neural network approach, the approximation problem becomes
$
  hat(x_t) approx alpha_0 + sum_(j=1)^q alpha_j Psi(beta_(0,j) + sum_(i=1)^m beta_(i,j) x_(t-i L)) + epsilon_t
$
where:
- q is the number of neurons of the network
- $A = {alpha_j}$ are inter-layers connections
- $alpha_0$ is the network threshold
- $B = {beta_(i,j)}$ are the connection weights
- $epsilon_t$ is the added white noise
- $Psi$ is the activation function

As the logistic function $Psi = 1/(1+e^(-u))$ is usually used, the hyperbolic tangent $Psi = tanh$ is more efficient for this type of application.

So, the reconstruction problem becomes
$
  "Find the best set of parameters" theta = (m,L,q,A,B) "so the reconstructed timeseries"
  \ hat(x_t) approx alpha_0 + sum_(j=1)^q alpha_j tanh(beta_(0,j) + sum_(i=1)^m beta_(i,j) x_(t-i L)) + epsilon_t \
  "fits the most with the initial one"
$

== Non linear square (NLS) approach
This is the approach chosent by BenSaïda. The objective function to minimize is the Sum of Squared Errors (SSE):
$
  S(theta) = sum_(t=m L + 1)^T r_t (theta)^2
$
where $r_t (theta) = x_t - hat(x)_t (theta)$ represents the residual at time $t$.

The LM algorithm relies on the gradient and the Hessian of the objective function. The gradient of $S(theta)$ is given by:

$nabla S(theta) = - J_theta^T r(theta),$

#impt[Do not confuse $J_theta$ with the state Jacobian $J_t$ defined earlier. Here, $J_theta$ is the Jacobian matrix of the residuals with respect to the network parameters $theta$, not the lagged variables.]

Instead of computing the exact and computationally expensive second derivative (the Hessian matrix $H$), the LM algorithm exploits the "sum of squares" structure of the objective function to approximate the Hessian using the Gauss-Newton method: $H approx J_theta^T J_theta$.

The parameter update step $Delta theta$ in the Levenberg-Marquardt algorithm is determined by solving the damped normal equations:

$(J_theta^T J_theta + mu I) Delta theta = - J_theta^T r(theta),$

where $I$ is the identity matrix and $mu > 0$ is a damping factor. When $mu$ is large, the algorithm behaves like gradient descent (taking small, robust steps). As the solution approaches the minimum, $mu$ decreases, and the algorithm behaves like the Gauss-Newton method, offering fast quadratic convergence. This explicit utilization of the problem's squared-error structure makes NLS highly efficient for chaotic signal reconstruction.

=== Dynamically Adapting Parameters per IMF
First IMFs = "fast", highly dynamic signals => needs big set of parameters to handle it
Last IMFs = "slow", smoother ones => nedds a set of parameters closer to (1,1,1) to not over-estimate the dynamic => Might need decimation

Four steps to adapt the parameters to the signal:

1. Mesure the dominant period $T_k$ and decimate adaptively: Estimate $T_k$ (in samples) from the peak of the IMF's Power Spectral Density or its mean zero-crossing period. To prevent phase-space collapse on slow modes (`IMF_6–IMF_10`), target a consistent 15–25 samples per cycle:
$ D_k = max(1, floor.l T_k/20 floor.r) $

Apply `decimate=D_k` (maybe anti-aliasing filter) and verify that the decimated length $N_("eff") = N / D_k$ still contains at least 20–30 full cycles (and >=500 samples). If `IMF_9` or `IMF_10` has fewer than ~10 cycles in your 10,000-sample record, exclude it—there is not enough trajectory recurrence to estimate $lambda$.

2. Set the lag $L$ from the decimated IMF's correlation time
Compute the first local minimum of Average Mutual Information (AMI) (or the first $1/e$ decay / zero-crossing of the autocorrelation function) on the decimated IMF. With ~20 samples per cycle, this naturally lands around $tau*L .approx 4"-"6$ decimated samples (roughly $T_k / 4$), unfolding the trajectory away from the main diagonal.

3. Estimate embedding dimension (m) per IMF
Use False Nearest Neighbors (FNN) or Cao's method on the decimated IMF to bracket a candidate range $[m_("min"), m_("max")]$ (typically $m in [2, 6]$ for narrowband IMFs).

4. Select (L, m, q) via out-of-sample penalized error (BIC / GCV)
Split each decimated IMF chronologically (e.g., 70% train, 30% validation—never random shuffle) and evaluate your candidate (L, m, q) grid on validation nmse. Select the configuration that minimizes the Bayesian Information Criterion (BIC) rather than raw training error:
$"BIC" = N_("val") ln("NMSE"_("val")) + p ln(N_("val"))$

where $p$ is the number of free parameters in your Jacobian/network model (for a single-hidden-layer network with $m$ inputs and $q$ hidden units, $p = q(m + 2) + 1$). This prevents both underfitting (q = 1, which caused your IMF_8–IMF_10 explosions) and overfitting to noise.

5. Convert $lambda_("orb")$ back to common physical units
Because each IMF uses its own decimate factor $D_k$, raw per-step exponents are not comparable across IMFs. Rescale every result to physical inverse time before comparing:$$\lambda_{\text{phys}, k} = \frac{\lambda_{\text{orb}, k}}{D_k \cdot \Delta t}$$

=== Selecting the largest _trustworthy_ Lyapunov exponent

Never take $max_k(lambda_k)$ over raw outputs. Instead, pass all 10 IMFs through four sequential trust gates, and only take the maximum among the survivors:
- Pre-Gate — Per-IMF Timescale & Recurrence Adequacy: Estimate each IMF's dominant period $T_k$ (from its PSD peak or mean zero-crossing period) and set an adaptive decimation factor $D_k$ so every mode has ~15–25 samples per cycle.Set the embedding lag $L$ from the first local minimum of Average Mutual Information (AMI) on the decimated series.Reject any IMF whose decimated length $N_("eff") = N / D_k$ has fewer than 500 samples or fewer than 20 full cycles (e.g., ultra-low-frequency trend IMFs that never recur in phase space).
- Gate 1 — Out-of-Sample Predictive Skill (NMSE):Fit $(L, m, q)$ on the first 70% of the decimated IMF and evaluate on the held-out final 30% (chronological split, never shuffled), using validation BIC to penalize excess parameters.Reject any IMF with $"NMSE"_("val") > 0.10$ (prefer $< 0.05$), since an inaccurate map cannot yield reliable local Jacobians.
- Gate 2 — Sampling-Rate Plausibility:Reject any estimate with $lambda_("orb", k)} >= 0.5"-"1.0$ per decimated sample. Divergence faster than a single sampling interval reflects an ill-conditioned Jacobian rather than resolved dynamics. #highlight[To be discussed]
- Gate 3 — Multi-Start & Capacity-Plateau Stability: Require tight convergence across polished random initializations ($"spread" < 0.25\, |lambda_("orb", k)|$).Verify that $lambda_("orb", k)$ sits on a flat plateau across neighboring hidden-unit counts $q^* plus.minus 1, 2$ rather than jumping erratically with network capacity.
- Gate 4 — Cross-Dimension ($m$) Invariance (Gençay & Dechert Manifold Check): Sweep the embedding dimension $m$ from $m_(min)$ to $m_(max)$ (e.g., $m = 2 .dots 6$). Once $m$ reaches the unfolding dimension (estimated via False Nearest Neighbors or Cao's method) and $"NMSE"_("val")$ has saturated, $lambda_("orb", k)$ must remain stable as $m$ increases further.Reject any estimate where $lambda_("orb", k)$ jumps upward at larger $m$ without improving $"NMSE"_("val")$—this is the classic signature of a spurious exponent emerging in the $m - n$ directions transverse to an $n$-dimensional manifold.
- Gate 5 — Geometric Cross-Check on the Attractor (Eckmann–Ruelle / Trajectory Divergence): Compute a second, non-Jacobian estimate of $lambda_1$ directly from the spreading of nearby trajectory vectors on the reconstructed attractor (Eckmann–Ruelle / Rosenstein / Kantz algorithm) using the same $(L^*, m^*)$.Because trajectory divergence is restricted to tangent vectors along the data manifold whereas an $m$-dimensional neural network differentiates across all of $R^m$, reject any IMF where the Jacobian $lambda_("orb", k)$ substantially exceeds the trajectory-divergence rate (and for smooth continuous-like IMFs, optionally verify Parlitz's time-reversal sign flip $lambda(-t) approx -lambda(t)$).
- Gate 6 — Nonlinearity Confirmation via IAAFT Surrogates:Generate $B = 19–39$ IAAFT surrogates of the decimated IMF (preserving its exact power spectrum and amplitude distribution while destroying nonlinear phase coupling) and run the exact same $(L^*, m^*, q^*)$ estimator on each surrogate.Reject any IMF whose $"NMSE"_("val")$ is not significantly lower than the surrogate distribution ($p < 0.05$) or whose $lambda_("orb", k)$ overlaps with the surrogate $lambda$ values—this filters out narrowband linear stochastic noise created by the EMD filter bank.

Once all 10 IMFs have been filtered, convert the survivors to common physical units via $lambda_("phys", k) = lambda_("orb", k) / (D_k . Delta t)$ and select $max_k lambda_("phys", k)$. If no IMF passes all six gates, report that the atmospheric noise series shows no verifiable deterministic chaos rather than forcing a maximum over spurious estimates.

#pagebreak()
== Limited-memory BFGS (L-BFGS) approach

An alternative to the NLS formulation is to treat the reconstruction simply as a general unconstrained optimization problem. Instead of exploiting the sum-of-squares structure, we can minimize the general scalar loss function $S(theta)$ using quasi-Newton methods, specifically the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm.

Like the Levenberg-Marquardt method, BFGS seeks to find the optimal step by approximating the Hessian matrix. The general Newton update rule is:

$theta_(k+1) = theta_k - alpha_k H_k^(-1) nabla S(theta_k),$

where $alpha_k$ is the step size (learning rate) determined by a line search, and $H_k^(-1)$ is the inverse Hessian matrix at iteration $k$.

Because computing and storing the exact inverse Hessian is mathematically demanding, BFGS iteratively updates an approximation of $H^(-1)$ using the history of the parameter updates and the gradients. Let $s_k = theta_(k+1) - theta_k$ be the change in parameters, and $y_k = nabla S(theta_(k+1)) - nabla S(theta_k)$ be the change in gradients. The secant equation requires that the new approximate Hessian satisfies $H_(k+1) s_k = y_k$.

The standard BFGS method stores a dense approximation matrix, which becomes memory-intensive as the parameter space $(m, L, q)$ grows. Therefore, the Limited-memory BFGS (L-BFGS) algorithm is preferred.

L-BFGS bypasses the need to store the full inverse Hessian matrix entirely. Instead, it implicitly computes the matrix-vector product $H_k^(-1) nabla S(theta_k)$ by storing only the $c$ most recent update vectors $(s_i, y_i)$.

While L-BFGS is a highly robust general-purpose optimizer, it relies purely on the first-order gradients to infer the curvature of the loss landscape. Because it does not explicitly construct the Gauss-Newton approximation $(J_theta^T J_theta)$ derived from the residuals, it generally requires more iterations to converge on tight local minima compared to the Levenberg-Marquardt method when applied strictly to least-squares problems.

#rslt[
  *Why this second method?*

  Compare to the NLS method, this approach is less smart and can be more heavy. Nevertheless, the approximation offered now by BenSaïda's `chaostest` function seems not to be precise enough for our use case.

  The real interest on the BFGS method is that it can be easily implemented in Python using the `PyTorch` library. Coupled with the fact that with Télécom Paris I have access to pretty powerful GPUs, we can explore bigger parameters and therefore hope to achieve an exploitable approximation of the initial timeseries.
]
#pagebreak()
= Open questions
- Is there any other/better method to do such timeseries reconstruction that may be less computationally expensive or/and more efficient ?
- BenSaïda's paper gives some suggestions but choosing the right range of $theta$ parameters to be tested remains delicate.
- In fine, `chaostest` not working on our datas is simply a question of pure computational implementation or is it something else ?

= First results

== MATLAB chaostest
After optimization, still struggles, especially with the first, highly chaotic, IMF's. Better after.

#nb[
  As we mainly use tanh as the activation function, we better should work on standardized amplitude rather than raw amplitude
]
Preventing tanh Saturation: $tanh(u)$ is only active and smooth inside $u in [-2, 2]$; outside $|u| > 3$, $tanh(u) approx 1 "or" -1$ and its derivative $"sech"^2(u) approx 0$. If a raw signal has a large offset ($mu eq 0$) or large variance ($sigma = 1$), the initial linear combinations $u = B x + C$ immediately land in the flat saturated tails, starving lsqnonlin / L-BFGS of gradient information. Conversely, if a raw signal has tiny amplitude ($sigma >> 1$, such as small IMFs or financial returns), $u approx 0$ and $tanh(u) approx u$ behaves purely linearly unless weights grow by $1/sigma$.Uniform Optimizer Tolerances & Comparable Errors: Levenberg-Marquardt and L-BFGS use fixed gradient/step tolerances ($10^-8$ to $10^-12$). Standardizing keeps parameters $O(1)$, improves Hessian conditioning, and puts SSE, RMSE, Mean Absolute Difference, and Maximum Absolute Difference on the exact same unit-variance scale across all IMFs and scripts.

== L-BFGS approach
Leads to better results but very long to compute on Telecom's GPU.
#highlight[Try to optimize even more the computation?]

== The third path: optimized NLS in python
Trying to keep the best of both worlds:
- Going back to BenSaïda's approach of optimizing S(theta) with NLS
- Implementing it in python and running it on GPU to achieve a better reconstruction?

= Understanding selection's criterias
Your intuition about NMSE is completely right and points directly to the main practical pitfall of BenSaïda’s original selection rule:
- Why BenSaïda (2012) maximizes $lambda$: BenSaïda formulates the problem as a conservative one-tailed hypothesis test ($H_0: lambda >= 0$ for chaos vs. $H_1: lambda <= 0$ for no chaos) rather than a forecasting competition. In Section 5 of the paper, he rejects Schwarz’s Information Criterion (BIC/SIC) because penalizing model complexity can discard higher-dimensional chaotic embeddings ($m > 2d$). Instead, he argues that if a deterministic chaotic map exists, at least one admissible triplet $(L, m, q)$ should uncover $lambda > 0$; therefore, taking $max_(L,m,q) hat(lambda)$ ensures that if even the largest exponent across the grid is significantly negative ($p < alpha$), chaos can be safely rejected.
- Where "Max $lambda$ Only" fails in practice: BenSaïda’s theory assumes every fitted neural network in the grid genuinely approximates the underlying conditional expectation $E[x_t, x_(t-L), dots, x_(t-m L)]$. On noisy or finite data, an over-parameterized or poorly converged network (especially at large $L$ or $m$) can fail to reconstruct the signal (NMSE $approx 0.98$, pure noise fit) while forming steep, artificial tanh cliffs between sparse points—creating a spurious positive $lambda > 0$ (a false positive for chaos).
- Why "Best NMSE Only" is also incomplete: In-sample NMSE decreases monotonically as you add parameters ($m, q$), which rewards overfitting noise; furthermore, a slightly larger embedding dimension $m$ can have nearly identical NMSE while properly unfolding a folded attractor so its Jacobian reflects the true divergence rate.
- The Principled Rule (Combining Both): You should never trust $lambda$ from a model with poor reconstruction (NMSE near 1) or unstable Jacobians. The most reliable workflow is a two-step filter (already supported in both Python scripts via --max-sse-ratio and --max-lam-spread):
  - Reconstruction & Stability Gate: Keep only (L, m, q) models whose SSE/NMSE is close to the global minimum (e.g., --max-sse-ratio 1.15) and whose $lambda_T$ is consistent across multiple polished starts (--max-lam-spread 0.1).
  - Selection: Pick the maximum $lambda_T$ (or lowest NMSE) inside that verified subset. If no model achieves a meaningfully low NMSE (like IMF_1 where NMSE $approx 0.70$), the signal is dominated by noise.

= How to state which method is more performant
Never use "largest $lambda$" to judge which solver/method is better. A poorly converged or overfitted model easily produces an artificially large (or falsely positive) $lambda$. Instead, evaluate solver performance using four objective criteria:
1. Lower NMSE / SSE / RMSE for the same (L, m, q): For a fixed architecture $(L, m, q)$ on the standardized signal, all three codes minimize the exact same nonlinear least-squares loss. Whichever method achieves the strictly lower NMSE (and SSE/RMSE) on the same (L, m, q) is objectively the stronger optimizer, because it avoided poor local minima.
2. Jacobian Stability Across Starts (lamT_min .. lamT_max): Between solvers achieving comparable NMSE, the superior method is the one whose top polished starts agree on $lambda_T$ (minimal spread), proving the network derivatives are well-regularized rather than saturated.
3. Benchmarking to compare the estimated exponent with the theoritical one

= Alternative approach (FNN)

The provided MATLAB script implements a complete, automated pipeline from *Nonlinear Time Series Analysis (NTSA)* to estimate the *Largest Lyapunov Exponent (LLE)*—a key metric used to detect and quantify deterministic chaos in a single scalar time series $x(t)$. Because no sampling frequency ($f_s$) is provided, the exponent is computed in units of $"samples"^(-1)$ (or $"bits/sample"$ if converted to base-2 logarithms).

= 1. Theoretical Foundations of the Code

== Phase Space Reconstruction (Takens' Embedding Theorem)
In experimental or empirical systems, we rarely observe all state variables simultaneously; instead, we measure a single scalar projection $x(n)$. *Takens' Embedding Theorem* proves that the topological and dynamical properties of the full, multidimensional attractor can be faithfully reconstructed from a single time series using *delay coordinates*.

For a time delay $tau$ and an embedding dimension $m$, a point $bold(X)_i$ in the reconstructed $m$-dimensional phase space is constructed as:
$ bold(X)_i = [ x(i), thick x(i + tau), thick x(i + 2 tau), thick dots.h, thick x(i + (m - 1) tau) ] $

Before doing this, the code normalizes the signal to zero mean and unit variance ($mu = 0, sigma = 1$) to ensure numerical stability and scale-invariance across different datasets.

== Step 2: Time Delay Estimation ($tau$) via Autocorrelation
To construct orthogonal, maximally informative coordinates, consecutive components $x(i)$ and $x(i + tau)$ must be independent enough not to collapse onto the main diagonal (redundancy), yet not so far apart that chaotic divergence destroys their causal link (irrelevance).

The code computes the *Autocorrelation Function (ACF)*:
$ C(tau) = 1 / (N - tau) sum_(i=1)^(N - tau) x(i) x(i + tau) $

- *Heuristic:* It selects the first lag $tau$ where $C(tau)$ drops below $1/e approx 0.3679$. This is the standard linear decay threshold indicating that linear correlations have sufficiently subsided. (If the time series behaves like a discrete map with fast decorrelation, it defaults to $tau = 1$).

== Step 3: Embedding Dimension ($m$) via False Nearest Neighbors (FNN)
If the chosen dimension $m$ is too small, the attractor is projected onto a lower-dimensional space, causing trajectories to intersect or fold over one another. Points that are far apart on the true attractor appear artificially close due to projection—these are *False Nearest Neighbors*.

Developed by *Kennel, Brown, and Abarbanel (1992)*, the FNN algorithm tests increasing dimensions $m = 1, 2, dots.h, m_max - 1$:

+ For a point $bold(X)_i^((m))$, find its nearest neighbor $bold(X)_j^((m))$ at Euclidean distance $d_m = norm(bold(X)_i^((m)) - bold(X)_j^((m)))_2$.
+ Increase the dimension to $m + 1$ by appending the next delayed coordinate $x(i + m tau)$ and $x(j + m tau)$. The new distance becomes:
  $ d_(m+1)^2 = d_m^2 + abs(x(i + m tau) - x(j + m tau))^2 $
+ A neighbor is flagged as *false* if either of two criteria is met:
  - *Relative Distance Criterion:* The distance increases drastically upon unfolding:
    $ abs(x(i + m tau) - x(j + m tau)) / d_m > R_"tol" quad (R_"tol" = 15) $
  - *Attractor Size Criterion:* The new distance becomes large relative to the overall size of the attractor ($R_A = sigma_x$):
    $ d_(m+1) / R_A > A_"tol" quad (A_"tol" = 2) $
+ The optimal $m$ is chosen as the first dimension where the percentage of FNN drops below *5%*, indicating that the attractor is completely unfolded.

== Step 4: Rosenstein’s Algorithm & The Theiler Window
Chaotic systems exhibit *sensitive dependence on initial conditions*: two infinitesimally close trajectories diverge exponentially at a rate governed by the Largest Lyapunov Exponent ($lambda_1$):
$ d(k) approx d(0) e^(lambda_1 k) $

*Rosenstein et al. (1993)* formulated a fast, robust method to estimate $lambda_1$ from small, noisy datasets:

+ *Nearest Neighbor Search with Theiler Window:* For every reference vector $bold(X)_i$, find its closest neighbor $bold(X)_(hat(i))$. However, points immediately preceding or following $bold(X)_i$ in time ($abs(i - hat(i)) <= W$) lie on the _same_ trajectory segment and are close simply due to temporal continuity rather than spatial recurrence. Excluding indices within a *Theiler Window* $W$ (`ventana_theiler`) prevents biased, artificially low divergence estimates.
+ *Tracking Divergence:* Track the Euclidean distance between each pair $(i, hat(i))$ over $k$ discrete time steps forward:
  $ d_i(k) = norm(bold(X)_(i+k) - bold(X)_(hat(i)+k))_2 $
+ *Averaging in Log-Space:* Compute the average logarithmic divergence across all $M_"valid"$ pairs:
  $ y(k) = < ln d_i(k) >_i $

== Step 5: Linear Scaling Region Fit
Because an attractor is bounded in phase space, trajectories cannot diverge forever—eventually, $y(k)$ saturates at the diameter of the attractor. Therefore, $lambda_1$ is estimated by performing a *least-squares linear regression* (`polyfit`) over the initial linear scaling region of $y(k)$ vs. $k$:
$ y(k) approx lambda_1 k + c $

- *$lambda_1 > 0$:* Deterministic chaos (exponential divergence).
- *$lambda_1 = 0$:* Periodic or quasi-periodic limit cycle (trajectories remain parallel).
- *$lambda_1 < 0$:* Convergence to a stable fixed point.

= 2. Key Optimizations in the Refactored Code

+ *Encapsulation & Optional Arguments:* Converted into a reusable MATLAB function using `arguments` validation, allowing default parameters while permitting user overrides and optional plotting.
+ *Vectorized Autocorrelation & Hankel Embedding:* Replaced loop-based phase space construction with direct index matrices (`idx_mat = (1:M)' + (0:m-1)*tau`), eliminating redundant memory allocations.
+ *Fast Pairwise Distance Calculations (`pdist2` / Vectorization):* Replaced the $O(M^2)$ inner loop distance calculations in both FNN and Rosenstein's algorithm with vectorized matrix operations (and `pdist2` when Statistics and Machine Learning Toolbox is available, with a fast fallback), yielding a *10x–50x speedup* on medium-to-large time series.
+ *Vectorized Divergence Tracking:* Eliminated the nested `for k = 0:max_pasos` loop by computing Euclidean norms across all time steps simultaneously.
+ *Bug Fix in FNN Loop Bounds:* In the original code, if the `for m_test` loop terminated early via `break` (`if M_test < 50`), the remaining entries in `fnn_porcentaje` stayed `0`, which falsely triggered `find(fnn_porcentaje < 5)` at an uncomputed dimension. Initializing `fnn_porcentaje` with `NaN` prevents this bug.
