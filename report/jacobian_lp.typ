#import "@preview/diverential:0.3.0": *

#set page(
  paper: "a4",
  margin : 2.5cm
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
This short report aims to explain and explore different approach to compute the Lyapunov exponent using the well-knownn Jacobian matrix as well as the Oseledet's theorem. It heavily relies on the work of Ahmed BenSaïda with its very interesting paper : #link("https://aijcr.thebrpi.org/journals/Vol_2_No_8_August_2012/6.pdf"
)[#underline(text(fill:blue)[Are Financial Markets Stochastic: A Test for Noisy Chaos])].

The idea is not to be as extensive as BenSaïda's work but to see how his approach can be customed to detect chaos.

= The method

Let's take a given timeseries ${x_t}^T#sub[t=1]$. The first step and in fine the core of this approach is to find the chaotic mapping $f$ to deduce $x_t$ from a sub group of its predecessors ${x#sub[t-L],x#sub[t-2L],...,x#sub[t-mL]}$:

$ x_t = f(x#sub[t-L],x#sub[t-2L],...,x#sub[t-mL]) + epsilon_t, $
where $epsilon_t$ is added white noise.

#nb[It introduces two parameters: $m$ the embedding time delay and $L$ the lag or embedding dimension. It is reminiscent of permutation entropy sampling's parameters but don't have to be confused]

Given this mapping, we denote J the Jacobian matrix defined as :

$
  J_t = mat(
    dvp(f,x#sub[t-L]),dvp(f,x#sub[t-2L]),...,...,dvp(f,x#sub[t-mL]);
    0,1,...,0,0;
    dots.v,dots.v,dots.down,dots.v,dots.v;
    0,0,...,1,0;
  ).
$
We then note T the Jacobian product of the M first Jacobian matrix
$
  T_M = product_(t=1)^M J_t.
$

Then, we introduce $U_0 = mat(1;0;dots.v;0)$ an unitary vector leading to
$
  V_M = (T_M U_0)'(T_M U_0),
$
where the (.)' is the matrix transpose operator.

Given the fact that $V_M$ is diagonalizable, we note $v_1 = max("Sp("V_M")")$ the largest eigenvalue of $V_M$.

Finally, the #link("https://www.people.maths.bris.ac.uk/~macpd/ads/oseledets.pdf")[#underline(text(fill:blue)[Oseledet's theorem])] states that
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
  hat(x_t) approx alpha_0 + sum_(j=1)^q alpha_j Psi(
  beta_(0,j) + sum_(i=1)^m beta_(i,j) x_(t-i L)
) + epsilon_t
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

  \ hat(x_t) approx alpha_0 + sum_(j=1)^q alpha_j tanh(
  beta_(0,j) + sum_(i=1)^m beta_(i,j) x_(t-i L)
) + epsilon_t \
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

= Alternative approach
