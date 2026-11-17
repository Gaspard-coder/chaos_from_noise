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
    Finding chaos in randomness
  ]
]

#outline()

= Introduction
[explanations]

== Datasets
- frac_alea : random gaussian white noises, 10 000 points, 10 samples
- ruido_blanco : atmospheric noise, 10 000 points, 10 samples

==> *$m=6,tau=1$*

= Decomposition methods

== EMD
[explanations on how it works]

IMF = Intrinsic Mode Function

==> MATLAB implementation with `Signal Multiresolution Analyzer` app and inline function:
`[imf,residula,info] = emd(data,varagin{:})`

Parameters:
- SiftRelativeTolerance - Convergence criterion threshold (default: 0.2).
- SiftMaxIterations     - Maximum sifting iterations per IMF (default: 100).
- MaxNumIMF             - Maximum number of IMFs to extract (default: 10).
- MaxNumExtrema         - Extrema count in residual to stop sifting (default: 1).
- MaxEnergyRatio        - Maximum signal-to-residual energy ratio (default: 20).
- Interpolation         - Envelope interpolation method: "spline" or "pchip".
- Display               - Toggle sifting diagnostic table: 0 or 1 (default: 0).
- SampleRate            - Sampling frequency of signal in Hz.

According to the bibliography `Empirical_Mode_Decomposition_-_an_introduction`, we first use emd with `MaxNumIMF = 8` (3 spikes in the residual) and `Interpolation = spline`, others parameters still at default. #highlight(fill:yellow)[(empirical quality criteria, to discuss)]

= Tools for IMF characterization

== runstest
Run test for *randomness*. For now, we just use `runstest(data,'Alpha')` with Alpha the significance level. #highlight(fill:yellow)[criterias to be discussed]

All parameters:
- x (Required Data Input): Vector containing the sequence of data values to be tested for randomness.

- v (Threshold / Cutoff Value): Scalar threshold value used to convert continuous data into a binary sequence of runs (values above vs. values at or below v). If not specified, the sample median of x is used as the default baseline.

- Alpha (Significance Level): Scalar value between 0 and 1 that defines the significance level of the hypothesis test. The default value is 0.05 (5% significance level).

- Method (P-Value Computation Method): String or character vector specifying how the p-value is computed:

  - 'exact' — Computes the exact p-value based on the exact permutation distribution of runs.

  - 'approximate' — Computes the p-value using a normal distribution approximation (recommended for larger datasets).

- Tail (Alternative Hypothesis): String or character vector indicating the direction of the two-sample hypothesis test:

    'both' (Default) — Two-tailed test testing whether the number of runs is significantly higher or lower than expected.

    'right' — One-tailed test testing whether the number of runs is significantly greater than expected.

    'left' — One-tailed test testing whether the number of runs is significantly less than expected.

*Example of output:*
File: IMF_1.mat       | h: 1 | p-value: 0.0000 | Runs: 6570 | Decision: REJECT Null (Non-Random)

with:
- h (Hypothesis Decision)
This is a binary indicator representing the final verdict of the statistical test. h = 0: the test fails to reject the null hypothesis. This means there is not enough evidence to say the data is non-random; the sequence behaves like random noise. h = 1: the test rejects the null hypothesis. This means the data is highly unlikely to be random and contains a distinct pattern, trend, or structured oscillation.
- p-value (Probability Value)
The p-value measures the probability of observing your specific sequence of data assuming the null hypothesis (that the data is perfectly random) is true.A p-value below your chosen significance level (e.g., < 0.05) means the observation is extremely rare for random data, prompting you to reject the assumption of randomness (h = 1).A higher p-value means the data is consistent with randomness.

- Runs
A "run" is an uninterrupted sequence of consecutive data points that all fall on the same side of a cutoff line (typically the sample median). For example, if a signal's values relative to the median are (+, +, -, -, -, +), that represents exactly 3 runs: one group of positives, one group of negatives, and one final positive. A perfectly random signal has a predictable, average number of runs; too many runs indicates high-frequency oscillation, while too few runs indicates a slow trend or flatline.

== chaostest
Run test for *chaos* using Bensaïda MATLAB function.
[Explanations of every parameters]

== Deter test

This test implements *Recurrence Quantification Analysis (RQA)* that measures the determinism of a 1-dimensional time series.

The method works in 3 parts:
1. Phase space reconstruction: To analyze the hidden, underlying dynamics of a 1D signal, the algorithm mathematically reconstructs a multi-dimensional phase space. It does this using time-delay embedding, which requires an Embedding Dimension ($m$) (set to 3 in the code) to define the number of dimensions in the new space, and a Time Delay ($tau$) (set to 2) to define the sample gap. This transforms the 1D time series into a matrix of multi-dimensional state vectors, capturing the system's trajectory over time

2. Recurrence Plots
Dynamical systems theory assumes that deterministic systems will eventually revisit previous states or regions in their phase space. The code computes a distance matrix by measuring the Euclidean distance between every pair of reconstructed state vectors. A threshold ($epsilon$, set to 10% of the maximum distance) is applied to this matrix. If two states are closer than $epsilon$, they are considered a "recurrence." This translates into a binary matrix visualized as a Recurrence Plot, where recurrences are plotted as black dots.

3. Quantifying determinism
The core theory of RQA states that the visual structures within a Recurrence Plot correlate to the system's physical behavior. Random noise produces isolated, scattered recurrence points. Conversely, deterministic systems produce points that organize into diagonal lines, which occur when two distinct segments of the phase space trajectory evolve similarly over consecutive time steps. The algorithm calculates the Determinism (DET) metric by dividing the number of recurrence points that form diagonal lines by the total number of recurrence points (excluding the main identity diagonal). A DET above 40% is classified here as primarily deterministic, while a lower score indicates random or noisy behavior. #highlight[Where the 40% criteria  comes from?]

= Experiments

== First tests

We first generate the EMDs with standard parameters `EMD_gen(frac_alea(1,:),'database/frac_alea/frac_alea_EMD_1','MaxNumIMF',8);`
Then we plot the decomposition of the first frac_alea signal on the HxC plane, leading to:

#figure(
  image("figures/HxC_fracalea_1.png"),
  caption: [IMFs positionning on the Entropy-Complexity plane from the decomposition of frac_alea_1]
)

RQ: IMF_frac_alea_1 is the original signal.

Observations: 3 types of IMF: random (1), chaotic (2 and 3 maybe), deterministic (4 ->8)

*Next step:* Categorize each IMF (random/chaotic/deterministic) and check the labelling with `runstest`, `chaostest` and maybe Lyapunov Exponent (which $F_s$?)

=== Randmoness test
For now, `alpha =0.05`

#figure(
  table(
    columns: (1fr,1fr,1fr,1fr,1fr),
    table.header([File],[h],[p-value],[runs], [Decision]),
    [IMF_1],[1],[0],[6749],[Non-Random],
    [IMF_2],[1],[0],[2696],[Non-Random],
    [IMF_3],[1],[0],[1130],[Non-Random],
    [IMF_4],[1],[0],[472],[Non-Random],
    [IMF_5],[1],[0],[204],[Non-Random],
    [IMF_6],[1],[0],[84],[Non-Random],
    [IMF_7],[1],[0],[34],[Non-Random],
    [IMF_8],[1],[0],[16],[Non-Random],
  ),
  caption: [Randomness classification for the EMD of frac_alea_1]
)
Quite accurate, nobody is random

=== Chaos test
For now we just tried with default parameters ie:
- ActiveFN:'tanh' Activation function used in the neural network estimation ('tanh', 'logistic', or 'sigmoid')
- maxOrders[5, 6, 5] Vector specifying maximum model orders $[L, m, q]$.
- alpha: 0.05 Significance level for the test ($5\%$).
- flag: 'VARY' Mode specifying whether to vary order triplets $[L, m, q]$ ('VARY') or fix them ('FIX').#highlight(fill:yellow)[To be discussed]
Nb : the lyapunov exponent is computed without the use of the associated matlab function => no sampling frequency

We obtain
#figure(
  table(
    columns: (1fr,1fr,1fr,1fr,1fr,1fr),
    table.header([File],[h],[p-value],[$lambda$],[Orders], [Decision]),
    [IMF_1],[1],[0],[-0.2858],[[1, 6, 4]],[Non-Chaotic],
    [IMF_2],[1],[0],[-0.0800],[[1, 3, 1]],[Non-Chaotic],
    [IMF_3],[0],[0.2123],[-0.0102],[[1, 3, 2]],[Chaotic],
    [IMF_4],[0],[1],[0.0401],[[1, 4, 1]],[Chaotic],
    [IMF_5],[0],[1],[0.6365],[[2, 3, 1]],[Chaotic],
    [IMF_6],[0],[1],[0.7834],[[3, 5, 1]],[Chaotic],
    [IMF_7],[0],[1],[0.7817],[[2, 3, 1]],[Chaotic],
    [IMF_8],[0],[1],[0.7834],[[5, 3, 1]],[Chaotic],
  ),
  caption: [Chaos classification for the EMD of frac_alea_1]
)

Observations: Efficient for rejecting randomness but not deterministic behavior. #highlight(fill:yellow)[Need to tune it]

=== Deter test
To give a first idea, we run the deter test on the EMD of a random atmospherical signal `frac_alea_EMD_2` with $m=3, tau=2 "and" epsilon=0.1$

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: (left, right, left),
    stroke: none,
    table.hline(),
    table.header(
      [*File Name*], [*DET (%)*], [*Classification*],
    ),
    table.hline(stroke: 0.5pt),
    [`IMF_1.mat`], [20.701], [Random/Noisy],
    [`IMF_2.mat`], [81.011], [Deterministic],
    [`IMF_3.mat`], [99.106], [Deterministic],
    [`IMF_4.mat`], [99.332], [Deterministic],
    [`IMF_5.mat`], [99.998], [Deterministic],
    [`IMF_6.mat`], [99.999], [Deterministic],
    [`IMF_7.mat`], [100], [Deterministic],
    [`IMF_8.mat`], [100], [Deterministic],
    [`IMF_9.mat`], [100], [Deterministic],
    [`IMF_10.mat`], [100], [Deterministic],
    [`IMF_frac_alea_2.mat`], [2.8544], [Random/Noisy],
    table.hline(),
  ),
  caption: [Determinism classification results $m=3, tau=2 "and" epsilon=0.1$],
)

The $epsilon$ seems too lose here, as almost every component of the decomposition are classified as deterministic. This does not align with the HxC feature plane:

#figure(
  image("figures/HxC_fracalea_EMD_2.png"),
  caption: [Complexity Entropy plane for the EMD of frac_alea_2]
)

With $epsilon = 0.05$ and $epsilon=0.01$

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: (left, right, left),
    stroke: none,
    table.hline(),
    table.header(
      [*File Name*], [*DET (%)*], [*Classification*],
    ),
    table.hline(stroke: 0.5pt),
    [`IMF_1.mat`], [8.9129], [Random/Noisy],
    [`IMF_2.mat`], [54.478], [Deterministic],
    [`IMF_3.mat`], [97.672], [Deterministic],
    [`IMF_4.mat`], [98.162], [Deterministic],
    [`IMF_5.mat`], [99.743], [Deterministic],
    [`IMF_6.mat`], [99.998], [Deterministic],
    [`IMF_7.mat`], [99.999], [Deterministic],
    [`IMF_8.mat`], [100], [Deterministic],
    [`IMF_9.mat`], [100], [Deterministic],
    [`IMF_10.mat`], [100], [Deterministic],
    [`IMF_frac_alea_2.mat`], [0.45311], [Random/Noisy],
    table.hline(),
  ),
  caption: [Determinism classification results $m=3, tau=2 "and" epsilon=0.05$],
)

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: (left, right, left),
    stroke: none,
    table.hline(),
    table.header(
      [*File Name*], [*DET (%)*], [*Classification*],
    ),
    table.hline(stroke: 0.5pt),
    [`IMF_1.mat`], [0.44798], [Random/Noisy],
    [`IMF_2.mat`], [6.898], [Random/Noisy],
    [`IMF_3.mat`], [80.276], [Deterministic],
    [`IMF_4.mat`], [95.757], [Deterministic],
    [`IMF_5.mat`], [97.557], [Deterministic],
    [`IMF_6.mat`], [98.069], [Deterministic],
    [`IMF_7.mat`], [99.673], [Deterministic],
    [`IMF_8.mat`], [99.999], [Deterministic],
    [`IMF_9.mat`], [99.999], [Deterministic],
    [`IMF_10.mat`], [99.999], [Deterministic],
    [`IMF_frac_alea_2.mat`], [0], [Random/Noisy],
    table.hline(),
  ),
  caption: [Determinism classification results $m=3, tau=2 "and" epsilon=0.01$],
)

Finally, with $epsilon=0.001$, each means a difference in the trajectory of 0.1%, we obtains a classification correlated with the HxC feature plane:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: (left, right, left),
    stroke: none,
    table.hline(),
    table.header(
      [*File Name*], [*DET (%)*], [*Classification*],
    ),
    table.hline(stroke: 0.5pt),
    [`IMF_1.mat`], [0], [Random/Noisy],
    [`IMF_2.mat`], [0], [Random/Noisy],
    [`IMF_3.mat`], [18.627], [Random/Noisy],
    [`IMF_4.mat`], [68.901], [Deterministic],
    [`IMF_5.mat`], [90.939], [Deterministic],
    [`IMF_6.mat`], [97.876], [Deterministic],
    [`IMF_7.mat`], [98.187], [Deterministic],
    [`IMF_8.mat`], [98.808], [Deterministic],
    [`IMF_9.mat`], [99.489], [Deterministic],
        [`IMF_10.mat`], [98.658], [Deterministic],

    [`IMF_frac_alea_2.mat`], [NaN], [Random/Noisy],
    table.hline(),
  ),
  caption: [Determinism classification results $m=3, tau=2 "and" epsilon=0.01$],
)

Fine tuning only $epsilon$ might not be enough to distinguish determinist from other signals. To also optimize $m "and" tau$, we will need a pre-classify set of signal as a training reference. #highlight[TO DO NEXT]

=== Fine-tuning deter test parameters
To proceed the most efficient sweep in the parameters ($m, tau, epsilon$) of deter test, we generated 10 000 points synthetical signals and labelled them:
- Deterministic: Chirp, exxponential, henon map, linera, logistic map, lorenz map, polynomial, quadratic, rossler map, sawtooth, sinusoid, two tone 
- Random: AR1 colored, random binary, random walk, smoothed noise, white gaussian, white uniform

We used the following quality parameters:
- Accuracy, BalancedAccuracy, TPR (deterministic recall),
- TNR (random recall)  -> at the fixed opts.Threshold
- AUC                  -> threshold-free separability
- OptThreshold, OptBalAccuracy -> best achievable threshold
- MeanDET_Det, MeanDET_Rnd     -> mean DET of each class
- Margin = min(DET of deterministic) - max(DET of random);-> 0 means a perfect separating threshold exists

We generated 3 independent realisations of each signal, leading to 54 labelled timeseries to be used. We swept for $m in [2:6], tau in [1:3] "and" epsilon in [0.01,0.03,0.05,0.07,0.1]$. To shorten the computation time, we first used only 2000 points for the first try:

#table(
  columns: (auto,auto,auto,auto,auto,auto,auto,auto,auto,auto,auto,auto,auto),
  align: center,
  inset: 5pt,
  stroke: 0.5pt,

  [*m*], [*tau*], [*eps*], [*Accuracy*], [*Balanced Accuracy*],
  [*TPR*], [*TNR*], [*AUC*], [*OptThreshold*],
  [*OptBalAccuracy*], [*MeanDET_Det*], [*MeanDET_Rnd*], [*Margin*],

  [5], [2], [0.01], [1], [1], [1], [1], [1], [50.034], [1], [96.092], [3.1745], [50.263],
  [6], [2], [0.01], [1], [1], [1], [1], [1], [47.22],  [1], [96.16],  [2.1236], [48.964],
  [4], [2], [0.01], [1], [1], [1], [1], [1], [52.834], [1], [96.078], [4.7419], [46.886],
  [3], [2], [0.01], [1], [1], [1], [1], [1], [55.905], [1], [95.599], [7.5883], [38.093],
  [4], [3], [0.01], [1], [1], [1], [1], [1], [45.167], [1], [96.114], [5.1],    [31.859],
)

#rslt[
  At first sight, the best configuration seems to be $(m=5,tau=2,epsilon=0.01)$
]

We did the computation again but with 10 000 points:

#text(weight: "bold")[Top parameter sets (ranked by BalancedAccuracy, threshold = 40%):]

#table(
  columns: (auto,auto,auto,auto,auto,auto,auto,auto,auto,auto,auto,auto,auto),
  align: center,
  inset: 5pt,
  stroke: 0.5pt,

  [*m*], [*tau*], [*eps*], [*Accuracy*], [*BalancedAccuracy*],
  [*TPR*], [*TNR*], [*AUC*], [*OptThreshold*],
  [*OptBalAccuracy*], [*MeanDET_Det*], [*MeanDET_Rnd*], [*Margin*],

  [6], [2], [0.01], [0.96296], [0.94444], [1], [0.88889], [1], [71.027], [1], [96.149], [7.9543], [11.912],
  [5], [2], [0.01], [0.96296], [0.94444], [1], [0.88889], [1], [71.696], [1], [95.992], [8.8499], [9.0605],
  [4], [2], [0.01], [0.96296], [0.94444], [1], [0.88889], [1], [73.043], [1], [95.852], [10.334], [6.355],
  [5], [3], [0.01], [0.96296], [0.94444], [1], [0.88889], [0.99846], [71.094], [0.98611], [95.808], [8.7746], [-7.3647],
  [4], [3], [0.01], [0.96296], [0.94444], [1], [0.88889], [0.99846], [72.814], [0.98611], [95.744], [10.498], [-8.2438],
)
This time, we got some misclassification:

#figure(
  image("figures/deter_test_10000.png")
)

But given the type of application, we can remove the binary and the random walk from the candidates and redo a parameter research.

== Checking the trustfullness of runstest
As we see, the rejection of randomness on chaotic signals works pretty well. We tried for each decomposition if runstest's results are consistent.

*frac_alea (atmospheric noise)*

#figure(
  image("figures/Random IMF Components per Subfolder fracalea.png"),
  caption: [Number of random signals for each EMD]
)

For every decomposition, the only random signal detected is the original signal itself (except for the first one that is somehow not considered as random)

*ruido_blanco (white noise)*

#figure(
  image("figures/Random IMF Components per Subfolder ruidoblanco.png"),
  caption: [Number of random signals for each EMD]
)

This time every original signal is well-detected and there is still no random sub-signals.

== Trying recursive decomposition
Given the fact that some tests sometimes struggle to characterize the first components of the decomposition (usually L1 and L2), we tried to apply a second time the EMD on them
to be more precise.

#figure(
  image("figures/frac_alea_EMD_2.1_energy.png",width: 130%),
  caption: [Repartition of energy in the EMD of L1, the first component of the frac_alea EMD]
)
It appears that the first component of the sub-decomposition, noted L1.1, concentrated for 99.17% of the total energy of L1.

When plotting the sub-decomposition on the HxC plane, we obtained:

#figure(
  image("figures/HxC_frac_alea_IMF_1_EMD_2.png"),
  caption: [EMD of L1, the first component of the EMD of frac_alea, on the HxC plane ($m=6,tau=1$)]
)

The L1.1 component appears clearly chaotic and we might conclude that L1 is also chaotic. #highlight(fill:yellow)[might need to confirm it with others metrics]

== Autocorrelation

First approach with $tau$ from 0 to 2000 (N#sub[sample]/4)
#figure(
  image("figures/autocorrelation_frac_alea_EMD_2.png",width: 120%),
  caption: [ACF for the components of the EMD of frac_alea_2]
)
All of them cross 0: at least partially chaotic. Difficult to use as a precise tool to distinguish determinist and chaotic behavior

#pagebreak()

== Correlation dimension
Type of fractal dimension #highlight[Why not also try fractal dimension as a chaotic classifier]


=== Phase Space Reconstruction

Given a discrete scalar time series ${x_1, x_2, ..., x_N}$ sampled at uniform time intervals, the underlying multidimensional dynamics of the system can be reconstructed using Takens' delay-coordinate embedding theorem. By choosing an embedding dimension $m$ and a time delay $tau$, a sequence of state vectors $bold(X)_i in RR^m$ is constructed as:

$ bold(X)_i = (x_i, x_(i + tau), ..., x_(i + (m - 1)tau)) in RR^m, quad i = 1, 2, ..., N_m $

where $N_m = N - (m - 1)tau$ is the total number of reconstructed vectors in the embedded phase space. According to Takens' theorem, if $m > 2 D_2 + 1$, the reconstructed trajectory preserves the topological and metric invariants of the original attractor without self-intersections.

=== The Correlation Integral

To quantify the spatial distribution of points on the reconstructed attractor, the Grassberger-Procaccia algorithm defines the empirical *correlation integral* (or correlation sum) $C(r)$ as the fraction of all distinct point pairs separated by a distance less than or equal to a threshold radius $r$:

$ C(r) = frac(2, N_m (N_m - 1)) sum_(i=1)^(N_m - 1) sum_(j=i+1)^(N_m) Theta(r - norm(bold(X)_i - bold(X)_j)) $

where $norm(dot)$ denotes the Euclidean norm in $RR^m$, and $Theta(dot)$ is the Heaviside step function:

$ Theta(u) = cases(
  1 & "if" u >= 0,
  0 & "if" u < 0
) $

=== Power-Law Scaling and Dimension Estimation

For a self-similar attractor in the limit of infinitely many data points ($N -> infinity$) and vanishingly small length scales ($r -> 0$), the correlation integral obeys a power-law scaling relation:

$ C(r) prop r^(D_2) $

The correlation dimension $D_2$ is formally defined as the scaling exponent:

$ D_2 = lim_(r -> 0) lim_(N -> infinity) frac(ln C(r), ln r) $

In practical applications with finite, noisy datasets, the limit $r -> 0$ cannot be evaluated directly. Instead, taking the logarithm of the scaling relation yields a linear relationship over an intermediate range of radii known as the *scaling region*:

$ ln C(r) approx D_2 ln r + c $

where $c$ is a constant. Consequently, $D_2$ is estimated as the slope $frac(d ln C(r), d ln r)$ obtained via linear regression within this intermediate linear regime—bounded below by the measurement noise floor and finite-sample depopulation, and bounded above by the overall diameter of the attractor.

=== Embedding Saturation and System Classification

Because the true dimension of the system is unknown _a priori_, $D_2$ is computed iteratively across increasing embedding dimensions $m$. The asymptotic behavior of $D_2(m)$ distinguishes the dynamical nature of the signal:

- *Deterministic Regular Signals (Periodic / Quasi-periodic):* As $m$ increases, $D_2$ rapidly converges and saturates at an integer value corresponding to the topological dimension of the manifold (e.g., $D_2 = 1$ for a simple limit cycle, or $D_2 = 2$ for a two-frequency torus).
- *Deterministic Chaotic Signals:* $D_2$ initially increases with $m$ until the attractor is fully unfolded ($m > 2 D_2 + 1$), at which point it plateaus and saturates at a finite, non-integer (fractal) value.
- *Stochastic Signals (Random Noise):* Because uncorrelated noise uniformly fills any space in which it is embedded, $D_2$ does not saturate and instead grows linearly with the embedding dimension ($D_2 approx m$).

=== Let's test it
One more time, we will try this test by comparing the classification result with the HxC feature plane associated with the used signal. We stil use the EMD of `frac_alea_2`

#figure(
  image("figures/HxC_fracalea_EMD_2.png"),
  caption: [Complexity Entropy plane for the EMD of frac_alea_2]
)

#figure(
  {
    set text(size: 8.5pt)
    table(
      columns: (auto, auto,auto,auto,auto,auto, 1fr),
      align: (left, center, center, center, center, center, left),
      inset: (x: 5pt, y: 6pt),
      stroke: 0.5pt + luma(180),
      fill: (_, y) => if y < 2 { luma(235) } else if calc.even(y) { white } else { luma(248) },
      table.header(
        table.cell(rowspan: 2, align: bottom + left)[*File Name*],
        table.cell(colspan: 5, align: center)[*Correlation Dimension ($D_2$)*],
        table.cell(rowspan: 2, align: bottom + left)[*Signal Classification*],
        [$m = 2$], [$m = 3$], [$m = 4$], [$m = 5$], [$m = 6$],
      ),
      [`IMF_1.mat`], [1.7750], [2.2492], [2.7087], [3.2104], [3.7157], [Random / Stochastic Noise],
      [`IMF_2.mat`], [1.9160], [2.2544], [2.5353], [2.7800], [3.0706], [Chaotic (Deterministic Chaos)],
      [`IMF_3.mat`], [1.8346], [1.9050], [1.9716], [2.0912], [2.2023], [Chaotic (Deterministic Chaos)],
      [`IMF_4.mat`], [1.5571], [1.7325], [1.8016], [1.8468], [1.8916], [Chaotic (Deterministic Chaos)],
      [`IMF_5.mat`], [1.3123], [1.4246], [1.5107], [1.5846], [1.6781], [Chaotic (Deterministic Chaos)],
      [`IMF_6.mat`], [1.1158], [1.2223], [1.2654], [1.3252], [1.3913], [Chaotic (Deterministic Chaos)],
      [`IMF_7.mat`], [1.0229], [1.0671], [1.1028], [1.1353], [1.1574], [Chaotic (Deterministic Chaos)],
      [`IMF_8.mat`], [0.9614], [0.9728], [0.9806], [1.0038], [1.0183], [Deterministic Regular (Periodic / Quasi-periodic)],
      [`IMF_9.mat`], [0.8223], [0.8256], [0.8354], [0.8402], [0.8444], [Deterministic Regular (Periodic / Quasi-periodic)],
      [`IMF_10.mat`], [0.8940], [0.8951], [0.8945], [0.8984], [0.8992], [Deterministic Regular (Periodic / Quasi-periodic)],
      [`IMF_frac_alea_2.mat`], [1.8882], [2.7148], [3.4541], [4.1162], [4.7309], [Random / Stochastic Noise],
    )
  },
  caption: [Evolution of the correlation dimension $D_2$ across embedding dimensions $m in [2, 6]$ and resulting signal classifications.],
)
#nb[
  The classification thresold of `cor_dim_batch.m` are approximative, one should always check the behaviour of the correlation dimension to really classify the signal.
]

*Ideas of improvements:*
- use greate values of m_max like 8 or 10 for chaotic signals to fully unfold and reach the plateau
- Try with a bit greater $tau$ for example 2 or 3.
- Reduce the saturation slop threshold `saturation_slope_thresh`
- Reduce a bit also the saturation flat threshold `integer_tol`

#rslt[
  After applying the above improvement suggestions, the only change is sometimes the classification of IMF_2 and IMF_3 as random rather than chaotic.

  At least, the correlation dimension is a very good candidate as a deterministic test.
]

#pagebreak()

= Stochastic analysis workflow
Let's summarize the aim of this work : demonstrating that stochastic ie random/pseudo-random signals can be decomposed into chaotic and deterministic components.

== Why ?
Achieving to demonstrate that we can generate "non-artifical" chaotic signals (in opposite to deterministic-generated chaotic signals like Hennon map,etc) from random (and maybe pseudo-random) samples.

== Potential applications
To give a new and simple way to generate unpredictable cryptographic keys

== Roadmap
1. 2 datasets: Random samples from atmospheric measurements (`frac_alea`) and pseudo-random samples generated from white noise (`ruido_blanco`)
2. Apply a decomposition: for now we only used Empirical Mode Decomposition (EMD) because it is simple and complete (we can fully recover the original signal)
3. We now need to characterize the elements of the EMD to prove that we obtained chaotic sub-signals. There are different criterias:

 3.1 `runtest` : as shown before, `runstest` is a reliable MATLAB function to detect the randomness of a signal, with a given trust interval.

 3.2 As the sub-components {$L_k$} cannot be random, they are rather deterministic or chaotic. We now need to measure the efficiency of different tools to complete the characterization. Good candidates are :
 - Autocorrelation : A chaotic signal is not correlated.
 - Phase-space (2D and maybe 3D): showing the existence of chaotic behavior as well as attractor is a very good sign of chaoticity.
 - HxC plane: an other visual way to decide between chaotic and deterministic behavior
 - Lyapunov exponent: obivous metric to detect chaotic behavior, it nevertheless requires a sampling frequency difficult to detect after the decomposition that implies interpolation. Severals techniques have to be tested to bypass this limit:
  - Using only the sign of the `LyapunovExponent`MATLAB function: as we just need a boolean attribution, getting the sign is enought to chose between deterministic and chaotic #highlight(fill:yellow)[Needs to be demonstrate that the sign is still right]
  - Using the `chaostest` function from Dr. Bensaïda: By applying A LOT of maths and a small neural network, the algorithm is able to compute the Lyapnuov Exponent without the sampling frequency of the input. Nevertheless, it appears that it didn't work that much in our case. Parameters need probably a fine tuning but the idea of this work is also to keep the parameters set as small as possible. #highlight(fill:yellow)[Nevertheless, we can try to implement a simpler version of this approach using dedicated tools like PyTorch lib rather then MATLAB hardcoding]
 - `deterministic test`: function from a previous work #highlight[To be tested]
