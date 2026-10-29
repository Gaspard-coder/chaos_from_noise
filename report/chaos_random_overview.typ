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