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
- frac_alea : random gaussian white noises, 10 000 points
- ruido_blanco : atmospheric noise, 10 000 points

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