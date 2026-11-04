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