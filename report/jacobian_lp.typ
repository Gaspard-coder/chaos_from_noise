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
  x_t approx alpha_0 + sum_(j=1)^q alpha_j Psi(
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

