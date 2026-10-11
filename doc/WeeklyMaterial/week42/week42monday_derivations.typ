// Week 42 Monday: every derivation of the lecture, written out.
// Companion to week42monday.tex (slides) and week42monday_whiteboard.md
// (lecturer notes).  For printing, or for showing on screen if the
// projector/board setup fails.  Same notation as the slides.
// Build:  typst compile week42monday_derivations.typ

#set document(title: "Week 42 Monday: derivations")
#set page(paper: "a4", margin: (x: 2cm, y: 2cm), numbering: "1")
#set text(size: 12pt, lang: "en")
#set par(justify: false, leading: 0.65em)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => {
  v(0.6em)
  block(text(size: 16pt, it))
  v(0.2em)
}
#show heading.where(level: 2): it => block(above: 1.1em, below: 0.6em, it)

#let boardblue = rgb(26, 64, 140)
#let had = sym.compose // element-wise product, as on the slides
#let board(title) = block(
  fill: boardblue.lighten(90%), stroke: (left: 3pt + boardblue),
  inset: (x: 8pt, y: 6pt), width: 100%,
  text(fill: boardblue, weight: "bold")[On the board: #title],
)
#let result(body) = block(
  stroke: 0.8pt + luma(120), inset: 8pt, radius: 3pt, width: 100%, body,
)
#let note(body) = text(size: 10.5pt, fill: luma(70), body)

#align(center)[
  #text(size: 20pt, weight: "bold")[Week 42: From Backpropagation to Differential Equations]
  #v(2pt)
  #text(size: 14pt)[All derivations, written out]
  #v(2pt)
  FYS-STK3155/4155 · Monday lecture
]

#v(1em)
*Notation* (as on the slides). One sample $bold(x)$, layers $l = 1, dots, L$,
$bold(a)^0 = bold(x)$:
$ bold(z)^l = bold(W)^l bold(a)^(l-1) + bold(b)^l, quad bold(a)^l = f(bold(z)^l), $
$bold(W)^l$ of shape (nodes out, nodes in). The last layer has the output
activation $f_L$. For one node:
$z^l_j = sum_i w^l_(j i) a^(l-1)_i + b^l_j$. The error of node $j$ in layer
$l$ is
$ delta^l_j = (partial C)/(partial z^l_j). $
$#had$ is the element-wise product. Boxes marked
#text(fill: boardblue, weight: "bold")[On the board] are the three
derivations the slides leave to the board.

#outline(depth: 1, indent: auto)

#pagebreak()

// ===================================================================
= Backpropagation: the chain rule, organised

#board[Chain rule]

== Step 1: the output layer
$C$ depends on $z^L_j$ only through $a^L_j = f_L (z^L_j)$:
$ delta^L_j = (partial C)/(partial a^L_j) (dif a^L_j)/(dif z^L_j)
  = f'_L (z^L_j) (partial C)/(partial a^L_j). $

== Step 2: one layer back
$z^l_j$ affects $C$ only through $a^l_j$, and $a^l_j$ goes into _every_
node of the next layer,
$ z^(l+1)_k = sum_j w^(l+1)_(k j) a^l_j + b^(l+1)_k
  quad => quad (partial z^(l+1)_k)/(partial a^l_j) = w^(l+1)_(k j). $
The chain rule sums over all these paths:
$ delta^l_j
  = sum_k underbrace((partial C)/(partial z^(l+1)_k), delta^(l+1)_k)
          underbrace((partial z^(l+1)_k)/(partial a^l_j), w^(l+1)_(k j))
          underbrace((dif a^l_j)/(dif z^l_j), f'(z^l_j))
  = (sum_k w^(l+1)_(k j) delta^(l+1)_k) f'(z^l_j). $
The error of node $j$ comes back along every edge it feeds.

== Step 3: the gradients
$z^l_j = sum_i w^l_(j i) a^(l-1)_i + b^l_j$ depends on $w^l_(j i)$ with
slope $a^(l-1)_i$ and on $b^l_j$ with slope $1$:
$ (partial C)/(partial w^l_(j i))
    = (partial C)/(partial z^l_j) (partial z^l_j)/(partial w^l_(j i))
    = delta^l_j a^(l-1)_i,
  quad
  (partial C)/(partial b^l_j) = delta^l_j. $
Weight gradient $=$ error at the end of the edge $times$ signal at its start.

== Step 4: matrix form
In step 2 the sum runs over the _first_ index of $w^(l+1)_(k j)$, which is a
multiplication by the transpose. Step 3 for all $j, i$ at once is an outer
product:
$ bold(delta)^L &= f'_L (bold(z)^L) #had (partial C)/(partial bold(a)^L), quad &
  bold(delta)^l &= ((bold(W)^(l+1))^T bold(delta)^(l+1)) #had f'(bold(z)^l), \
  (partial C)/(partial bold(W)^l) &= bold(delta)^l (bold(a)^(l-1))^T, quad &
  (partial C)/(partial bold(b)^l) &= bold(delta)^l. $
Shape check: $bold(delta)^l (bold(a)^(l-1))^T$ is
$(n_l times 1)(1 times n_(l-1)) = (n_l, n_(l-1))$, the shape of $bold(W)^l$.

Once $bold(delta)^(l+1)$ is known, $bold(delta)^l$ costs one matrix-vector
product. That is why we go _backwards_, and why the whole gradient costs
about as much as one extra forward pass.

== Step 5 (optional): unroll, and the vanishing gradient
Follow one path from the output back to layer 1. Each layer contributes one
factor $w f'$:
$ delta^1 prop delta^L product_(l=2)^L w^l f'(z^(l-1)). $
With the sigmoid, $f' <= 1/4$, so the $f'$ factors alone shrink the error by
at least $(1/4)^(L-1)$.

#result[
  $ delta^L_j = f'_L (z^L_j) (partial C)/(partial a^L_j), quad
    delta^l_j = (sum_k w^(l+1)_(k j) delta^(l+1)_k) f'(z^l_j), quad
    (partial C)/(partial w^l_(j i)) = delta^l_j a^(l-1)_i, quad
    (partial C)/(partial b^l_j) = delta^l_j $
]


// ===================================================================
= The general case: a batch of samples

Stack the $n$ samples as *rows*: $bold(A)^0 = bold(X)$ of shape $(n, n_0)$,
and store $bold(W)^l$ as (in, out), shape $(n_(l-1), n_l)$:
$ bold(Z)^l = bold(A)^(l-1) bold(W)^l + bold(b)^l, quad
  bold(A)^l = f(bold(Z)^l), $
with $bold(b)^l$ added to every row. In components, for sample $s$:
$ Z^l_(s j) = sum_i A^(l-1)_(s i) W^l_(i j) + b^l_j. $
The cost is the batch mean $C = 1/n sum_s C_s$, and
$Delta^l_(s j) = partial C \/ partial Z^l_(s j)$. The $1/n$ sits in
$partial C \/ partial bold(A)^L$.

*Backwards.* Sample $s$ in layer $l+1$ depends only on sample $s$ in layer
$l$: $Z^(l+1)_(s k) = sum_j A^l_(s j) W^(l+1)_(j k) + b^(l+1)_k$. So
$ (partial C)/(partial A^l_(s j))
  = sum_k Delta^(l+1)_(s k) W^(l+1)_(j k)
  = (bold(Delta)^(l+1) (bold(W)^(l+1))^T)_(s j), $
and multiplying by $f'(Z^l_(s j))$ gives
$bold(Delta)^l = (bold(Delta)^(l+1) (bold(W)^(l+1))^T) #had f'(bold(Z)^l)$.

*Weights.* $W^l_(i j)$ appears in $Z^l_(s j)$ for _every_ sample $s$, with
slope $A^(l-1)_(s i)$:
$ (partial C)/(partial W^l_(i j))
  = sum_s Delta^l_(s j) A^(l-1)_(s i)
  = ((bold(A)^(l-1))^T bold(Delta)^l)_(i j). $
The matrix product does the sum over the samples.

*Biases.* $b^l_j$ appears in every row with slope 1:
$partial C \/ partial b^l_j = sum_s Delta^l_(s j)$, the sum of the rows.

#result[
  $ bold(Delta)^L &= f'_L (bold(Z)^L) #had (partial C)/(partial bold(A)^L), &
    bold(Delta)^l &= (bold(Delta)^(l+1) (bold(W)^(l+1))^T) #had f'(bold(Z)^l), \
    (partial C)/(partial bold(W)^l) &= (bold(A)^(l-1))^T bold(Delta)^l, &
    (partial C)/(partial bold(b)^l) &= sum_s bold(Delta)^l_(s,:) $
  Shapes: $bold(Delta)^l$ is $(n, n_l)$;
  $(bold(A)^(l-1))^T bold(Delta)^l$ is $(n_(l-1), n_l)$, the shape of
  $bold(W)^l$. These are the one-sample equations, transposed.
]

// ===================================================================
= Quick question 1: why not start at zero?

All weights and biases are zero.

- *Hidden errors vanish.* $bold(W)^(l+1) = bold(0)$ gives
  $bold(delta)^l = ((bold(W)^(l+1))^T bold(delta)^(l+1)) #had f'(bold(z)^l) = bold(0)$
  for every hidden layer, whatever $f$ is. No hidden weight or bias gets a
  gradient. Only the output layer can move.
- *If $f(0) = 0$* (tanh, ReLU): every hidden $bold(a)^l = f(bold(0)) = bold(0)$,
  so even $partial C \/ partial bold(W)^L = bold(delta)^L (bold(a)^(L-1))^T = bold(0)$.
  Only the output bias moves: the network learns a constant and never leaves
  $bold(W) = bold(0)$.
- *If $f(0) != 0$* (sigmoid): $bold(W)^L$ moves first, then layer $L-1$ in the
  next step, and so on: learning creeps backwards one layer per step, with
  gradients that are products of tiny weights.
- *Symmetry.* All units in a layer start identical, get identical gradients
  and stay copies of each other. The same holds for any equal non-zero start.
  Random weights break the symmetry.


// ===================================================================
= Activation functions and their derivatives

*Sigmoid* $sigma(z) = 1/(1 + e^(-z))$:
$ sigma'(z) = e^(-z)/(1 + e^(-z))^2
  = 1/(1 + e^(-z)) dot e^(-z)/(1 + e^(-z))
  = sigma(z) (1 - sigma(z)). $
$sigma(1-sigma)$ is a parabola in $sigma in (0,1)$ with maximum at
$sigma = 1/2$ (that is $z = 0$), so $sigma' <= 1/4$.

*tanh*: $tanh z = (e^z - e^(-z))/(e^z + e^(-z))$, and with the quotient
rule
$ tanh' z = ((e^z + e^(-z))^2 - (e^z - e^(-z))^2)/(e^z + e^(-z))^2
  = 1 - tanh^2 z <= 1, $
with equality at $z = 0$. Differentiating once more:
$tanh'' z = -2 tanh z dot tanh' z$.

*ReLU* $f(z) = max(0, z)$: $f'(z) = 1$ for $z > 0$, $0$ for $z < 0$, and
$f''(z) = 0$ everywhere except at the kink.

*Why non-linear?* If $f(z) = z$, then
$bold(a)^2 = bold(W)^2 (bold(W)^1 bold(x) + bold(b)^1) + bold(b)^2
 = underbrace(bold(W)^2 bold(W)^1, bold(W)) bold(x)
 + underbrace(bold(W)^2 bold(b)^1 + bold(b)^2, bold(b))$:
any number of layers collapses to one linear map.

// ===================================================================
= Initialisation: the variance of $z$

$bold(z) = bold(W) bold(a) + bold(b)$ with $M$ inputs. The entries of
$bold(W)$ and $bold(b)$ are independent, with zero mean and variances
$sigma_w^2$ and $sigma_b^2$, and independent of $bold(a)$. Component by
component:
$ bb(E)[z_j z_k]
  = sum_(i, m) bb(E)[w_(j i) w_(k m)] bb(E)[a_i a_m] + bb(E)[b_j b_k]
  = delta_(j k) (sigma_w^2 sum_i bb(E)[a_i^2] + sigma_b^2), $
since $bb(E)[w_(j i) w_(k m)] = sigma_w^2$ only if $j = k$ and $i = m$ (cross
terms between $bold(W)$ and $bold(b)$ vanish because the means are zero).
With $overline(a^2) = 1/M sum_i bb(E)[a_i^2]$:
$ bb(E)[bold(z) bold(z)^T] = (M sigma_w^2 overline(a^2) + sigma_b^2) bold(I),
  quad
  op("Var")(z) = M sigma_w^2 overline(a^2) quad (bold(b) = bold(0)). $

*tanh.* Near zero $a = tanh z approx z$, so $overline(a^2) approx op("Var")(z^"prev")$
and each layer multiplies the variance by $M sigma_w^2$. To keep it constant:
$sigma_w^2 = 1\/M$.

*ReLU.* If $z^"prev"$ is symmetric around zero, $a = max(0, z^"prev")$ keeps
only the positive half:
$ overline(a^2) = bb(E)[z^2 ; z > 0] = 1/2 bb(E)[z^2] = 1/2 op("Var")(z^"prev")
  quad => quad op("Var")(z) = (M sigma_w^2)/2 op("Var")(z^"prev"). $
Constant variance needs $sigma_w^2 = 2\/M$ (He).

*ReLU with the tanh rule* $sigma_w^2 = 1\/M$: the variance halves in every
layer, so the standard deviation shrinks by $sqrt(2)$. The first layer sees
the raw input, so after 10 layers there are 9 such factors:
$2^(-9\/2) approx 1\/23$. After 50 layers:
$2^(-49\/2) approx 4 times 10^(-8)$.


// ===================================================================
= Output activation and cost: $bold(delta)^L = bold(a)^L - bold(y)$

In every case $delta^L = f'_L (z^L) dot partial C \/ partial a^L$ (one
sample).

*Regression: linear output, squared error.* $a = z$, $C = 1/2 (a - y)^2$:
$ f'_L = 1, quad (partial C)/(partial a) = a - y
  quad => quad delta^L = a - y. $

*Binary: sigmoid output, cross entropy.* $a = sigma(z)$,
$C = -y ln a - (1-y) ln(1-a)$:
$ (partial C)/(partial a) = -y/a + (1-y)/(1-a)
  = (-y(1-a) + (1-y) a)/(a(1-a)) = (a - y)/(a(1-a)), $
$ delta^L = underbrace(a(1-a), sigma'(z)) dot (a - y)/(a(1-a)) = a - y. $

*$K$ classes: softmax, cross entropy.* $a_k = e^(z_k) \/ sum_l e^(z_l)$,
$C = -sum_k y_k ln a_k$, $bold(y)$ one-hot ($sum_k y_k = 1$). Softmax is not
element-wise, so we need the Jacobian:
$ (partial a_k)/(partial z_j)
  = (delta_(k j) e^(z_k) sum_l e^(z_l) - e^(z_k) e^(z_j))/(sum_l e^(z_l))^2
  = a_k (delta_(k j) - a_j). $
Then
$ delta_j = sum_k (partial C)/(partial a_k) (partial a_k)/(partial z_j)
  = sum_k (-y_k/a_k) a_k (delta_(k j) - a_j)
  = -y_j + a_j sum_k y_k = a_j - y_j. $

#result[
  For all three matched pairs $f'_L$ cancels against
  $partial C \/ partial a$:
  $ bold(delta)^L = bold(a)^L - bold(y) quad "(prediction minus target)." $
  Each cost is the negative log-likelihood of a Gaussian, Bernoulli or
  categorical distribution.
]

// ===================================================================
= A mismatched pair: sigmoid $+$ squared error

Sigmoid output, target $y = 1$.

$ "cross entropy:" quad & delta^L_"CE" = a - 1, \
  "squared error:" quad & delta^L_"SE" = (partial C)/(partial a) sigma'(z)
    = (a - 1) underbrace(a(1-a), sigma'(z)). $

The ratio is $delta^L_"SE" \/ delta^L_"CE" = sigma'(z)$, and
$sigma'(z) -> 0$ as $a -> 0$, which is exactly when the prediction is most
wrong. At $z = -5$:
$ a = sigma(-5) approx 0.00669, quad sigma'(-5) = a(1-a) approx 0.00665
  approx 1/150. $
$delta^L_"SE"$ is about 150 times smaller than $delta^L_"CE"$: the more
confidently wrong, the _less_ it learns.

// ===================================================================
= Quick question 2: which image trains the network?

Softmax $+$ cross entropy, so $bold(delta)^L = bold(a)^L - bold(y)$.

- *A, label 0:* $bold(a) approx bold(y) = (1, 0, dots, 0)$, so
  $bold(delta)^L approx bold(0)$. Nothing to learn.
- *B, label 1:* $bold(y) = (0, 1, 0, dots, 0)$.
  $delta_1 = a_1 - 1 = 0.24 - 1 = -0.76$, the only negative component; every
  other $delta_k = a_k - 0 = a_k > 0$.

B changes the weights more: raise the score of class 1, lower the rest.


// ===================================================================
= Universal approximation: hats from ReLU units

== A hat from three ReLUs (optional board work)
One unit $v op("ReLU")(w x + b)$ is zero, then a straight line: one kink at
$x = -b\/w$. Add three of them,
$ h(x) = op("ReLU")(x) - 2 op("ReLU")(x - 1) + op("ReLU")(x - 2), $
interval by interval:
$ h(x) = cases(
  0 & quad x <= 0,
  x & quad 0 <= x <= 1,
  x - 2(x-1) = 2 - x & quad 1 <= x <= 2,
  x - 2(x-1) + (x-2) = 0 & quad x >= 2,
) $
A hat with peak $1$ at $x = 1$. Shifted and scaled, with nodes spaced by
$h$:
$ "hat"_i (x) = 1/h [op("ReLU")(x - x_(i-1)) - 2 op("ReLU")(x - x_i)
  + op("ReLU")(x - x_(i+1))], $
which is $1$ at $x_i$ and $0$ at all other nodes. So
$G(x) = sum_i F(x_i) "hat"_i (x)$ connects the dots $(x_i, F(x_i))$ with
straight lines: a one-hidden-layer ReLU network.

== How accurate? (the $L(b-a)\/(2N)$ bound)
Let $|F'| <= L$ on $[a, b]$, split into $N$ intervals of length
$h = (b-a)\/N$. On $[x_i, x_(i+1)]$ write $x = x_i + t h$, $t in [0, 1]$.
Then $G(x) = (1-t) F(x_i) + t F(x_(i+1))$, and
$ F(x) - G(x) = (1-t)(F(x) - F(x_i)) + t (F(x) - F(x_(i+1))). $
By the mean value theorem $|F(x) - F(x_i)| <= L t h$ and
$|F(x) - F(x_(i+1))| <= L (1-t) h$, so
$ |F(x) - G(x)| <= 2 t (1-t) L h <= (L h)/2 = (L (b-a))/(2 N). $
Doubling the units halves the error bound.

== Why not a polynomial?
If $f$ is a polynomial of degree $p$, then each $f(bold(w)_j^T bold(x) + b_j)$
is a polynomial of degree at most $p$ in $bold(x)$, and so is the sum
$G(bold(x)) = sum_j v_j f(bold(w)_j^T bold(x) + b_j)$, for _any_ $N$.
Polynomials of bounded degree cannot approximate, say, $x^(p+1)$ or
$e^x$ arbitrarily well.

#pagebreak()

// ===================================================================
= Exponential decay

#board[Exponential decay]

Problem: $g'(x) + alpha g(x) = 0$, $g(0) = g_0$, with $alpha = 2$,
$g_0 = 10$. Exact solution $g = g_0 e^(-alpha x)$.

+ *Trial solution.* We want $g_t (0) = g_0$ _whatever_ the network does:
  $ g_t (x) = g_0 + x N(x). $
  At $x = 0$ the network term vanishes. (Recipe $g_t = h + B N$ with $h = g_0$,
  $B = x$.)
+ *Derivative* (product rule):
  $ g'_t (x) = N(x) + x N'(x). $
+ *Residual:*
  $ r(x) = g'_t + alpha g_t = N + x N' + alpha (g_0 + x N). $
  $r = 0$ for the exact solution.
+ *Cost:* mean squared residual at collocation points $x_i in [0, 1]$ that we
  choose:
  $ C = 1/n sum_(i=1)^n r(x_i)^2. $
  Training changes the weights in $N$ to make $C$ small.
+ *What is new:* $N'(x)$ is a derivative with respect to the _input_, not the
  weights. Backpropagation for the weights is unchanged; it is regression
  with target $r = 0$.

#result[
  $ g_t = g_0 + x N(x), quad
    r = underbrace(N + x N', g'_t) + alpha (g_0 + x N) $
  Two hidden layers of 40 tanh units, 50 collocation points: max error
  $3 times 10^(-3)$.
]

== Same recipe: logistic growth
$g' - 2 g(1-g) = 0$, $g(0) = 1.2$. Trial solution $g_t = 1.2 + t N(t)$,
$ r = N + t N' - 2 g_t (1 - g_t). $
Exact solution (separable equation):
$ g(t) = 1/(1 + (1\/g_0 - 1) e^(-2 t)) = 1/(1 - e^(-2t)\/6). $
The residual is non-linear in $N$; nothing else changes.


// ===================================================================
#pagebreak()

= Second derivatives: why ReLU fails

#board[Second derivatives]

Poisson $g'' + f = 0$, diffusion $partial_t g - partial_x^2 g = 0$ and wave
$partial_t^2 g - c^2 partial_x^2 g = 0$ all contain a second derivative, so we
need $N''$. One hidden layer:

+ $ N(x) = sum_j v_j f(w_j x + b_j) + c $
+ Chain rule, $w_j$ comes out:
  $ N'(x) = sum_j v_j w_j f'(w_j x + b_j) $
+ Once more:
  $ N''(x) = sum_j v_j w_j^2 f''(w_j x + b_j) $
+ ReLU: $f'' = 0$ everywhere (except at the kink), so $N'' = 0$ for *any*
  weights.

This is not about training: a ReLU network is piecewise linear, so it has no
curvature at all. More layers do not help: compositions of piecewise-linear
functions are piecewise linear.

#result[
  $ N''(x) = sum_j v_j w_j^2 f''(w_j x + b_j) $
  *ReLU*: $f'' = 0$, so $N'' = 0$; it cannot solve $g'' + f = 0$. \
  *tanh*: smooth, and both derivatives come from $f$ itself:
  $f' = 1 - f^2$, $f'' = -2 f f'$.
]

$max |N''|$ of a randomly initialised network: tanh $2.9$, GELU $1.2$,
sigmoid $0.06$, ReLU $0$.

// ===================================================================
= Boundary conditions built into the trial solution

Recipe: $g_t = h(x) + B(x) N(x)$, with $h$ satisfying the conditions and
$B = 0$ wherever a condition is imposed.

*$g(0) = a$, $g(1) = b$:* $g_t = a + (b - a) x + x(1-x) N$.
$ g_t (0) = a + 0 + 0 = a, quad g_t (1) = a + (b - a) + 0 = b. $

*$g(0) = g_0$, $g'(0) = v_0$:* $g_t = g_0 + v_0 x + x^2 N$.
$ g_t (0) = g_0, quad g'_t = v_0 + 2 x N + x^2 N' => g'_t (0) = v_0. $

*$g(0) = a$, $g'(1) = b$ (needs a trick):*
$g_t = a + b x + x (N(x) - N(1) - N'(1))$.
$ g_t (0) = a, quad
  g'_t = b + (N(x) - N(1) - N'(1)) + x N'(x)
  => g'_t (1) = b + 0 - N'(1) + N'(1) = b. $

== Poisson residual
$g'' + (3x + x^2) e^x = 0$, $g(0) = g(1) = 0$, trial solution
$g_t = x(1-x) N$:
$ g'_t &= (1 - 2x) N + x(1-x) N', \
  g''_t &= -2 N + 2(1 - 2x) N' + x(1-x) N''. $
The residual $r = g''_t + (3x + x^2) e^x$ contains $N''$, so the previous
section applies. Exact solution $g = x(1-x) e^x$; check:
$ g' = (1 - x - x^2) e^x, quad
  g'' = (-1 - 2x) e^x + (1 - x - x^2) e^x = -(3x + x^2) e^x. $


// ===================================================================
= Standard methods, for comparison

Grid $t_n = n Delta t$, $g_n approx g(t_n)$, equation $g' = F(t, g)$.

*Forward Euler:* follow the slope for one step,
$ g_(n+1) = g_n + Delta t F(t_n, g_n). quad "Error" prop Delta t. $

*RK4:* four slope evaluations per step,
$ k_1 &= F(t_n, g_n), & k_2 &= F(t_n + Delta t\/2, g_n + Delta t k_1\/2), \
  k_3 &= F(t_n + Delta t\/2, g_n + Delta t k_2\/2), & k_4 &= F(t_n + Delta t, g_n + Delta t k_3), $
$ g_(n+1) = g_n + (Delta t)/6 (k_1 + 2 k_2 + 2 k_3 + k_4). quad
  "Error" prop Delta t^4. $

*Finite differences* for $g'' + f = 0$: Taylor expand around $x_i$,
$ g_(i plus.minus 1) = g_i plus.minus Delta x g'_i + (Delta x^2)/2 g''_i
  plus.minus (Delta x^3)/6 g'''_i + O(Delta x^4)
  => (g_(i+1) - 2 g_i + g_(i-1))/(Delta x^2) = g''_i + O(Delta x^2). $
With $g_0 = g_(N) = 0$ this is a tridiagonal linear system
$-g_(i-1) + 2 g_i - g_(i+1) = Delta x^2 f(x_i)$. Error $prop Delta x^2$.

#table(
  columns: 4, align: (left, left, right, right), stroke: none,
  table.hline(),
  table.header([], [method], [max error], [time]),
  table.hline(stroke: 0.5pt),
  [logistic], [Euler, $Delta t = 0.01$], [$9 times 10^(-4)$], [0.04 ms],
  [], [RK4, $Delta t = 0.01$], [$2 times 10^(-10)$], [0.14 ms],
  [], [network], [$1 times 10^(-4)$], [0.5 s],
  table.hline(stroke: 0.5pt),
  [Poisson], [FD, $Delta x = 0.01$], [$2 times 10^(-5)$], [0.12 ms],
  [], [network], [$1 times 10^(-5)$], [0.6 s],
  table.hline(),
)

// ===================================================================
= Diffusion: a trial solution in two variables

$ partial_t g - partial_x^2 g = 0, quad g(0, t) = g(1, t) = 0, quad
  g(x, 0) = sin pi x. $

*Exact solution:* $g = e^(-pi^2 t) sin pi x$. Check:
$partial_t g = -pi^2 g$ and $partial_x^2 g = -pi^2 g$, so the residual is
zero; $g(x,0) = sin pi x$ and $sin 0 = sin pi = 0$.

*Trial solution:*
$ g_t = (1 - t) sin pi x + x(1-x) t N(x, t). $
- $t = 0$: $g_t = sin pi x$ (the $N$ term has the factor $t$).
- $x = 0$ or $x = 1$: $sin pi x = 0$ and $x(1-x) = 0$, so $g_t = 0$.

*Derivatives for the residual:*
$ partial_t g_t &= -sin pi x + x(1-x) (N + t partial_t N), \
  partial_x^2 g_t &= -(1-t) pi^2 sin pi x
    + t [-2 N + 2(1 - 2x) partial_x N + x(1-x) partial_x^2 N], $
and $r = partial_t g_t - partial_x^2 g_t$ at collocation points
$(x_i, t_i)$. Again $partial_x^2 N$ appears: no ReLU.


// ===================================================================
= Soft constraints: conditions as penalties

Let the network itself be the solution, $u(x, t) approx N(x, t)$, and put
each condition into the cost:
$ L = L_"PDE" + lambda (L_"IC" + L_"BC"), $
$ L_"PDE" &= 1/n sum_i (partial_t N - partial_x^2 N)^2_i
    && "inside the domain", \
  L_"IC" &= 1/n sum_i (N(x_i, 0) - sin pi x_i)^2 && "at" t = 0, \
  L_"BC" &= 1/n sum_i (N(0, t_i)^2 + N(1, t_i)^2) && "at" x = 0, 1. $

Each term that the trial solution enforced exactly is now a penalty. A new
condition is a new term; $lambda$ weighs the terms, like the penalty in Ridge.

#table(
  columns: 3, align: (left, right, right), stroke: none,
  table.hline(),
  table.header([], [RMSE], [error at $t = 0$]),
  table.hline(stroke: 0.5pt),
  [hard ($g_t$)], [$1.8 times 10^(-3)$], [0 exactly],
  [soft, $lambda = 10$], [$6.5 times 10^(-3)$], [$6 times 10^(-3)$],
  table.hline(),
)

// ===================================================================
= Quick question 3: a ReLU PINN

Same soft cost, but $u = N(x, t)$ is a ReLU network.

+ ReLU gives $partial_x^2 N = 0$ at every sample, so
  $ L_"PDE" = 1/n sum_i (partial_t N)_i^2. $
+ The cheapest way to make all three terms small: $partial_t N = 0$ (no time
  dependence), $N(x, 0) approx sin pi x$ (IC), and $N(0, t) = N(1, t) = 0$
  (BC, already true for $sin pi x$). So
  $ u approx sin pi x quad "for all" t. $
  (Piecewise linear in $x$, but close to $sin pi x$.)
+ The true solution decays as $e^(-pi^2 t) sin pi x$, so the network never
  diffuses: small cost ($6 times 10^(-4)$), wrong answer (RMSE $0.65$ vs.
  $1.8 times 10^(-3)$).

*A small residual is not a solution.* Use smooth activations, and check
against something you know.

// ===================================================================
= Inverse problems: an unknown parameter

$partial_t u - D partial_x^2 u = 0$ with $D$ unknown, no initial or boundary
conditions, only 40 noisy measurements $y_i$. Treat $D$ as one more trainable
parameter:
$ L(N, D) = 1/n sum_j underbrace((partial_t N - D partial_x^2 N)_j, r_j)^2
  + lambda 1/40 sum_(i=1)^40 (N(x_i, t_i) - y_i)^2. $
Its gradient with respect to $D$ is one line:
$ (partial L)/(partial D) = -2/n sum_j r_j (partial_x^2 N)_j, $
and gradient descent updates $D$ together with the weights. No trial solution
is possible: the conditions are unknown, so soft constraints are the only
option.

True $D = 0.5$; three starting guesses all give $D approx 0.51$. With five
times more noise, $D = 0.69$: a recovered parameter needs an error bar (for
example from the bootstrap).
