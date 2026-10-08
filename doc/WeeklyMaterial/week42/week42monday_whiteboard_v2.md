# Week 42 Monday: whiteboard notes (v2)

Lecturer notes for `week42monday_lecture_v2.tex`.

**How slides and board split the work**
- The **slide** sets up the question; a blue box "On the board (Wn)" says what to derive.
- You derive it **on the board** (sections below).
- **Click once**: the slide shows the result. That is what students keep, so the board work and the slide never repeat each other.
- Anything marked *optional* is only for when time allows.

**Notation** (same as the exercises): one sample, $z^l = W^l a^{l-1} + b^l$, $a^l = f(z^l)$.

## Time plan

| Min | Slides | Board |
|---|---|---|
| 0–3 | title, plan | |
| 3–6 | forward pass | |
| 6–18 | four equations of backpropagation | **W1** |
| 18–21 | hidden activations | |
| 21–27 | initialisation (2 slides) | |
| 27–35 | output ↔ cost | **W2** |
| 35–39 | mismatched pair, MNIST question | |
| 39–45 | UAT, kinks | **W3** |
| *break* | | |
| 0–4 | motivation, equation as the cost | |
| 4–11 | exponential decay | **W4** |
| 11–17 | activation matters | **W5** |
| 17–25 | boundary conditions | **W6** |
| 25–28 | results vs standard methods | |
| 28–31 | diffusion | |
| 31–36 | soft constraints | |
| 36–39 | inverse problems | |
| 39–45 | pros/cons, summary | |

If time runs short, skip the shape check in W1 and the optional parts.

---

## W1: The four equations of backpropagation (≈ 12 min)

**On the slide already:** the goal ($\partial C/\partial w^l_{jk}$, $\partial C/\partial b^l_j$) and the definition of the error $\delta^l_j = \partial C/\partial z^l_j$.

**On the board.** Draw three layers $l-1$, $l$, $l+1$. Highlight one node $j$ in layer $l$, with one arrow coming in from node $k$ in layer $l-1$ and arrows going out to *all* nodes in layer $l+1$.

1. **Components.** $z^l_j = \sum_k w^l_{jk}\, a^{l-1}_k + b^l_j$ and $a^l_j = f(z^l_j)$.
2. **Output layer.** $C$ depends on $z^L_j$ only through $a^L_j$:
   $$\delta^L_j = \frac{\partial C}{\partial a^L_j}\,\frac{\partial a^L_j}{\partial z^L_j} = \frac{\partial C}{\partial a^L_j}\, f'(z^L_j).$$
3. **One layer back.** Point at the picture: $z^l_j$ changes $a^l_j$, which changes **every** $z^{l+1}_k$. The chain rule sums over all these paths:
   $$\delta^l_j = \sum_k \underbrace{\frac{\partial C}{\partial z^{l+1}_k}}_{\delta^{l+1}_k}\, \underbrace{\frac{\partial z^{l+1}_k}{\partial a^l_j}}_{w^{l+1}_{kj}}\, \underbrace{\frac{\partial a^l_j}{\partial z^l_j}}_{f'(z^l_j)} .$$
   The sum runs over the *first* index of $w$, so in matrix form it is the transpose:
   $$\boldsymbol\delta^l = \big((W^{l+1})^T\boldsymbol\delta^{l+1}\big)\circ f'(\mathbf z^l).$$
4. **Gradients.** $z^l_j$ depends on $b^l_j$ with slope 1 and on $w^l_{jk}$ with slope $a^{l-1}_k$:
   $$\frac{\partial C}{\partial b^l_j} = \delta^l_j, \qquad \frac{\partial C}{\partial w^l_{jk}} = \delta^l_j\, a^{l-1}_k \;\Rightarrow\; \frac{\partial C}{\partial W^l} = \boldsymbol\delta^l (\mathbf a^{l-1})^T .$$
5. **Shape check** (optional). $W^l$ is (out, in) $= (M_l, M_{l-1})$. $\boldsymbol\delta^l$ is $(M_l)$ and $(\mathbf a^{l-1})^T$ is $(1, M_{l-1})$, so the product is $(M_l, M_{l-1})$ ✓.

**Click:** the four equations and the take-home bullets.

**Say:** once $\boldsymbol\delta^{l+1}$ is known, $\boldsymbol\delta^l$ costs one matrix-vector product. That is why we go *backwards*, and why the whole gradient costs about as much as one extra forward pass. Tomorrow's session does exactly this in code.

---

## Initialisation (slides only, ≈ 6 min)

There is no board derivation here; talk through the two slides.

- **Slide 1, symmetry:** ask *"what if all weights start at zero?"* All units in a layer get the same input, the same output and the same gradient, so they stay copies of each other forever.
- **Slide 1, scale:** point at $\operatorname{Var}(z) \approx M\sigma_w^2\,\overline{a^2}$. A node sums $M$ random terms, so the variance grows with $M$, and the weights must shrink like $1/\sqrt{M}$ to compensate.
- **Slide 1, link to activations:** tanh and sigmoid want $z$ near 0, where $f' \approx$ its maximum. ReLU zeroes half its inputs, so it needs twice the variance (He).
- **Slide 2, link to gradients:** backprop multiplies by $W^T$ and $f'$ in every layer (W1). Too large weights make the gradients *grow* layer by layer (middle panel: $10^4$ times larger in layer 1 than in layer 10). Too small weights make them tiny everywhere. With $1/M$ they stay the same size.
- In the exercises/Project 2: `create_layers` already uses $1/M$ (or $2/M$ for ReLU).

---

## W2: Output activation ↔ cost (≈ 9 min)

**On the slide already:** the table of the three pairs (the $\delta^L$ column is hidden).

**On the board**, using $\delta^L = f'(z)\,\partial C/\partial a$ from W1:

1. *Linear + squared error* (warm-up, 30 s): $f' = 1$, $\partial C/\partial a = a - y$, so $\delta = a - y$.
2. *Sigmoid + cross entropy.*
   - Derive $\sigma'$:
     $$\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \underbrace{\frac{1}{1+e^{-z}}}_{a}\cdot\underbrace{\frac{e^{-z}}{1+e^{-z}}}_{1-a} = a(1-a).$$
   - Differentiate the cost:
     $$\frac{\partial C}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a} = \frac{a - y}{a(1-a)}.$$
   - Multiply: $\delta = a(1-a)\cdot\dfrac{a-y}{a(1-a)} = a - y$. **Circle the cancellation.**
3. *Sigmoid + squared error* (sets up the next slide): $\delta = a(1-a)\,(a - y)$. Nothing cancels.

**Click:** the δ column fills in, and the boxed $\delta^L = a^L - y$ appears.

**Next slide** has the figure and the numbers ($z=-5$: 150 times smaller). Don't compute them on the board.

*Optional (if a student asks about softmax):*
$$\frac{\partial a_k}{\partial z_j} = a_k(\delta_{kj}-a_j),\qquad \delta_j = \sum_k\Big(-\frac{y_k}{a_k}\Big)a_k(\delta_{kj}-a_j) = -y_j + a_j\sum_k y_k = a_j - y_j.$$

---

## W3: A hat from three ReLUs (≈ 4 min)

**On the board:**

1. Draw the three pieces: $\mathrm{ReLU}(x)$, $-2\,\mathrm{ReLU}(x-1)$, $\mathrm{ReLU}(x-2)$.
2. Add them interval by interval:
   - $x\le0$: $0$
   - $[0,1]$: $x$
   - $[1,2]$: $x - 2(x-1) = 2 - x$
   - $x\ge2$: $x - 2(x-1) + (x-2) = 0$
   
   This is a hat with peak 1 at $x = 1$.
3. Sketch a curve $F$, place hats at a few points with heights $F(x_i)$, and add them. You get connect-the-dots. More hats, closer to $F$.

**Click:** the hat picture and the trained Runge network (kinks where the function bends).

*Optional (one sentence):* the error is at most $L(b-a)/(2N)$, where $L$ bounds $|F'|$. Doubling the units halves the error bound.

---

## W4: Exponential decay (≈ 6 min)

**On the slide already:** $g' = -\kappa g$, $g(0) = g_0$, and the three ingredients from the previous slide.

**On the board:**

1. Ask: how can $g_t$ equal $g_0$ at $x = 0$ *whatever* $N$ is? Answer: $g_t = g_0 + x\,N(x)$.
2. Product rule: $g_t' = N + x N'$.
3. Residual: $r = g_t' + \kappa g_t = N + xN' + \kappa(g_0 + xN)$.
4. Cost: $C = \frac1n\sum_i r(x_i)^2$ over points $x_i$ in $[0,1]$ that we choose. Training changes the weights in $N$ to make $C$ small.
5. Point at $N'$: this needs the derivative of the network with respect to its *input*.

**Click:** the result and the figure.

---

## W5: Why ReLU fails for second derivatives (≈ 4 min)

**On the board:**

1. $N(x) = \sum_j v_j f(w_jx + b_j) + c$.
2. $N'(x) = \sum_j v_j w_j f'(w_jx + b_j)$ (chain rule, $w_j$ comes out).
3. $N''(x) = \sum_j v_j w_j^2 f''(w_jx + b_j)$.
4. Ask: what is $f''$ for ReLU? It is $0$ everywhere (except at the kink). So $N'' = 0$ for **any** weights.

**Click:** the ReLU box, the table, and "we use tanh".

**Say:** this is not about training. A ReLU network is piecewise linear, so it *has* no curvature. More layers don't help either: compositions of piecewise-linear functions are piecewise linear.

---

## W6: Trial solutions for boundary conditions (≈ 8 min)

Make it interactive: write the conditions, let students propose $g_t$ for 1 minute, then check.

1. $g(0) = g(1) = 0$: we need a factor that is zero at both ends, so $g_t = x(1-x)N$. Check $x=0$ and $x=1$.
2. $g(0) = a$, $g(1) = b$: $h$ is the straight line through $(0,a)$ and $(1,b)$, so $g_t = a + (b-a)x + x(1-x)N$. Check both ends.
3. $g(0) = g_0$, $g'(0) = v_0$: $g_t = g_0 + v_0x + x^2N$.
   - $g_t(0) = g_0$ ✓
   - $g_t' = v_0 + 2xN + x^2N'$, so $g_t'(0) = v_0$ ✓ (every $N$ term has a factor $x$)
4. Ask: what about $g'(1) = 0$? Let them struggle for a moment. No simple factor works, which motivates soft constraints later.

**Click:** the table and the good/bad boxes.

*Optional:* for the Poisson residual you need $g_t'' = -2N + 2(1-2x)N' + x(1-x)N''$. It contains $N''$, so W5 applies.

---

## Optional questions, if time is left

- **After the PINN slide:** we use a ReLU network $u = N(x,t)$ for $u_t = u_{xx}$ with the soft cost. What happens?
  *Answer:* $N_{xx} = 0$, so the PDE term is $\frac1n\sum N_t^2$. The optimiser makes $N$ independent of $t$: a perfect residual for a solution that does not diffuse. A small residual does not mean a correct solution.
- **After the kinks slide:** for the Runge function, $|F'| \le 3.25$ on $[-1,1]$, and the bound needs $N \ge 325$ units for error 0.01. Training managed with 50. Why?
  *Answer:* the bound assumes equally spaced kinks; training puts them where the function bends.
