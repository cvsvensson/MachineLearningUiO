# Week 42 Monday: whiteboard notes (v2)

Lecturer notes for `week42monday_lecture_v2.tex`.

**How slides and board split the work**
- The **slide** sets up the question; a blue box "On the board (Wn)" says what to derive.
- You derive it **on the board** (sections below).
- **Click once**: the slide shows the result. That is what students keep, so the board work and the slide never repeat each other.
- Anything marked *optional* is only for when time allows.

**What the students already have from week 41 Monday:** the four backpropagation equations, $\delta^L = a - y$ for the three output/cost pairs, vanishing gradients and Xavier/He. So the first hour is a *recap with a new angle*, kept short; the time goes to the second hour.

**Notation** (same as the exercises): one sample, $z^l = W^l a^{l-1} + b^l$, $a^l = f(z^l)$.

## Time plan

About 80 min of content and 10 min of slack (5 per hour). The first hour now ends with the start of Part 4 (motivation, equation as the cost); if you are behind, move those two slides to after the break.

| Min | Slides | Board |
|---|---|---|
| 0–2 | title, plan | |
| 2–4 | forward pass | |
| 4–10 | backpropagation with one node per layer | **W1** |
| 10–14 | many nodes per layer: where the matrices come in | (W1, picture on the slide) |
| 14–16 | hidden activations | |
| 16–19 | **Quick question 1**: all weights zero | |
| 19–24 | initialisation (3 slides) | |
| 24–27 | output ↔ cost (recall from week 41) | |
| 27–30 | mismatched pair | |
| 30–33 | **Quick question 2** (MNIST) | |
| 33–38 | UAT, kinks | **W2** |
| 38–42 | motivation, equation as the cost | |
| 42–45 | *slack* | |
| *break* | | |
| 0–6 | exponential decay | **W3** |
| 6–11 | activation matters | **W4** |
| 11–14 | boundary conditions (table on the slide) | |
| 14–18 | results vs standard methods (**guess first**) | |
| 18–22 | diffusion | |
| 22–27 | soft constraints (PINN) | |
| 27–32 | **Quick question 3**: ReLU PINN | |
| 32–36 | inverse problems | |
| 36–39 | pros/cons | |
| 39–40 | read more, summary | |
| 40–45 | *slack* | |

Compared with the previous plan, backpropagation went from 11 to 10 min (now two slides), the output/cost board work (old W2, 8 min) and the trial-solution board work (old W6, 8 min) are gone, and the time went to diffusion (+2), soft constraints (+1), Quick question 3 (+2) and the inverse problem (+1), plus slack.

If time runs short, skip the optional parts.

---

## W1: Backpropagation with one node per layer (≈ 6 min), then the picture (≈ 4 min)

**On the slide already:** the chain $x \to z^1 \to a^1 \to \cdots \to z^L \to a^L \to C$, all scalars, $z^l = w^l a^{l-1} + b^l$, $a^l = f(z^l)$, and the definition $\delta^l = \partial C/\partial z^l$.

**Say first:** "You derived the four equations last week. Today we do it once more in the simplest possible network, one node per layer, so that every equation is a single chain-rule factor. Then we see where the matrices come from."

**On the board.** Copy the chain, and write each step as *one* chain-rule factor added to the previous one.

1. **Output.** $C$ depends on $z^L$ only through $a^L = f_L(z^L)$:
   $$\delta^L = \frac{\partial C}{\partial z^L} = \frac{\partial C}{\partial a^L}\,\frac{da^L}{dz^L} = f_L'(z^L)\,\frac{\partial C}{\partial a^L}.$$
2. **One layer back.** $z^l$ affects $C$ only through $a^l$, which affects $C$ only through $z^{l+1} = w^{l+1}a^l + b^{l+1}$:
   $$\delta^l = \underbrace{\frac{\partial C}{\partial z^{l+1}}}_{\delta^{l+1}}\,\underbrace{\frac{\partial z^{l+1}}{\partial a^l}}_{w^{l+1}}\,\underbrace{\frac{da^l}{dz^l}}_{f'(z^l)} = \delta^{l+1}\,w^{l+1}\,f'(z^l).$$
3. **Gradients.** $z^l$ depends on $w^l$ with slope $a^{l-1}$ and on $b^l$ with slope 1:
   $$\frac{\partial C}{\partial w^l} = \delta^l\,a^{l-1}, \qquad \frac{\partial C}{\partial b^l} = \delta^l.$$
4. **Unroll** (optional, 30 s, links to vanishing gradients): $\delta^1 = \delta^L \prod_{l=2}^{L} w^l f'(z^{l-1})$. One factor $w f'$ per layer; with sigmoid, $f' \le 1/4$.

**Click:** the four scalar lines, each with its chain-rule factor next to it, and two bullets (go backwards, reusing $\delta^{l+1}$; one factor $wf'$ per layer).

**Next slide (no board, point at the pictures): many nodes per layer.**

- **Left picture.** Node $j$ in layer $l$ feeds *every* node $k$ in layer $l+1$ (highlighted edges). So $z^l_j$ affects $C$ along every edge, and the chain rule sums over them: $\delta^l_j = \sum_k w^{l+1}_{kj}\,\delta^{l+1}_k\,f'(z^l_j)$. The single factor $w^{l+1}$ of the scalar chain becomes a sum. **Point at the indices:** $w_{kj}$ goes *from* $j$ *into* $k$, and the sum runs over the *first* index. That is a multiplication by the transpose, $(W^{l+1})^T\boldsymbol\delta^{l+1}$.
- **Right picture.** One weight $w^l_{jk}$ from $a^{l-1}_k$ into $z^l_j$. As in the scalar chain, $\partial C/\partial w^l_{jk} = \delta^l_j\,a^{l-1}_k$: error at the node it goes *into* × signal from the node it comes *from*. All $j,k$ at once: the outer product $\boldsymbol\delta^l(\mathbf a^{l-1})^T$.
- **Shape check** (on the slide): $W^{l+1}$ is $(n_{l+1}, n_l)$, so $(W^{l+1})^T\boldsymbol\delta^{l+1}$ has $n_l$ entries; $\boldsymbol\delta^l(\mathbf a^{l-1})^T$ is $(n_l, n_{l-1})$, the shape of $W^l$.
- **Click:** the four vector equations, and one line saying the batched code (samples in rows) is the transpose.

**Say:** once $\boldsymbol\delta^{l+1}$ is known, $\boldsymbol\delta^l$ costs one matrix-vector product. That is why we go *backwards*, and why the whole gradient costs about as much as one extra forward pass. Tomorrow's session does exactly this in code.

---

## Quick question 1: all weights zero (≈ 3 min)

Comes right after the hidden-activations slide. Students *apply* the four equations; they don't derive anything. The relevant equations are repeated on the slide.

- Read the question: a tanh network starts with all weights and biases zero. Which gradients are non-zero after the first backward pass?
- 1 minute in pairs, then collect a few answers ("all zero?", "only the last layer?").
- **Click:** only the output bias. Hidden $a^l = \tanh(0) = 0$ kills $\partial C/\partial W^l = \delta^l(a^{l-1})^T$ for $l\ge2$. $\delta^l = (W^{l+1})^T\delta^{l+1} = 0$ kills everything in the hidden layers, including $W^1$. The network can only learn a constant.
- **Say:** with equal but non-zero weights, all units in a layer get the same gradient and stay copies of each other. Random weights break the symmetry. The notebook checks this with `backpropagation`.

## Initialisation (slides only, ≈ 5 min)

There is no board derivation here; talk through the three slides. Symmetry was covered by the question, so start with the scale. Students met Xavier/He last week, so keep it brisk.

- **Slide 1, scale:** point at $\operatorname{Var}(z) \approx M\sigma_w^2\,\overline{a^2}$. A node sums $M$ random terms, so the variance grows with $M$, and the weights must shrink like $1/\sqrt{M}$ to compensate.
- **Slide 1, link to activations:** tanh wants $z$ near 0, where $f' \approx 1$, so $1/M$ works. ReLU zeroes half its inputs, so it needs twice the variance (He). For the sigmoid, $f'\le 1/4$, so even with $1/M$ the gradients shrink layer by layer.
- **Slide 2, both failure modes in a 10-layer tanh network:** left panel forward, right panel backward, layer on the x-axis. Too large ($\sigma_w=1$): units stuck at $\pm1$, and gradients $10^4$ times larger in layer 1 than in layer 10 (backprop multiplies by $W^T$ and $f'$ in every layer, W1). Too small: the signal dies and the gradients are $\sim10^{-7}$ everywhere. With $1/M$ both keep their size.
- **Slide 3, the right scale depends on the activation:** the same network with ReLU and the tanh rule $1/M$ loses a factor $\sqrt2$ per layer, silently (no error, just bad training). Reason in one sentence: ReLU zeroes half its inputs, so it passes on half the variance; He ($2/M$) makes up for it. Take-away: change the activation, check the scale.
- In the exercises/Project 2: `create_layers` already uses $1/M$ (or $2/M$ for ReLU).

## Output activation ↔ cost (slides only, ≈ 3 min)

No board work: students derived $\delta^L = a - y$ last week, and the slide shows the sigmoid cancellation in two lines.

- The table shows the three pairs with the $\delta^L$ column hidden. Ask: "Last week you computed $\delta^L = f_L'(z^L)\,\partial C/\partial a^L$ for these. What did you get?" Take an answer or two.
- **Click:** the column fills in ($a-y$ for all three), and the box "You showed last week: $f_L'$ cancels" with $\sigma' = a(1-a)$ and $\partial C/\partial a = (a-y)/(a(1-a))$. Point at the cancellation.
- The **mismatched-pair** slide that follows shows the two errors, $a-1$ and $(a-1)\,a(1-a)$, with the surviving factor $\sigma'$ highlighted, next to the figure. Point at that factor; don't recompute it on the board.

*Optional (if a student asks about softmax; the result is on the slide, the derivation is here and in the notebook):*
$$\frac{\partial a_k}{\partial z_j} = a_k(\delta_{kj}-a_j),\qquad \delta_j = \sum_k\Big(-\frac{y_k}{a_k}\Big)a_k(\delta_{kj}-a_j) = -y_j + a_j\sum_k y_k = a_j - y_j.$$

---

## W2: A hat from three ReLUs (≈ 4 min)

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

## W3: Exponential decay (≈ 6 min)

**On the slide already:** $g' = -\kappa g$, $g(0) = g_0$, and the three ingredients from the previous slide.

**On the board:**

1. Ask: how can $g_t$ equal $g_0$ at $x = 0$ *whatever* $N$ is? Answer: $g_t = g_0 + x\,N(x)$.
2. Product rule: $g_t' = N + x N'$.
3. Residual: $r = g_t' + \kappa g_t = N + xN' + \kappa(g_0 + xN)$.
4. Cost: $C = \frac1n\sum_i r(x_i)^2$ over points $x_i$ in $[0,1]$ that we choose. Training changes the weights in $N$ to make $C$ small.
5. Point at $N'$: this needs the derivative of the network with respect to its *input*.

**Click:** the result and the figure.

---

## W4: Why ReLU fails for second derivatives (≈ 4 min)

**On the slide already:** the Poisson, diffusion and wave equations written out, $-g''=f$, $\partial_t g = \partial_x^2 g$, $\partial_t^2 g = c^2\partial_x^2 g$. Point out that all three contain a second derivative.

**On the board:**

1. $N(x) = \sum_j v_j f(w_jx + b_j) + c$.
2. $N'(x) = \sum_j v_j w_j f'(w_jx + b_j)$ (chain rule, $w_j$ comes out).
3. $N''(x) = \sum_j v_j w_j^2 f''(w_jx + b_j)$.
4. Ask: what is $f''$ for ReLU? It is $0$ everywhere (except at the kink). So $N'' = 0$ for **any** weights.

**Click:** the ReLU box, the table, and "we use tanh".

**Say:** this is not about training. A ReLU network is piecewise linear, so it *has* no curvature. More layers don't help either: compositions of piecewise-linear functions are piecewise linear.

---

## Boundary conditions (slide only, ≈ 3 min)

No board work. The table of conditions and trial solutions is on the slide from the start, with the good/bad boxes.

- Ask the room to **check one row**: put $x=0$ and $x=1$ into $g_t = a + (b-a)x + x(1-x)N$. The $N$ term drops out at both ends. (Last row: also differentiate once, $g_t' = v_0 + 2xN + x^2N'$, so $g_t'(0) = v_0$.)
- Read the good and the bad. **Say:** a derivative condition like $g'(1)=0$ needs a trick, and every new condition is a new puzzle. Now imagine a curved boundary in 2D. This motivates soft constraints later.
- *If asked how to handle $g'(1)=0$:* see the lecturer's guide (the $g(0)=0$, $g'(1)=0$ trick).

*Optional:* for the Poisson residual you need $g_t'' = -2N + 2(1-2x)N' + x(1-x)N''$. It contains $N''$, so W4 applies.

---

## Results vs standard methods: guess first (≈ 4 min)

The slide first shows only the plots and a "Guess first" box: *network or standard method, which is more accurate? Which is faster?* Take a show of hands for each question (30 s).

**Click:** the table and "Read it honestly". The network beats forward Euler ($1\times10^{-4}$ vs $9\times10^{-4}$), but RK4 with the same step gets $2\times10^{-10}$, and the standard methods take microseconds. The plots use $\Delta = 0.1$ so the points are visible; the table uses $0.01$.

---

## Diffusion and soft constraints (≈ 4 + 5 min)

These now have more time; use it to slow down.

- **Diffusion:** let the students check the trial solution $g_t=(1-t)\sin\pi x+x(1-x)\,t\,N(x,t)$ themselves ($t=0$, $x=0$, $x=1$) before you say it. Then read the error map: zero on the three edges, largest near $t\approx0.05$, where the solution decays fastest.
- **Soft constraints:** write the three loss terms side by side with the trial solution, so students see that each condition that used to be built into $g_t$ is now a penalty. Compare the RMSE (hard $1.8\times10^{-3}$, soft $6.5\times10^{-3}$) and the error at $t=0$ (exactly 0 vs. $6\times10^{-3}$).

---

## Quick question 3: a ReLU PINN (≈ 5 min)

Comes right after the soft-constraints (PINN) slide. Students reason from W4; no derivation.

- Read the question: the same soft cost for $u_t = u_{xx}$, but a **ReLU** network $u = N(x,t)$. What does training converge to?
- 1–2 minutes in pairs. A hint if needed: what did W4 say about $N''$ for ReLU?
- **Click:** three bullets and the figure (exact solution solid, ReLU PINN dashed: the dashed curves stay at $\sin\pi x$). $N_{xx} = 0$ at every sample, so the PDE term only asks for $N_t = 0$. The network learns $N \approx \sin\pi x$ for all times: it never diffuses. Small cost ($6\times10^{-4}$), wrong answer (RMSE $0.65$ vs. $1.8\times10^{-3}$).
- **Say:** a small residual is not a solution. Use smooth activations, and always check against something you know. (The notebook has a *Try it* for this.)

---

## Inverse problems (≈ 4 min)

- Write the cost with $D$ as a parameter, and ask: which term would a trial solution replace? (None: we don't know the conditions. Soft constraints are the only option.)
- Point at the three runs converging to $D\approx0.51$, then at the warning: five times more noise gives $D=0.69$. Ask how they would put an error bar on $D$ (bootstrap, from Project 1).
