"""Slide figures for week42monday_lecture_v2.tex (Part 1), with fonts sized for the slides.

Reuses the numpy code of week42monday.ipynb (Part 1); needs numpy and matplotlib only.
Usage: python make_week42_v2_figures.py week42monday.ipynb figures <path/to/mnist.npz>
"""
import sys, os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nb_path, out_dir, mnist_path = sys.argv[1], sys.argv[2], sys.argv[3]
nb = json.load(open(nb_path, encoding="utf-8"))
src = "\n\n".join("".join(nb["cells"][i]["source"]) for i in (1, 3, 4, 8))
src = "\n".join(line for line in src.splitlines()
                if not line.startswith(("import jax", "jax.config", "from IPython")))
g = {"__name__": "nbcode"}
exec(compile(src, "notebook_cells", "exec"), g)
for k in ("sigmoid", "relu", "create_layers", "train", "predict", "feed_forward",
          "runge_data", "one_hot", "accuracy", "BLUE", "RED", "GREEN", "GREY"):
    globals()[k] = g[k]

plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                     "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(os.path.join(out_dir, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- activations
z = np.linspace(-5, 5, 400)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
for f, fp, lab, col, ls in ((sigmoid(z), sigmoid(z) * (1 - sigmoid(z)), "sigmoid", BLUE, "-"),
                            (np.tanh(z), 1 - np.tanh(z) ** 2, "tanh", GREEN, "--"),
                            (relu(z), (z > 0).astype(float), "ReLU", RED, "-")):
    ax[0].plot(z, f, color=col, lw=2.2, ls=ls, label=lab)
    ax[1].plot(z, fp, color=col, lw=2.2, ls=ls, label=lab)
ax[0].set_ylim(-1.2, 3)
ax[0].set_title("activation $f(z)$")
ax[1].set_title("derivative $f'(z)$")
ax[1].axhline(0.25, color=GREY, lw=0.8, ls=":")
for a in ax:
    a.set_xlabel("$z$")
ax[0].legend(loc="upper left", frameon=False)
fig.tight_layout()
save(fig, "v2_activations.png")

# ---------------------------------------------------------------- mismatched pair
z = np.linspace(-8, 8, 400)
a = sigmoid(z)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.9))
ax[0].plot(z, -np.log(a), color=BLUE, lw=2.2, label="cross entropy")
ax[0].plot(z, 0.5 * (a - 1) ** 2, color=RED, lw=2.2, label="squared error")
ax[0].set_ylim(0, 4)
ax[0].set_title("cost, target $y=1$")
ax[1].plot(z, np.abs(a - 1), color=BLUE, lw=2.2, label="cross entropy")
ax[1].plot(z, np.abs(a - 1) * a * (1 - a), color=RED, lw=2.2, label="squared error")
ax[1].set_title(r"gradient signal $|\delta^L|$")
ax[1].axvline(-5, color=GREY, lw=0.8, ls=":")
for x_ in ax:
    x_.set_xlabel("$z$  (wrong $\\leftarrow$ $\\rightarrow$ right)")
ax[1].legend(loc="upper right", frameon=False)
fig.tight_layout()
save(fig, "v2_delta_heads.png")

# ---------------------------------------------------------------- Runge: kinks of a trained ReLU network
t0 = time.time()
X, Y, Xt, Yt = runge_data()
layers = create_layers([1, 50, 1], "relu", np.random.default_rng(2026))
layers, hist = train(X, Y, layers, "relu", "mse", gamma=0.1, epochs=20000)
(W1, b1), (W2, b2) = layers
test_mse = float(np.mean((predict(Xt, layers, "relu", "mse") - Yt) ** 2))
kinks = -b1 / W1[0]
active = np.abs(W2[:, 0] * W1[0]) > 1e-3          # units that actually bend the output
kinks = np.sort(kinks[(kinks > -1) & (kinks < 1) & active])
xx = np.linspace(-1, 1, 801)[:, None]
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(X, Y, "o", color=GREY, ms=3, alpha=0.6, label="data")
ax.plot(xx, 1 / (1 + 25 * xx ** 2), color="k", lw=1.2, ls=":", label="Runge function")
ax.plot(xx, predict(xx, layers, "relu", "mse"), color=BLUE, lw=2.2, label="network")
ax.plot(kinks, predict(kinks[:, None], layers, "relu", "mse")[:, 0], "o", color=RED, ms=6,
        label="kinks")
ax.set_xlabel("$x$")
ax.legend(loc="upper right", frameon=False, fontsize=11)
fig.tight_layout()
save(fig, "v2_runge_kinks.png")
print(f"Runge: test MSE {test_mse:.4f}, {len(kinks)} kinks inside [-1,1]  ({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- MNIST: the quick question
t0 = time.time()
d = np.load(mnist_path)
X_train = d["x_train"].reshape(len(d["x_train"]), -1).astype(float) / 255.0
X_test = d["x_test"].reshape(len(d["x_test"]), -1).astype(float) / 255.0
y_train, y_test = d["y_train"].astype(int), d["y_test"].astype(int)
layers = create_layers([784, 100, 10], "relu", np.random.default_rng(2026))
layers, hist = train(X_train, one_hot(y_train, 10), layers, "relu", "softmax", gamma=0.1, epochs=20,
                     batch_size=100, X_test=X_test, y_test=y_test)
P = predict(X_test, layers, "relu", "softmax")
pred, pmax = P.argmax(axis=1), P.max(axis=1)
i_sure = int(np.argmax(np.where(pred == y_test, pmax, -1)))
i_unsure = int(np.argmin(pmax))
fig = plt.figure(figsize=(6.2, 5.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.6], wspace=0.45, hspace=0.5)
for row, i in enumerate((i_sure, i_unsure)):
    axi = fig.add_subplot(gs[row, 0])
    axi.imshow(X_test[i].reshape(28, 28), cmap="gray_r")
    axi.set_title(f"label {y_test[i]}")
    axi.axis("off")
    axb = fig.add_subplot(gs[row, 1])
    k = np.arange(10)
    axb.bar(k, P[i], color=RED, width=0.7)
    axb.set_xticks(k)
    axb.set_ylim(0, 1.05)
    axb.set_ylabel("$a_k$")
    axb.set_title(f"image {'A' if row == 0 else 'B'}: softmax output")
    if row == 1:
        axb.set_xlabel("class $k$")
save(fig, "v2_mnist_question.png")
print(f"MNIST: test acc {hist['test_accuracy'][-1]:.4f}; sure: true {y_test[i_sure]} p={pmax[i_sure]:.4f}; "
      f"unsure: true {y_test[i_unsure]} pred {pred[i_unsure]} top3 {np.sort(P[i_unsure])[::-1][:3]} "
      f"all {np.round(P[i_unsure], 3)}  ({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- initialisation
rng = np.random.default_rng(0)
M, depth, n = 256, 10, 500
Xi = rng.normal(size=(n, M))
Yi = rng.normal(size=(n, 1))


def deep_stats(act, scale_fn, seed=1):
    """Forward and backward pass through `depth` hidden layers of width M; linear output, C = 1/2 mean (a-y)^2."""
    r = np.random.default_rng(seed)
    f = np.tanh if act == "tanh" else relu
    fp = (lambda z: 1 - np.tanh(z) ** 2) if act == "tanh" else (lambda z: (z > 0).astype(float))
    Ws = [r.normal(0, scale_fn(M), (M, M)) for _ in range(depth)] + [r.normal(0, np.sqrt(1 / M), (M, 1))]
    a, zs, As = Xi, [], [Xi]
    for l, W in enumerate(Ws):
        zl = a @ W
        zs.append(zl)
        a = f(zl) if l < depth else zl
        As.append(a)
    delta = (a - Yi) / n
    gnorm = []
    for l in range(depth, -1, -1):
        gnorm.append(np.linalg.norm(As[l].T @ delta))
        if l > 0:
            delta = (delta @ Ws[l].T) * fp(zs[l - 1])
    return As[1:depth + 1], gnorm[::-1][:depth]


cases = [("tanh", lambda m: 1.0, r"$\sigma_w=1$ (too large)", RED, "-"),
         ("tanh", lambda m: 0.01, r"$\sigma_w=0.01$ (too small)", GREY, "--"),
         ("tanh", lambda m: np.sqrt(1 / m), r"$\sigma_w^2=1/M$", BLUE, "-")]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
bins = np.linspace(-1, 1, 41)
for act, sf, lab, col, ls in cases:
    As, gn = deep_stats(act, sf)
    ax[0].hist(As[-1].ravel(), bins=bins, density=True, histtype="step", lw=2.2, color=col, ls=ls, label=lab)
    ax[1].semilogy(np.arange(1, depth + 1), gn, "o-", color=col, ls=ls, lw=2, ms=4, label=lab)
ax[0].set_yscale("log")
ax[0].set_title("tanh: activations, layer 10")
ax[0].set_xlabel("$a$")
ax[1].set_title(r"tanh: gradient size $\|\partial C/\partial W^l\|$")
ax[1].set_xlabel("layer $l$")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.36, -0.12), ncol=3, frameon=False, fontsize=12)
for lab, sf, col, ls in ((r"$\sigma_w^2=1/M$", lambda m: np.sqrt(1 / m), GREY, "--"),
                         (r"$\sigma_w^2=2/M$ (He)", lambda m: np.sqrt(2 / m), BLUE, "-")):
    As, _ = deep_stats("relu", sf)
    ax[2].semilogy(np.arange(1, depth + 1), [A.std() for A in As], "o-", color=col, ls=ls, lw=2, ms=4, label=lab)
ax[2].set_title("ReLU: spread of activations")
ax[2].set_xlabel("layer $l$")
ax[2].set_ylabel("std of $a^l$")
ax[2].legend(frameon=False, fontsize=11)
fig.tight_layout()
save(fig, "v2_init.png")
for act, sf, lab, *_ in cases:
    As, gn = deep_stats(act, sf)
    sat = np.mean(np.abs(As[-1]) > 0.99)
    print(f"init {lab}: layer-10 std {As[-1].std():.3g}, saturated {sat:.2f}, grad layer1/layer10 {gn[0]:.2e}/{gn[-1]:.2e}")
for lab, sf in (("relu 1/M", lambda m: np.sqrt(1 / m)), ("relu 2/M", lambda m: np.sqrt(2 / m))):
    As, gn = deep_stats("relu", sf)
    print(f"{lab}: std layer1 {As[0].std():.3g} layer10 {As[-1].std():.3g}")
