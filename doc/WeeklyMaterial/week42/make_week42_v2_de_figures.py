"""DE figures for week42monday_lecture_v2.tex, with fonts sized for the slides.

Reuses the definition cells of week42monday.ipynb (Parts 1-4).
Usage: python make_week42_v2_de_figures.py week42monday.ipynb figures   (needs jax)
"""
import sys, os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nb_path, out_dir = sys.argv[1], sys.argv[2]
nb = json.load(open(nb_path, encoding="utf-8"))
src = "\n\n".join("".join(nb["cells"][i]["source"]) for i in (1, 3, 4, 15, 18, 20, 24, 28, 31))
src = src.replace("from IPython.display import Image", "")
g = {"__name__": "nbcode"}
exec(compile(src, "notebook_cells", "exec"), g)
globals().update({k: v for k, v in g.items() if not k.startswith("__")})

plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                     "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(os.path.join(out_dir, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- the three ODEs
t0 = time.time()
res = {}
for name, residual, trial, exact, sizes, n_iter, gamma, n_col in (
        ("decay", residual_decay, trial_decay, exact_decay, [1, 40, 40, 1], 3000, 2e-2, 50),
        ("logistic", residual_logistic, trial_logistic, exact_logistic, [1, 40, 40, 1], 3000, 2e-2, 50),
        ("poisson", residual_poisson, trial_poisson, exact_poisson, [1, 30, 30, 1], 4000, 1e-2, 60)):
    X = np.linspace(0, 1, n_col)[:, None]
    layers, hist = solve_de(residual, sizes, X, "tanh", n_iter=n_iter, gamma=gamma, rng=np.random.default_rng(1))
    xx = np.linspace(0, 1, 401)
    gx = np.asarray(trial(layers, jnp.asarray(xx[:, None])))
    res[name] = (xx, gx, np.abs(gx - exact(xx)).max())
    print(f"{name}: max error {res[name][2]:.2e}")
print(f"ODEs ({time.time() - t0:.0f} s)")

xx, gx, err = res["decay"]
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(xx, exact_decay(xx), color="k", lw=1.5, label="exact")
ax.plot(xx[::16], gx[::16], "o", color=BLUE, ms=7, mfc="none", mew=1.8, label="network")
ax.set_xlabel("$x$")
ax.set_ylabel("$g$")
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "v2_ode_decay.png")

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))
xx, gx, err = res["logistic"]
t10, g10 = euler_logistic(10)
ax[0].plot(xx, exact_logistic(xx), color="k", lw=1.5, label="exact")
ax[0].plot(xx[::16], gx[::16], "o", color=BLUE, ms=7, mfc="none", mew=1.8, label="network")
ax[0].plot(t10, g10, "s--", color=RED, ms=5, lw=1.2, label=r"Euler, $\Delta t=0.1$")
ax[0].set_title("logistic growth: $g'=2g(1-g)$")
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$g$")
ax[0].legend(frameon=False)
xx, gx, err = res["poisson"]
x10, u10 = fd_poisson(10)
ax[1].plot(xx, exact_poisson(xx), color="k", lw=1.5, label="exact")
ax[1].plot(xx[::16], gx[::16], "o", color=BLUE, ms=7, mfc="none", mew=1.8, label="network")
ax[1].plot(x10, u10, "s--", color=RED, ms=5, lw=1.2, label=r"finite diff., $\Delta x=0.1$")
ax[1].set_title("Poisson: $-g''=f$, $g(0)=g(1)=0$")
ax[1].set_xlabel("$x$")
ax[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.62, 0.0), fontsize=11)
fig.tight_layout()
save(fig, "v2_ode_results.png")

# ---------------------------------------------------------------- diffusion, hard constraints
t0 = time.time()
ev = solve_diffusion_hard(n_iter=800)
xfd, snaps, fd_err, dt, n_steps = fd_diffusion()
print(f"diffusion: network max error {ev['max']:.2e}, rmse {ev['rmse']:.2e}, t=0 error {ev['t0']:.1e}; "
      f"FD max error {fd_err[:, 1].max():.2e} ({time.time() - t0:.0f} s)")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.6})
for ts, col in zip((0.0, 0.1, 0.3), ("k", BLUE, RED)):
    it = int(round(ts * 100))
    ax[0].plot(ev["x"], exact_diffusion(ev["x"], ev["t"][it]), color=col, lw=1.5, label=f"$t={ts:.1f}$")
    ax[0].plot(ev["x"][::6], ev["g"][::6, it], "o", color=col, ms=6, mfc="none", mew=1.5)
ax[0].plot([], [], "o", color=GREY, mfc="none", label="network")
ax[0].set_xlabel("$x$")
ax[0].set_ylabel("$g(x,t)$")
ax[0].set_title("exact (lines) and network")
ax[0].legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(0.98, 1.0))
im = ax[1].imshow(ev["map"].T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
fig.colorbar(im, ax=ax[1])
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$t$")
ax[1].set_title("error $|g_t-g|$")
fig.tight_layout()
save(fig, "v2_diffusion.png")

# ---------------------------------------------------------------- inverse problem
t0 = time.time()
X_obs, y_obs = observations()
runs = []
for D0 in (2.0, 1.0, 0.1):
    params, trace = pinn_inverse(X_obs, y_obs, D0=D0)
    runs.append((D0, params, trace))
    print(f"D0={D0}: D={float(params[1]):.4f}")
Xo, yo = observations(noise=0.05)
p5, _ = pinn_inverse(Xo, yo, D0=2.0)
print(f"noise 0.05: D={float(p5[1]):.4f}  ({time.time() - t0:.0f} s)")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
for (D0, params, trace), col in zip(runs, (BLUE, RED, GREEN)):
    ax[0].plot(trace[:, 0], trace[:, 2], color=col, lw=2, label=f"start $D={D0}$")
ax[0].axhline(D_TRUE, color="k", lw=1.2, ls="--", label="true $D=0.5$")
ax[0].set_xlabel("training step")
ax[0].set_ylabel("$D$")
ax[0].set_yscale("log")
ax[0].set_title("$D$ during training")
ax[0].legend(frameon=False, fontsize=11)
layers, D = runs[0][1]
xx = np.linspace(0, 1, 200)
Xn, yn = np.asarray(X_obs), np.asarray(y_obs)
for ts, col in zip((0.05, 0.3, 0.7), (BLUE, RED, GREEN)):
    Xs = jnp.asarray(np.column_stack([xx, np.full_like(xx, ts)]))
    ax[1].plot(xx, np.asarray(u_net(layers, Xs)), color=col, lw=2, label=f"$t={ts}$")
    ax[1].plot(xx, exact_diffusion(xx, ts, D_TRUE), color=col, lw=1, ls="--")
ax[1].scatter(Xn[:, 0], yn, color=GREY, s=16, zorder=3, label="measurements\n(all times)")
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$u$")
ax[1].set_title("network (solid), exact (dashed)")
ax[1].legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
save(fig, "v2_pinn_inverse.png")
