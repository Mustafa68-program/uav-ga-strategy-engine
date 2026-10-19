"""All plotting helpers (matplotlib, headless-safe)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

DRONE_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]


def _draw_map(ax, env, show_targets=True):
    n = env.size
    ax.imshow(env.priority, origin="lower", cmap="YlOrRd", alpha=0.75, extent=(-0.5, n - 0.5, -0.5, n - 0.5))
    ax.contour(env.X, env.Y, env.hazard, levels=[0.35, 0.65], colors=["#6a3d9a", "#3b0f70"],
               linewidths=[0.8, 1.4], linestyles=["--", "-"])
    for z in env.no_fly:
        ax.add_patch(Circle((z.cx, z.cy), z.r, facecolor="none", edgecolor="black", hatch="///", lw=1.2))
    if show_targets:
        ax.scatter(env.targets[:, 0], env.targets[:, 1], marker="x", c="black", s=18, zorder=3)
    ax.plot(*env.base, marker="s", color="cyan", mec="black", ms=8, zorder=5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")


def plot_environment(env, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    _draw_map(ax, env)
    ax.set_title("Hazard grid: priority (colour), hazards (purple), no-fly (hatched), targets (x)", fontsize=8)
    ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_paths(ax, ind, env, mission, title=""):
    _draw_map(ax, env)
    base = np.asarray(mission.base)
    for d in range(ind.wp.shape[0]):
        pts = np.vstack([base, ind.wp[d], base])
        c = DRONE_COLORS[d % len(DRONE_COLORS)]
        for s in range(len(pts) - 1):
            style = "-" if ind.active[d, s] else ":"       # solid = sensor ON, dotted = OFF
            ax.plot(pts[s:s + 2, 0], pts[s:s + 2, 1], style, color=c, lw=1.6)
        ax.scatter(ind.wp[d, :, 0], ind.wp[d, :, 1], color=c, s=14, zorder=4, edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def plot_coverage(ax, ind, env, evaluator, title=""):
    n = env.size
    mask = evaluator.coverage_mask(evaluator.simulate(ind))
    ax.imshow(env.priority, origin="lower", cmap="Greys", alpha=0.55, extent=(-0.5, n - 0.5, -0.5, n - 0.5))
    overlay = np.ma.masked_where(~mask, np.ones_like(env.priority))
    ax.imshow(overlay, origin="lower", cmap="winter", alpha=0.55, vmin=0, vmax=1.4, extent=(-0.5, n - 0.5, -0.5, n - 0.5))
    for z in env.no_fly:
        ax.add_patch(Circle((z.cx, z.cy), z.r, facecolor="none", edgecolor="black", hatch="///", lw=1.0))
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5); ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def plot_generation_panels(snapshots, env, mission, evaluator, path):
    """Row 1: evolved flight paths; row 2: coverage heat-map, at selected generations."""
    gens = sorted(snapshots)
    fig, axes = plt.subplots(2, len(gens), figsize=(3.3 * len(gens), 6.6))
    axes = np.atleast_2d(axes)
    for j, g in enumerate(gens):
        ind = snapshots[g]
        evaluator.evaluate(ind)
        c = ind.components
        plot_paths(axes[0, j], ind, env, mission, f"Generation {g}\nfitness = {ind.fitness:.3f}")
        plot_coverage(axes[1, j], ind, env, evaluator,
                      f"coverage {100 * c['coverage']:.0f}% | detection {100 * c['detection']:.0f}%")
    fig.suptitle("Evolution of the best UAV plan (top: flight paths, bottom: coverage heat-map; solid = sensor ON)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=140); plt.close(fig)


def plot_convergence(history, path, rs_curve=None, batch=80):
    gens = np.arange(len(history["best"]))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(gens, history["best"], color="#d62728", lw=2, label="GA best")
    ax.plot(gens, history["mean"], color="#1f77b4", lw=1.6, label="GA average")
    ax.fill_between(gens, history["worst"], history["best"], color="#1f77b4", alpha=0.08)
    if rs_curve is not None:
        ax.plot(np.arange(1, len(rs_curve) + 1), rs_curve, color="gray", ls="--", lw=1.5, label="Random search best")
    ax.set_ylim(min(-1.0, np.percentile(history["mean"], 3)), max(history["best"]) + 0.1)
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness")
    ax2 = ax.twinx()
    ax2.plot(gens, history["diversity"], color="green", ls=":", lw=1.2, label="Diversity")
    ax2.set_ylabel("Diversity (norm. std)", color="green"); ax2.tick_params(axis="y", colors="green")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
    ax.set_title("Fitness convergence: best vs. average fitness", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_sensitivity(results, path):
    """results: {'population_size': {value: {...}}, 'mutation_rate': {...}, 'tournament_size': {...}}"""
    titles = {"population_size": "Population size", "mutation_rate": "Mutation rate",
              "tournament_size": "Selection pressure (tournament size)"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (param, runs) in zip(axes, results.items()):
        for i, (value, r) in enumerate(runs.items()):
            curve = np.mean(r["best_curves"], axis=0)
            ax.plot(np.arange(len(curve)), curve, lw=1.8, label=f"{value}  (final {r['mean']:.3f})")
        ax.set_title(titles[param], fontsize=9); ax.set_xlabel("Generation"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("Best fitness (mean of seeds)")
    lo = min(np.mean(r["best_curves"], axis=0)[5] for runs in results.values() for r in runs.values())
    axes[0].set_ylim(max(lo - 0.05, -0.6), None)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_baseline(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    data = [summary["ga"]["fitness"], summary["rs"]["fitness"], summary["rs5"]["fitness"]]
    labels = ["GA", "Random\n(same budget)", "Random\n(5x budget)"]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#d62728", "#999999", "#bbbbbb"]):
        patch.set_facecolor(c)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Best fitness over independent runs", fontsize=9); ax.set_ylabel("Fitness"); ax.grid(alpha=0.3)

    ax = axes[1]
    ga_curve = np.mean(summary["ga"]["eval_curves"], axis=0)
    rs_curve = np.mean(summary["rs"]["eval_curves"], axis=0)
    ax.plot(np.arange(1, len(ga_curve) + 1) * summary["batch"], ga_curve, color="#d62728", lw=2, label="GA (best-so-far)")
    ax.plot(np.arange(1, len(rs_curve) + 1) * summary["batch"], rs_curve, color="gray", lw=2, ls="--", label="Random search")
    ax.set_xlabel("Fitness evaluations"); ax.set_ylabel("Best fitness so far"); ax.legend(fontsize=7)
    ax.set_title("Search efficiency (mean of runs)", fontsize=9); ax.grid(alpha=0.3)
    ax.set_ylim(-1.0, None)

    ax = axes[2]
    keys = ["coverage", "detection", "energy", "hazard"]
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w / 2, [np.mean(summary["ga"][k]) for k in keys], w, label="GA", color="#d62728")
    ax.bar(x + w / 2, [np.mean(summary["rs"][k]) for k in keys], w, label="Random search", color="#999999")
    ax.set_xticks(x); ax.set_xticklabels(["Coverage\n(max)", "Detection\n(max)", "Energy\n(min)", "Hazard\n(min)"], fontsize=8)
    ax.set_ylim(0, 1.35)
    ax.set_title("Objective values of the best plans", fontsize=9); ax.legend(fontsize=7, ncol=2, loc="upper right"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_solution_diversity(inds, env, mission, path):
    """Best plan of several independent GA runs - different seeds give different good solutions."""
    k = len(inds)
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.4))
    for i, (ax, ind) in enumerate(zip(np.atleast_1d(axes), inds)):
        plot_paths(ax, ind, env, mission, f"seed {i}: fitness {ind.fitness:.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_tradeoff(population, path):
    cov = [i.components["coverage"] for i in population]
    en = [i.components["energy"] for i in population]
    hz = [i.components["hazard"] for i in population]
    feas = np.array([i.components["feasible"] for i in population])
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    sc = ax.scatter(en, cov, c=hz, cmap="viridis_r", s=26, edgecolor=np.where(feas, "black", "none"), linewidth=0.6)
    fig.colorbar(sc, ax=ax, label="Hazard exposure (normalised)")
    ax.set_xlabel("Energy (normalised, lower is better)"); ax.set_ylabel("Priority coverage (higher is better)")
    ax.set_title("Trade-off in the final population (outlined = feasible)", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
