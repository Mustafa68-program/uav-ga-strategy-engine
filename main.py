"""Command-line entry point for the UAV Strategy Engine.

Usage:
    python main.py run            # one GA run + plots (convergence, paths, coverage)
    python main.py sensitivity    # hyper-parameter sensitivity study
    python main.py baseline       # GA vs random search comparison
    python main.py all            # everything above
Add --quick for a fast smoke test.
"""
import argparse
import json
import os
import time
from dataclasses import asdict, replace

import numpy as np

from uav_ga import (Environment, FitnessEvaluator, GAConfig, MissionConfig, Penalties,
                    Weights, GeneticAlgorithm, random_search)
from uav_ga import visualization as viz
from uav_ga.experiments import sensitivity, baseline_comparison, coverage_similarity

OUT = "results"


def save_json(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2)


def solution_to_dict(ind):
    return dict(waypoints=ind.wp.round(2).tolist(), speed=ind.speed.round(3).tolist(),
                sensor_power=ind.power.round(3).tolist(), sensor_active=ind.active.tolist(),
                fitness=float(ind.fitness),
                components={k: (v if isinstance(v, bool) else float(v)) for k, v in ind.components.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "sensitivity", "baseline", "all"])
    ap.add_argument("--quick", action="store_true", help="small budgets for a fast test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    env, mission, weights, penalties = Environment(), MissionConfig(), Weights(), Penalties()
    cfg = GAConfig(seed=args.seed)
    if args.quick:
        cfg = replace(cfg, population_size=30, generations=30)
    save_json(dict(mission=asdict(mission), weights=asdict(weights), penalties=asdict(penalties),
                   ga=asdict(cfg)), "config_used.json")
    viz.plot_environment(env, os.path.join(OUT, "fig1_environment.png"))

    if args.command in ("run", "all"):
        print("== Single GA run ==")
        snap = [0, 10, 40, cfg.generations]
        ev = FitnessEvaluator(env, mission, weights, penalties)
        t0 = time.time()
        res = GeneticAlgorithm(ev, env, mission, cfg, snapshot_gens=snap).run()
        print(f"best fitness {res.best.fitness:.4f} after {res.evaluations} evaluations "
              f"({res.generations_run} generations, {time.time() - t0:.1f}s)")
        print({k: (v if isinstance(v, bool) else round(v, 3)) for k, v in res.best.components.items()})
        ev_rs = FitnessEvaluator(env, mission, weights, penalties)
        _, rs_curve = random_search(ev_rs, env, mission, res.evaluations, seed=100, batch=cfg.population_size)
        viz.plot_convergence(res.history, os.path.join(OUT, "fig2_convergence.png"), rs_curve=None)
        snaps = {g: res.snapshots[g] for g in sorted(res.snapshots)}
        viz.plot_generation_panels(snaps, env, mission, ev, os.path.join(OUT, "fig3_paths_and_coverage.png"))
        viz.plot_tradeoff(res.final_population, os.path.join(OUT, "fig6_tradeoff.png"))
        save_json(solution_to_dict(res.best), "best_solution.json")
        save_json({k: [float(x) for x in v] for k, v in res.history.items()}, "history.json")

    if args.command in ("sensitivity", "all"):
        print("== Hyper-parameter sensitivity ==")
        gens = 30 if args.quick else 100
        seeds = (0,) if args.quick else (0, 1, 2)
        sens = sensitivity(env, mission, weights, penalties, cfg, seeds=seeds, generations=gens)
        viz.plot_sensitivity(sens, os.path.join(OUT, "fig4_sensitivity.png"))
        save_json({p: {str(v): {k: x for k, x in r.items() if k != "best_curves"} for v, r in runs.items()}
                   for p, runs in sens.items()}, "sensitivity.json")

    if args.command in ("baseline", "all"):
        print("== GA vs random search ==")
        n_seeds = 3 if args.quick else 10
        summary, ga_best, rs_best = baseline_comparison(env, mission, weights, penalties, cfg, n_seeds=n_seeds)
        viz.plot_baseline(summary, os.path.join(OUT, "fig5_baseline.png"))
        ev = FitnessEvaluator(env, mission, weights, penalties)
        viz.plot_solution_diversity(ga_best[:4], env, mission, os.path.join(OUT, "fig7_solution_diversity.png"))
        stats = {m: {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items()
                     if k not in ("eval_curves",)} for m, d in summary.items() if m != "batch"}
        stats["coverage_similarity_ga"] = coverage_similarity(ga_best, ev)
        stats["coverage_similarity_rs"] = coverage_similarity(rs_best, ev)
        stats["ga_feasible"] = float(np.mean([i.components["feasible"] for i in ga_best]))
        stats["rs_feasible"] = float(np.mean([i.components["feasible"] for i in rs_best]))
        save_json(stats, "baseline.json")
        save_json({m: {k: [float(x) for x in v] for k, v in d.items() if k != "eval_curves"}
                   for m, d in summary.items() if m != "batch"}, "baseline_raw.json")
        print("coverage-map similarity between runs  GA:", round(stats["coverage_similarity_ga"], 3),
              " random:", round(stats["coverage_similarity_rs"], 3))
    print("Done. Results are in the 'results/' folder.")


if __name__ == "__main__":
    main()
