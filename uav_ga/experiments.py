"""Experiments: hyper-parameter sensitivity and comparison against random search."""
import time
from dataclasses import replace
import numpy as np
from .config import GAConfig
from .fitness import FitnessEvaluator
from .genetic_algorithm import GeneticAlgorithm
from .baseline import random_search


def run_ga(env, mission, weights, penalties, cfg, snapshot_gens=()):
    ev = FitnessEvaluator(env, mission, weights, penalties)
    return GeneticAlgorithm(ev, env, mission, cfg, snapshot_gens).run(), ev


def sensitivity(env, mission, weights, penalties, base_cfg, seeds=(0, 1, 2), generations=100, log=print):
    """Vary one hyper-parameter at a time; every setting is run with several seeds."""
    grid = {
        "population_size": [20, 80, 200],
        "mutation_rate": [0.005, 0.03, 0.30],
        "tournament_size": [1, 3, 8],
    }
    out = {}
    for param, values in grid.items():
        out[param] = {}
        for v in values:
            finals, curves, evals, divs, feas = [], [], [], [], []
            for s in seeds:
                cfg = replace(base_cfg, generations=generations, seed=s, patience=10 ** 6, **{param: v})
                res, _ = run_ga(env, mission, weights, penalties, cfg)
                finals.append(res.best.fitness)
                curves.append(res.history["best"])
                evals.append(res.evaluations)
                divs.append(res.history["diversity"][-1])
                feas.append(res.history["feasible"][-1])
            out[param][v] = dict(mean=float(np.mean(finals)), std=float(np.std(finals)),
                                 finals=[float(f) for f in finals], best_curves=curves,
                                 evaluations=float(np.mean(evals)), diversity=float(np.mean(divs)),
                                 feasible=float(np.mean(feas)))
            log(f"  {param}={v}: {np.mean(finals):.3f} +/- {np.std(finals):.3f} "
                f"(evals {np.mean(evals):.0f}, final diversity {np.mean(divs):.3f})")
    return out


def baseline_comparison(env, mission, weights, penalties, cfg, n_seeds=10, log=print):
    """GA vs random search (same budget) and random search with a 5x budget."""
    keys = ["fitness", "coverage", "detection", "energy", "hazard", "time"]
    summary = {m: {k: [] for k in keys} for m in ("ga", "rs", "rs5")}
    summary["ga"]["eval_curves"], summary["rs"]["eval_curves"] = [], []
    ga_best, rs_best = [], []
    batch = cfg.population_size
    for s in range(n_seeds):
        c = replace(cfg, seed=s)
        res, ev = run_ga(env, mission, weights, penalties, c)
        budget = res.evaluations
        ev_rs = FitnessEvaluator(env, mission, weights, penalties)
        rs, curve = random_search(ev_rs, env, mission, budget, seed=100 + s, batch=batch)
        for name, ind in (("ga", res.best), ("rs", rs)):
            summary[name]["fitness"].append(ind.fitness)
            for k in keys[1:]:
                summary[name][k].append(ind.components[k])
        ga_c = np.maximum.accumulate(np.repeat(res.history["best"], batch)[:len(curve) * batch])
        summary["ga"]["eval_curves"].append(list(ga_c[batch - 1::batch][:len(curve)]))
        summary["rs"]["eval_curves"].append(curve)
        ga_best.append(res.best)
        rs_best.append(rs)
        log(f"  seed {s}: GA {res.best.fitness:.3f} | random {rs.fitness:.3f} (budget {budget})")
    summary["batch"] = batch
    # random search with 5x budget on 2 seeds (shows more luck does not close the gap)
    for s in range(2):
        ev_rs = FitnessEvaluator(env, mission, weights, penalties)
        rs, _ = random_search(ev_rs, env, mission, 5 * budget, seed=500 + s, batch=batch)
        for k in keys[1:]:
            summary["rs5"][k].append(rs.components[k])
        summary["rs5"]["fitness"].append(rs.fitness)
        log(f"  random search x5 budget, seed {s}: {rs.fitness:.3f}")
    return summary, ga_best, rs_best


def coverage_similarity(inds, evaluator):
    """Mean pairwise Jaccard similarity of the coverage maps (1 = identical, 0 = disjoint)."""
    masks = [evaluator.coverage_mask(evaluator.simulate(i)).ravel() for i in inds]
    sims = []
    for a in range(len(masks)):
        for b in range(a + 1, len(masks)):
            inter = np.logical_and(masks[a], masks[b]).sum()
            union = np.logical_or(masks[a], masks[b]).sum()
            sims.append(inter / union if union else 1.0)
    return float(np.mean(sims))
