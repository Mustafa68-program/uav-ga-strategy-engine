"""Baseline: pure random search with the same evaluation budget."""
import numpy as np
from .chromosome import Individual


def random_search(evaluator, env, mission, n_evals, seed=0, batch=80):
    """Sample random (repaired) plans; return best plan and best-so-far curve per batch."""
    rng = np.random.default_rng(seed)
    best, curve = None, []
    for i in range(n_evals):
        ind = Individual.random(mission, rng).repair(env, mission)
        evaluator.evaluate(ind)
        if best is None or ind.fitness > best.fitness:
            best = ind
        if (i + 1) % batch == 0:
            curve.append(best.fitness)
    return best, curve
