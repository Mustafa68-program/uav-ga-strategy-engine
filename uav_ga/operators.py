"""Genetic operators: selection, spatial crossover and spatial mutation."""
import numpy as np
from .chromosome import Individual


# ------------------------------------------------------------------ selection
def tournament_selection(pop, k, rng):
    """Pick k random individuals, return the fittest (k controls selection pressure)."""
    idx = rng.integers(0, len(pop), size=k)
    return max((pop[i] for i in idx), key=lambda ind: ind.fitness)


def rank_selection(pop_sorted, rng):
    """Linear rank selection; pop_sorted must be sorted best-first."""
    n = len(pop_sorted)
    ranks = np.arange(n, 0, -1, dtype=float)       # best gets rank n
    probs = ranks / ranks.sum()
    return pop_sorted[rng.choice(n, p=probs)]


# ------------------------------------------------------------------ crossover
def segment_swap_crossover(p1, p2, rng):
    """For every drone, swap a contiguous block of waypoints (and their segment genes)."""
    c1, c2 = p1.copy(), p2.copy()
    D, K = p1.wp.shape[0], p1.wp.shape[1]
    for d in range(D):
        if rng.random() < 0.6 and K > 1:
            i, j = sorted(rng.integers(0, K, size=2))
            j += 1
            c1.wp[d, i:j], c2.wp[d, i:j] = p2.wp[d, i:j].copy(), p1.wp[d, i:j].copy()
            # segment genes between waypoint i and j travel with the block
            for arr in ("speed", "power", "active"):
                a1, a2 = getattr(c1, arr), getattr(c2, arr)
                a1[d, i:j + 1], a2[d, i:j + 1] = getattr(p2, arr)[d, i:j + 1].copy(), getattr(p1, arr)[d, i:j + 1].copy()
    return c1, c2


def geometric_crossover(p1, p2, rng):
    """Blend crossover in space: each child waypoint lies on the line between the parents'."""
    c1, c2 = p1.copy(), p2.copy()
    a = rng.random(p1.wp.shape[:2] + (1,))
    c1.wp = a * p1.wp + (1 - a) * p2.wp
    c2.wp = (1 - a) * p1.wp + a * p2.wp
    b = rng.random(p1.speed.shape)
    c1.speed = b * p1.speed + (1 - b) * p2.speed
    c2.speed = (1 - b) * p1.speed + b * p2.speed
    c1.power = b * p1.power + (1 - b) * p2.power
    c2.power = (1 - b) * p1.power + b * p2.power
    flip = rng.random(p1.active.shape) < 0.5      # uniform crossover for the binary genes
    c1.active = np.where(flip, p1.active, p2.active)
    c2.active = np.where(flip, p2.active, p1.active)
    return c1, c2


def drone_swap_crossover(p1, p2, rng):
    """Exchange the complete route of one randomly chosen drone between the parents."""
    c1, c2 = p1.copy(), p2.copy()
    d = rng.integers(0, p1.wp.shape[0])
    for arr in ("wp", "speed", "power", "active"):
        getattr(c1, arr)[d], getattr(c2, arr)[d] = getattr(p2, arr)[d].copy(), getattr(p1, arr)[d].copy()
    return c1, c2


def crossover(p1, p2, rng):
    """Randomly choose one of the three spatial crossovers."""
    r = rng.random()
    if r < 0.4:
        return segment_swap_crossover(p1, p2, rng)
    if r < 0.8:
        return geometric_crossover(p1, p2, rng)
    return drone_swap_crossover(p1, p2, rng)


# ------------------------------------------------------------------- mutation
def mutate(ind, rate, sigma, mission, rng):
    """Spatial mutation: Gaussian shifts, random reset, waypoint swap, speed/power/sensor tweaks."""
    D, K = ind.wp.shape[0], ind.wp.shape[1]
    n = mission.grid_size
    lo, hi = mission.speed_range
    # 1) Gaussian shift of waypoint coordinates (local search)
    shift = rng.random(ind.wp.shape[:2]) < rate
    ind.wp += shift[..., None] * rng.normal(0.0, sigma, size=ind.wp.shape)
    # 2) random reset of a waypoint (global jump -> keeps diversity)
    reset = rng.random(ind.wp.shape[:2]) < rate / 3
    ind.wp[reset] = rng.uniform(0, n - 1, size=(int(reset.sum()), 2))
    # 3) swap two waypoints of a drone (changes visiting order)
    for d in range(D):
        if rng.random() < rate and K > 1:
            i, j = rng.choice(K, size=2, replace=False)
            ind.wp[d, [i, j]] = ind.wp[d, [j, i]]
    # 4) speed and sensor power perturbation
    m = rng.random(ind.speed.shape) < rate
    ind.speed += m * rng.normal(0.0, 0.4 * (hi - lo) / 3, size=ind.speed.shape)
    m = rng.random(ind.power.shape) < rate
    ind.power += m * rng.normal(0.0, 0.15, size=ind.power.shape)
    # 5) bit-flip on the sensor activation schedule
    f = rng.random(ind.active.shape) < rate / 2
    ind.active = np.where(f, 1 - ind.active, ind.active).astype(np.int8)
    ind.fitness = None
    return ind
