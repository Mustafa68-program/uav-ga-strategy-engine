"""The Genetic Algorithm engine (object-oriented)."""
from dataclasses import dataclass, field
import numpy as np
from .chromosome import Individual
from .operators import tournament_selection, rank_selection, crossover, mutate


@dataclass
class GAResult:
    best: Individual
    history: dict                      # per-generation statistics
    snapshots: dict                    # generation -> copy of best individual
    final_population: list
    evaluations: int
    generations_run: int


class GeneticAlgorithm:
    """Generational GA with elitism, adaptive mutation and macro-mutation immigrants.

    Pipeline per generation:
        evaluate -> record statistics -> keep elites -> select parents
        -> crossover -> mutation -> repair -> (immigrants if stagnating)
    Termination: maximum generations OR no improvement for `patience` generations.
    """

    def __init__(self, evaluator, env, mission, cfg, snapshot_gens=()):
        self.ev, self.env, self.m, self.cfg = evaluator, env, mission, cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.snapshot_gens = set(snapshot_gens)

    # -------------------------------------------------------------- utilities
    def _new_random(self):
        return Individual.random(self.m, self.rng).repair(self.env, self.m)

    def _macro_mutant(self, pop_sorted):
        """Immigrant: a strongly mutated copy of one of the better plans (escapes local optima
        while staying in a sensible region of the search space)."""
        top = max(2, len(pop_sorted) // 4)
        parent = pop_sorted[self.rng.integers(0, top)].copy()
        mutate(parent, 0.5, 3.0 * self.cfg.mutation_sigma, self.m, self.rng)
        return parent.repair(self.env, self.m)

    def _select(self, pop_sorted):
        if self.cfg.selection == "rank":
            return rank_selection(pop_sorted, self.rng)
        return tournament_selection(pop_sorted, self.cfg.tournament_size, self.rng)

    @staticmethod
    def diversity(pop, size):
        """Mean standard deviation of the waypoint genes, normalised by map size."""
        wp = np.stack([ind.wp.ravel() for ind in pop])
        return float(wp.std(axis=0).mean() / size)

    # --------------------------------------------------------------------- run
    def run(self):
        cfg = self.cfg
        pop = [self._new_random() for _ in range(cfg.population_size)]
        for ind in pop:
            self.ev.evaluate(ind)
        hist = {k: [] for k in ("best", "mean", "worst", "diversity", "feasible")}
        snaps = {}
        best_ever, stagnation, gens_run = None, 0, 0

        for gen in range(cfg.generations + 1):
            pop.sort(key=lambda i: i.fitness, reverse=True)
            fits = np.array([i.fitness for i in pop])
            hist["best"].append(float(fits[0]))
            hist["mean"].append(float(fits.mean()))
            hist["worst"].append(float(fits[-1]))
            hist["diversity"].append(self.diversity(pop, self.m.grid_size))
            hist["feasible"].append(float(np.mean([i.components["feasible"] for i in pop])))
            if best_ever is None or fits[0] > best_ever.fitness + 1e-9:
                best_ever, stagnation = pop[0].copy(), 0
            else:
                stagnation += 1
            if gen in self.snapshot_gens:
                snaps[gen] = pop[0].copy()
            gens_run = gen
            if gen == cfg.generations or stagnation >= cfg.patience:
                break

            # adaptive control: more mutation when diversity collapses, annealed step size
            rate, sigma = cfg.mutation_rate, cfg.mutation_sigma
            if cfg.adaptive:
                if hist["diversity"][-1] < 0.06:
                    rate = min(0.6, rate * 2.0)
                sigma = cfg.mutation_sigma * (1.0 - 0.6 * gen / cfg.generations)

            n_imm = 0
            if cfg.adaptive and stagnation >= cfg.stagnation_trigger:
                n_imm = max(1, int(cfg.immigrant_fraction * cfg.population_size))
            n_children = cfg.population_size - cfg.elite_count - n_imm

            children = []
            while len(children) < n_children:
                p1, p2 = self._select(pop), self._select(pop)
                if self.rng.random() < cfg.crossover_rate:
                    c1, c2 = crossover(p1, p2, self.rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for c in (c1, c2):
                    mutate(c, rate, sigma, self.m, self.rng)
                    c.repair(self.env, self.m)
                    children.append(c)
            children = children[:n_children]
            immigrants = [self._macro_mutant(pop) for _ in range(n_imm)]
            for ind in children + immigrants:
                self.ev.evaluate(ind)
            pop = [i.copy() for i in pop[:cfg.elite_count]] + children + immigrants

        pop.sort(key=lambda i: i.fitness, reverse=True)
        if gens_run not in snaps:
            snaps[gens_run] = best_ever.copy()
        return GAResult(best_ever, hist, snaps, pop, self.ev.evaluations, gens_run)
