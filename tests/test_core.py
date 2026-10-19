"""Unit tests for the UAV GA project. Run with:  pytest -q"""
import numpy as np
import pytest

from uav_ga import (Environment, FitnessEvaluator, GAConfig, GeneticAlgorithm, Individual,
                    MissionConfig, Penalties, Weights, random_search)
from uav_ga.operators import crossover, mutate, tournament_selection


@pytest.fixture(scope="module")
def setup():
    env, m = Environment(), MissionConfig()
    return env, m, FitnessEvaluator(env, m, Weights(), Penalties())


def test_environment_shapes(setup):
    env, m, _ = setup
    assert env.priority.shape == (m.grid_size, m.grid_size)
    assert env.hazard.shape == (m.grid_size, m.grid_size)
    assert 0.0 <= env.hazard.min() and env.hazard.max() <= 1.0
    assert env.targets.shape[1] == 2


def test_no_fly_detection(setup):
    env, _, _ = setup
    z = env.no_fly[0]
    assert env.inside_no_fly(z.cx, z.cy)
    assert not env.inside_no_fly(0.0, 49.0)


def test_repair_enforces_hard_constraints(setup):
    env, m, _ = setup
    rng = np.random.default_rng(1)
    for _ in range(50):
        ind = Individual.random(m, rng).repair(env, m)
        assert ind.wp.min() >= 0 and ind.wp.max() <= m.grid_size - 1
        assert not env.inside_no_fly(ind.wp[..., 0], ind.wp[..., 1]).any()
        assert ind.speed.min() >= m.speed_range[0] and ind.speed.max() <= m.speed_range[1]
        assert ind.power.min() >= 0 and ind.power.max() <= 1


def test_fitness_is_deterministic_and_finite(setup):
    env, m, ev = setup
    ind = Individual.random(m, np.random.default_rng(2)).repair(env, m)
    f1 = ev.evaluate(ind)
    f2 = ev.evaluate(ind)
    assert f1 == f2 and np.isfinite(f1)
    for key in ("coverage", "detection"):
        assert 0.0 <= ind.components[key] <= 1.0


def test_sensor_off_means_no_coverage(setup):
    env, m, ev = setup
    ind = Individual.random(m, np.random.default_rng(3)).repair(env, m)
    ind.active[:] = 0
    ev.evaluate(ind)
    assert ind.components["coverage"] == 0.0 and ind.components["detection"] == 0.0


def test_crossover_preserves_structure(setup):
    env, m, _ = setup
    rng = np.random.default_rng(4)
    p1, p2 = Individual.random(m, rng), Individual.random(m, rng)
    for _ in range(30):
        c1, c2 = crossover(p1, p2, rng)
        assert c1.wp.shape == p1.wp.shape and c2.speed.shape == p1.speed.shape
        assert set(np.unique(c1.active)) <= {0, 1}


def test_mutation_changes_individual(setup):
    env, m, _ = setup
    rng = np.random.default_rng(5)
    ind = Individual.random(m, rng).repair(env, m)
    before = ind.wp.copy()
    mutate(ind, rate=1.0, sigma=3.0, mission=m, rng=rng)
    assert not np.allclose(before, ind.wp)


def test_tournament_prefers_fitter(setup):
    env, m, ev = setup
    rng = np.random.default_rng(6)
    pop = [Individual.random(m, rng).repair(env, m) for _ in range(30)]
    for i in pop:
        ev.evaluate(i)
    picks = [tournament_selection(pop, 5, rng).fitness for _ in range(200)]
    assert np.mean(picks) > np.mean([i.fitness for i in pop])


def test_ga_improves_and_beats_random_search(setup):
    env, m, _ = setup
    ev = FitnessEvaluator(env, m, Weights(), Penalties())
    cfg = GAConfig(population_size=30, generations=30, seed=0)
    res = GeneticAlgorithm(ev, env, m, cfg).run()
    assert res.history["best"][-1] > res.history["best"][0]
    ev_rs = FitnessEvaluator(env, m, Weights(), Penalties())
    rs, _ = random_search(ev_rs, env, m, res.evaluations, seed=1)
    assert res.best.fitness > rs.fitness
