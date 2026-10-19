"""Chromosome encoding: a hybrid real-valued / integer-binary multi-segment string."""
import numpy as np


class Individual:
    """One candidate mission plan for the whole fleet.

    For each of D drones the route is  base -> w_1 -> ... -> w_K -> base,
    i.e. K + 1 flight segments. The chromosome stores, per drone:

      wp     (K, 2)    real   waypoint coordinates (x, y) in grid units
      speed  (K + 1,)  real   cruise speed on every segment
      power  (K + 1,)  real   sensor power level (0..1) on every segment
      active (K + 1,)  binary sensor ON/OFF flag on every segment (activation schedule)

    so the full genome has D * (2K + 3(K + 1)) genes of mixed type.
    """

    __slots__ = ("wp", "speed", "power", "active", "fitness", "components")

    def __init__(self, wp, speed, power, active):
        self.wp = wp
        self.speed = speed
        self.power = power
        self.active = active
        self.fitness = None
        self.components = None

    # -------------------------------------------------------------- factories
    @staticmethod
    def random(mission, rng):
        D, K, n = mission.n_drones, mission.n_waypoints, mission.grid_size
        lo, hi = mission.speed_range
        wp = rng.uniform(0, n - 1, size=(D, K, 2))
        speed = rng.uniform(lo, hi, size=(D, K + 1))
        power = rng.uniform(0.0, 1.0, size=(D, K + 1))
        active = (rng.random((D, K + 1)) < 0.85).astype(np.int8)
        return Individual(wp, speed, power, active)

    def copy(self):
        c = Individual(self.wp.copy(), self.speed.copy(), self.power.copy(), self.active.copy())
        c.fitness = self.fitness
        c.components = self.components
        return c

    def genes(self):
        """Flatten the real-valued genes (used for diversity measurements)."""
        return np.concatenate([self.wp.ravel(), self.speed.ravel(), self.power.ravel()])

    # ------------------------------------------------------------------ repair
    def repair(self, env, mission):
        """Constraint repair mechanism (applied after every variation operator).

        1. bounds       - clip all genes into their legal ranges;
        2. range limit  - if a route is longer than max_range, scale it towards the base
                          (route length scales linearly, so one step is enough);
        3. no-fly zones - push waypoints lying inside a restricted circle to its boundary;
        4. battery      - if a route needs more energy than the battery holds, reduce speeds.
        Violations that cannot be repaired (e.g. a straight leg crossing a no-fly zone,
        or the mission deadline) are left to the penalty terms of the fitness function.
        """
        n = mission.grid_size
        base = np.asarray(mission.base, dtype=float)
        np.clip(self.wp, 0, n - 1, out=self.wp)
        np.clip(self.speed, mission.speed_range[0], mission.speed_range[1], out=self.speed)
        np.clip(self.power, 0.0, 1.0, out=self.power)

        for d in range(self.wp.shape[0]):
            # --- range repair
            pts = np.vstack([base, self.wp[d], base])
            length = np.hypot(*(pts[1:] - pts[:-1]).T).sum()
            if length > mission.max_range * 0.99:
                self.wp[d] = base + (self.wp[d] - base) * (mission.max_range * 0.99 / length)
            # --- no-fly zone repair
            for k in range(self.wp.shape[1]):
                x, y = env.push_out_of_no_fly(self.wp[d, k, 0], self.wp[d, k, 1])
                self.wp[d, k, 0] = min(max(x, 0), n - 1)
                self.wp[d, k, 1] = min(max(y, 0), n - 1)
            # --- battery repair (slow down until the route is affordable)
            pts = np.vstack([base, self.wp[d], base])
            seg = np.hypot(*(pts[1:] - pts[:-1]).T)
            v, pw, on = self.speed[d], self.power[d], self.active[d]
            for _ in range(15):
                energy = (seg * (mission.e0 + mission.e1 * v ** 2)).sum() \
                         + (mission.sensor_cost * pw * on * seg / v).sum()
                if energy <= mission.battery * 0.99:
                    break
                v = np.maximum(v * 0.92, mission.speed_range[0])
            self.speed[d] = v
        self.fitness = None
        self.components = None
        return self
