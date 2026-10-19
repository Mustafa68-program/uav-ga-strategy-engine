"""The hazard grid: priority zones, hazard pockets, no-fly zones and targets."""
from dataclasses import dataclass
import numpy as np


@dataclass
class NoFlyZone:
    cx: float
    cy: float
    r: float


class Environment:
    """A discretised 2-D map (grid_size x grid_size cells).

    Each cell (x, y) stores:
      * priority  - importance of observing that cell (disaster zones are high),
      * hazard    - risk level in [0, 1] (high wind / radiation pockets).
    No-fly zones are circles; targets are points to be detected.
    The scenario is deterministic for a given seed so results are reproducible.
    """

    def __init__(self, grid_size: int = 50, seed: int = 42, n_targets: int = 20):
        self.size = grid_size
        n = grid_size
        ys, xs = np.mgrid[0:n, 0:n]
        self.X = xs.astype(float)
        self.Y = ys.astype(float)

        # (cx, cy, sigma, weight) - disaster zones raise the priority around them
        self.disaster_zones = [(38, 40, 5.5, 5.0), (14, 36, 5.0, 4.0),
                               (42, 12, 4.5, 3.5), (26, 24, 4.0, 2.5)]
        priority = np.full((n, n), 0.3)
        for cx, cy, s, w in self.disaster_zones:
            priority += w * np.exp(-((self.X - cx) ** 2 + (self.Y - cy) ** 2) / (2 * s ** 2))
        self.priority = priority

        # (cx, cy, sigma, intensity) - wind / radiation pockets
        self.hazard_pockets = [(30, 32, 6.0, 0.9), (12, 18, 5.0, 0.7),
                               (46, 26, 4.0, 0.8), (24, 44, 4.0, 0.6)]
        hazard = np.zeros((n, n))
        for cx, cy, s, h in self.hazard_pockets:
            hazard += h * np.exp(-((self.X - cx) ** 2 + (self.Y - cy) ** 2) / (2 * s ** 2))
        self.hazard = np.clip(hazard, 0.0, 1.0)

        # circular no-fly zones (restricted airspace)
        self.no_fly = [NoFlyZone(22, 12, 5.0), NoFlyZone(32, 46, 3.5), NoFlyZone(8, 28, 4.0)]

        self.base = np.array([2.0, 2.0])

        # targets are more likely to appear where the priority is high
        rng = np.random.default_rng(seed)
        p = priority.copy()
        p[self.inside_no_fly(self.X, self.Y)] = 0.0
        p[(self.X < 6) & (self.Y < 6)] = 0.0
        p = (p / p.sum()).ravel()
        idx = rng.choice(n * n, size=n_targets, replace=False, p=p)
        self.targets = np.stack([idx % n, idx // n], axis=1).astype(float)
        self.target_weight = self.priority[self.targets[:, 1].astype(int), self.targets[:, 0].astype(int)]

    # ------------------------------------------------------------------ helpers
    def inside_no_fly(self, x, y):
        """Boolean array: True where (x, y) is inside any no-fly zone."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.zeros(np.broadcast(x, y).shape, dtype=bool)
        for z in self.no_fly:
            mask |= (x - z.cx) ** 2 + (y - z.cy) ** 2 < z.r ** 2
        return mask

    def hazard_at(self, x, y):
        """Hazard value at (possibly fractional) coordinates (nearest cell)."""
        xi = np.clip(np.rint(x).astype(int), 0, self.size - 1)
        yi = np.clip(np.rint(y).astype(int), 0, self.size - 1)
        return self.hazard[yi, xi]

    def push_out_of_no_fly(self, x, y):
        """Repair: move a point lying inside a no-fly circle to just outside it."""
        for z in self.no_fly:
            dx, dy = x - z.cx, y - z.cy
            d = float(np.hypot(dx, dy))
            if d < z.r:
                if d < 1e-9:
                    dx, dy, d = 1.0, 0.0, 1.0
                x = z.cx + dx / d * (z.r + 0.5)
                y = z.cy + dy / d * (z.r + 0.5)
        return x, y
