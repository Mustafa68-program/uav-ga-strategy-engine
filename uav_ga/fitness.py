"""Multi-objective fitness: flight simulation + weighted score - constraint penalties."""
import numpy as np
from .config import MissionConfig, Weights, Penalties


class FitnessEvaluator:
    """Evaluates an Individual.

        F = w_cov*Coverage + w_det*Detection
            - w_en*Energy - w_t*Time - w_haz*Hazard
            - penalties(range, battery, no-fly, deadline)

    Every term is normalised so the weights are directly comparable.
    """

    def __init__(self, env, mission: MissionConfig, weights: Weights, penalties: Penalties):
        self.env, self.m, self.w, self.p = env, mission, weights, penalties
        self.GX = env.X.ravel()
        self.GY = env.Y.ravel()
        self.pri = env.priority.ravel()
        self.pri_sum = self.pri.sum()
        self.tx = env.targets[:, 0]
        self.ty = env.targets[:, 1]
        self.tw = env.target_weight
        self.evaluations = 0
        self._disks = {}

    # ------------------------------------------------------------ flight model
    def simulate(self, ind):
        """Fly every route (vectorised); return sensing samples and per-drone resource usage."""
        m, env = self.m, self.env
        D, K = ind.wp.shape[0], ind.wp.shape[1]
        base = np.broadcast_to(np.asarray(m.base, dtype=float), (D, 1, 2))
        pts = np.concatenate([base, ind.wp, base], axis=1)            # (D, K+2, 2)
        A = pts[:, :-1, :].reshape(-1, 2)                             # segment starts
        B = pts[:, 1:, :].reshape(-1, 2)                              # segment ends
        seg_drone = np.repeat(np.arange(D), K + 1)
        diff = B - A
        dist = np.hypot(diff[:, 0], diff[:, 1])
        v = ind.speed.ravel()
        power = ind.power.ravel()
        on = ind.active.ravel().astype(float)

        n = np.maximum(1, np.ceil(dist / m.sample_step).astype(int))  # samples per segment
        seg_idx = np.repeat(np.arange(n.size), n)
        local = np.arange(seg_idx.size) - (np.cumsum(n) - n)[seg_idx]
        t = (local + 0.5) / n[seg_idx]
        x = A[seg_idx, 0] + diff[seg_idx, 0] * t
        y = A[seg_idx, 1] + diff[seg_idx, 1] * t
        sl = (dist / n)[seg_idx]                                      # length each sample covers
        dt = sl / v[seg_idx]                                          # time each sample takes
        drone_s = seg_drone[seg_idx]

        e_s = sl * (m.e0 + m.e1 * v[seg_idx] ** 2) + m.sensor_cost * power[seg_idx] * on[seg_idx] * dt
        per = lambda w: np.bincount(drone_s, weights=w, minlength=D)
        active = on[seg_idx] > 0.5
        r = m.radius_range[0] + (m.radius_range[1] - m.radius_range[0]) * power[seg_idx]
        return dict(sx=x[active], sy=y[active], sr=r[active], sp=power[seg_idx][active],
                    range=np.bincount(seg_drone, weights=dist, minlength=D),
                    energy=per(e_s), time=per(dt),
                    hazard=per(env.hazard_at(x, y) * dt),
                    nofly=per(env.inside_no_fly(x, y) * sl))

    def coverage_mask(self, sim):
        """Boolean grid of cells observed by at least one sensor footprint (disk stamping)."""
        n = self.env.size
        mask = np.zeros((n, n), dtype=bool)
        if sim["sx"].size == 0:
            return mask
        cx = np.clip(np.rint(sim["sx"]).astype(int), 0, n - 1)
        cy = np.clip(np.rint(sim["sy"]).astype(int), 0, n - 1)
        level = np.rint(sim["sr"] * 2).astype(int)                    # radius quantised to 0.5 cells
        for lv in np.unique(level):
            ox, oy = self._disk(lv)
            sel = level == lv
            ix = cx[sel][:, None] + ox[None, :]
            iy = cy[sel][:, None] + oy[None, :]
            ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
            mask[iy[ok], ix[ok]] = True
        return mask

    def _disk(self, level):
        if level not in self._disks:
            r = level / 2.0
            k = int(np.ceil(r))
            oy, ox = np.mgrid[-k:k + 1, -k:k + 1]
            keep = ox ** 2 + oy ** 2 <= r * r
            self._disks[level] = (ox[keep].astype(int), oy[keep].astype(int))
        return self._disks[level]

    # ---------------------------------------------------------------- fitness
    def evaluate(self, ind):
        self.evaluations += 1
        m, w, p = self.m, self.w, self.p
        sim = self.simulate(ind)
        mask = self.coverage_mask(sim).ravel()
        coverage = float(self.pri[mask].sum() / self.pri_sum)

        if sim["sx"].size:
            d2 = (self.tx[:, None] - sim["sx"][None, :]) ** 2 + (self.ty[:, None] - sim["sy"][None, :]) ** 2
            inr = d2 <= sim["sr"][None, :] ** 2
            prob = np.where(inr, 0.5 + 0.5 * sim["sp"][None, :], 0.0).max(axis=1)
            detection = float((prob * self.tw).sum() / self.tw.sum())
        else:
            detection = 0.0

        D = ind.wp.shape[0]
        energy = float(sim["energy"].sum()) / (D * m.battery)
        makespan = float(sim["time"].max())
        time_n = makespan / m.time_ref
        hazard = float(sim["hazard"].sum()) / m.hazard_ref

        v_range = float(np.maximum(0, sim["range"] - m.max_range).sum()) / m.max_range
        v_batt = float(np.maximum(0, sim["energy"] - m.battery).sum()) / m.battery
        v_nfz = float(sim["nofly"].sum()) / m.max_range
        v_dead = max(0.0, makespan - m.deadline) / m.deadline
        penalty = p.range * v_range + p.battery * v_batt + p.no_fly * v_nfz + p.deadline * v_dead

        fit = (w.coverage * coverage + w.detection * detection
               - w.energy * energy - w.time * time_n - w.hazard * hazard - penalty)
        ind.fitness = fit
        ind.components = dict(coverage=coverage, detection=detection, energy=energy,
                              time=time_n, hazard=hazard, penalty=penalty,
                              makespan=makespan, total_range=float(sim["range"].sum()),
                              feasible=bool(penalty <= 1e-9))
        return fit
