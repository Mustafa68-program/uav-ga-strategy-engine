"""All tunable parameters live here (dataclasses) so experiments are reproducible."""
from dataclasses import dataclass


@dataclass
class MissionConfig:
    """Physical description of the mission and the UAV fleet."""
    grid_size: int = 50              # the discretised map is grid_size x grid_size cells
    n_drones: int = 3                # fleet size
    n_waypoints: int = 6             # free waypoints per drone (base -> w1..wK -> base)
    base: tuple = (2.0, 2.0)         # launch / landing position
    speed_range: tuple = (1.0, 4.0)  # cells per time unit
    radius_range: tuple = (1.5, 5.0) # sensor footprint radius (cells) for power 0 -> 1
    # --- energy model: E = distance * (e0 + e1 * v^2) + sensor_cost * power * time ---
    e0: float = 0.5
    e1: float = 0.25
    sensor_cost: float = 0.8
    # --- hard constraints ---
    max_range: float = 140.0         # maximum flight distance per drone (cells)
    battery: float = 300.0           # battery capacity per drone (energy units)
    deadline: float = 80.0           # mission deadline (time units)
    # --- normalisation constants used inside the fitness function ---
    time_ref: float = 100.0
    hazard_ref: float = 40.0
    sample_step: float = 1.0         # path discretisation step (cells)


@dataclass
class Weights:
    """Multi-objective weights (sum of positive terms is 0.55, negative terms 0.45)."""
    coverage: float = 0.35     # maximise priority-weighted coverage
    detection: float = 0.20    # maximise target detection rate
    energy: float = 0.15       # minimise energy
    time: float = 0.10         # minimise mission time (makespan)
    hazard: float = 0.20       # minimise hazard exposure


@dataclass
class Penalties:
    """Penalty coefficients for violated hard constraints."""
    range: float = 2.0
    battery: float = 2.0
    no_fly: float = 4.0
    deadline: float = 2.0


@dataclass
class GAConfig:
    """Genetic algorithm hyper-parameters."""
    population_size: int = 80
    generations: int = 150
    selection: str = "tournament"    # "tournament" or "rank"
    tournament_size: int = 3         # selection pressure
    crossover_rate: float = 0.85
    mutation_rate: float = 0.03      # per-gene mutation probability (tuned, see report)
    mutation_sigma: float = 4.0      # std of Gaussian waypoint perturbation (cells)
    elite_count: int = 2
    adaptive: bool = True            # adaptive mutation + random immigrants
    immigrant_fraction: float = 0.10
    stagnation_trigger: int = 8      # generations without improvement before immigrants
    patience: int = 60               # early stop after this many stagnant generations
    seed: int = 0
