# UAV Strategy Engine – Genetic Algorithm for Multi-Objective UAV Fleet Planning

An object-oriented Python implementation of a **Genetic Algorithm (GA)** that plans the
waypoints, speeds and sensor schedules of a fleet of UAVs surveying a hazard grid.
The plan must balance **coverage and detection** (maximise) against **energy, flight time and
hazard exposure** (minimise) while respecting **hard constraints** (range, battery, deadline,
no-fly zones).

*Course: Artificial Intelligence (BSCS 5), Bahria University Lahore Campus – Assignment 2 (CLO4).*

## 1. Project structure

```
uav-ga-strategy-engine/
├── main.py                  # command-line entry point (run / sensitivity / baseline / all)
├── requirements.txt
├── README.md
├── uav_ga/
│   ├── config.py            # MissionConfig, Weights, Penalties, GAConfig (all parameters)
│   ├── environment.py       # 50x50 grid: priority map, hazard map, no-fly zones, targets
│   ├── chromosome.py        # Individual: hybrid real/binary chromosome + repair mechanism
│   ├── fitness.py           # flight simulation + multi-objective fitness + penalties
│   ├── operators.py         # selection, crossover (3 kinds), mutation (5 kinds)
│   ├── genetic_algorithm.py # GA engine: elitism, adaptive mutation, immigrants, termination
│   ├── baseline.py          # random search baseline
│   ├── experiments.py       # sensitivity study + GA-vs-random comparison
│   └── visualization.py     # convergence, path, coverage, trade-off plots
├── tests/test_core.py       # unit tests (pytest)
└── results/                 # figures and JSON produced by main.py
```

## 2. Installation

```bash
git clone https://github.com/<your-username>/uav-ga-strategy-engine.git
cd uav-ga-strategy-engine
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9+ is required. Only NumPy and Matplotlib are used (pytest for the tests).

## 3. How to run

```bash
python main.py run            # one GA run + convergence / path / coverage plots   (~15 s)
python main.py sensitivity    # population size, mutation rate, selection pressure (~4 min)
python main.py baseline       # GA vs random search over 10 independent runs        (~10 min)
python main.py all            # everything
python main.py run --quick    # tiny budget, useful as a smoke test
python main.py run --seed 3   # different random seed
pytest -q                     # run the unit tests
```

All figures and JSON files are written to `results/`.

## 4. Design summary

**Environment.** The mission area is a 50 x 50 grid. Each cell has a *priority* (disaster
zones are high), a *hazard* value in [0, 1] (wind / radiation pockets), 3 circular
*no-fly zones* and 20 *targets* to be detected. The scenario is deterministic (seed 42).

**Chromosome.** A fleet plan of D = 3 drones. For each drone the route is
`base -> w1 ... w6 -> base` (7 segments). Genes per drone:

| Gene | Type | Range | Meaning |
|---|---|---|---|
| waypoints (6 x 2) | real | 0..49 | (x, y) of each waypoint |
| speed (7) | real | 1..4 | cruise speed per segment |
| sensor power (7) | real | 0..1 | sensor power per segment (sets footprint radius 1.5..5 cells) |
| sensor active (7) | binary | {0, 1} | sensor activation schedule |

That is 33 genes per drone, **99 genes for the fleet (78 real-valued and 21 binary)**: a hybrid real / binary encoding.

**Fitness.** `F = 0.35*Coverage + 0.20*Detection - 0.15*Energy - 0.10*Time - 0.20*Hazard - Penalties`.
Every term is normalised; see `uav_ga/fitness.py` for the exact definitions.

**Operators.** Tournament (or rank) selection; crossover chosen at random from *segment swap*,
*geometric (blend)* and *drone-route swap*; mutation = Gaussian waypoint shift, random waypoint
reset, waypoint swap, speed/power perturbation and sensor bit-flip. Elitism, annealed step size,
adaptive mutation rate when diversity collapses, and macro-mutation immigrants when the search
stagnates keep the population diverse and avoid premature convergence.

**Constraints.** *Repair:* clip to bounds, shrink over-long routes towards the base, push
waypoints out of no-fly zones, slow drones down until the battery suffices. *Penalties:* leftover
violations (flight legs crossing a no-fly zone, mission deadline, any residual range or battery
excess) are subtracted from the fitness.

**Termination.** Maximum number of generations, or no improvement for `patience` generations.

## 5. Customising

Edit the dataclasses in `uav_ga/config.py` (fleet size, grid size, weights, penalties, GA
hyper-parameters). Edit `Environment` in `uav_ga/environment.py` to change disaster zones,
hazard pockets or no-fly zones.

## 6. Outputs

| File | Content |
|---|---|
| `results/fig1_environment.png` | the hazard grid |
| `results/fig2_convergence.png` | best vs average fitness per generation |
| `results/fig3_paths_and_coverage.png` | evolved flight paths and coverage heat-maps across generations |
| `results/fig4_sensitivity.png` | hyper-parameter sensitivity curves |
| `results/fig5_baseline.png` | GA vs random search |
| `results/fig6_tradeoff.png` | coverage / energy / hazard trade-off in the final population |
| `results/fig7_solution_diversity.png` | best plans from different seeds |
| `results/*.json` | numeric results, configuration used, best solution |

## 7. Key results (reproduced by `python main.py all`)

| Method (10 independent runs, same evaluation budget ≈ 11,800) | Best fitness | Coverage | Detection | Hazard (lower is better) |
|---|---|---|---|---|
| Genetic algorithm | 0.238 ± 0.017 | 84% | 97% | 0.44 |
| Random search | -0.138 ± 0.071 | 51% | 61% | 1.04 |

Hyper-parameter sensitivity (final best fitness, 100 generations, 3 seeds):
population size 20 / 80 / 200 = 0.186 / 0.220 / 0.227;
mutation rate 0.005 / 0.03 / 0.3 = 0.194 / 0.220 / -0.000;
tournament size 1 / 3 / 8 = -0.042 / 0.220 / 0.250.
Details and discussion are in the technical report.

