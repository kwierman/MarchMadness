# March Madness MCMC Predictor

A Python package that predicts NCAA March Madness outcomes using **Markov Chain Monte Carlo (MCMC)** simulation with real sports statistics.

<p align="center">
  <img class="center" src="public/Cover.png" alt="March Madness" width="400"/>
</p>

---

## How It Works

### Statistical Model

We use a **Bayesian Bradley-Terry model**:

```
For each game (team i vs team j):
    P(i beats j) = σ(αᵢ − αⱼ)

where:
    αᵢ ~ Normal(μᵢ, 1)   team strength parameters (latent)
    μᵢ  = data-driven prior from net rating, SOS, eFG%, win%
    σ   = sigmoid function
```

### Pipeline

```
1. DATA COLLECTION
   └─ sports-reference.com/cbb  →  team stats (off/def rating, pace, eFG%, etc.)
   └─ ESPN API                  →  schedules, game results
   └─ (offline fallback)        →  realistic synthetic 64-team dataset

2. MCMC INFERENCE  (Metropolis-Hastings)
   └─ Build informative prior from observable team stats
   └─ Likelihood: Bradley-Terry over historical game results
   └─ Sample posterior P(α | games)  over team strength params
   └─ Diagnostics: acceptance rate, trace plots, autocorrelation

3. TOURNAMENT SIMULATION  (Monte Carlo)
   └─ For each simulation:
       • Draw one posterior sample α
       • Simulate all 63 games probabilistically using σ(αᵢ − αⱼ)
       • Record champion, Final Four, Elite 8, Sweet 16
   └─ Aggregate N=10,000 simulations → probabilities

4. VISUALISATION
   └─ Championship probability bar chart
   └─ MCMC posterior density plots (uncertainty bands)
   └─ Round-by-round survival heat-map
   └─ First-round upset probability chart
   └─ Head-to-head win probability matrix
   └─ MCMC trace & autocorrelation diagnostics
```

---

## Quick Start

```python
from march_madness import MarchMadnessPredictor

predictor = MarchMadnessPredictor(season=2025)

# Step 1: Fetch data (auto-falls back to realistic synthetic data if offline)
predictor.load_data()

# Step 2: Run MCMC to estimate team strengths
predictor.fit_mcmc(n_warmup=2000, n_samples=4000)

# Step 3: Simulate the tournament 10,000 times
results = predictor.run_tournament_simulation(n_simulations=10_000)

# Step 4: Print report + save visualisation
predictor.report(results, save_path="march_madness_2025.png")
```

---

## Command-Line Usage

```bash
# Quick run (offline demo, ~60 seconds)
python run_prediction.py

# Full run with live internet data
python run_prediction.py --live

# More simulations = more accurate probabilities
python run_prediction.py --n-sims 50000 --n-warmup 3000 --n-samples 6000

# Custom output path
python run_prediction.py --output my_bracket.png
```

---

## API Reference

### `MarchMadnessPredictor`

| Method | Description |
|--------|-------------|
| `load_data(offline=False)` | Fetch team/game/player stats |
| `fit_mcmc(n_warmup, n_samples, step_size)` | Run M-H MCMC sampler |
| `run_tournament_simulation(n_simulations)` | Simulate N full brackets |
| `report(results, save_path)` | Print summary + save PNG |
| `win_probability(team_a, team_b)` | P(a beats b) from posterior |
| `posterior_summary()` | DataFrame of α estimates per team |

### Direct class access

```python
from march_madness.data.scraper       import DataScraper
from march_madness.models.mcmc        import MCMCModel
from march_madness.utils.bracket      import BracketSimulator
from march_madness.analysis.visualizer import TournamentVisualizer
```

---

## Example Outputs

### Text report
```
══════════════════════════════════════════════════════════════
  🏀  2025 NCAA MARCH MADNESS — MCMC PREDICTION REPORT
       Based on 10,000 Monte Carlo tournament simulations
══════════════════════════════════════════════════════════════

RANK  TEAM          SEED  CONF      CHAMPION  FINAL4  ELITE8  SWEET16
  1   Duke          1     ACC        23.2%    43.0%   56.3%    70.8%
  2   Memphis       5     AAC         9.6%    25.3%   37.2%    57.8%
  3   Tennessee     2     SEC         7.5%    26.2%   39.9%    57.4%
  ...

🏆  Predicted Champion: Duke  (23.2% probability)
```

### Head-to-head queries
```python
p = predictor.win_probability("Duke", "Auburn")
# → 0.681  (Duke has 68% chance of beating Auburn)
```

---

## Data Sources

| Source | URL | Data |
|--------|-----|------|
| Sports Reference | sports-reference.com/cbb | Team stats, schedules |
| ESPN API | site.api.espn.com/... | Live scores, rosters |
| NCAA.com | ncaa.com | Official bracket, seeds |

When network is unavailable, the package generates statistically realistic synthetic data calibrated to historical D1 tournament team distributions.

---

## Package Structure

```
marchmadness/
├── __init__.py
├── predictor.py          ← High-level API
├── run_prediction.py     ← CLI entry point
├── data/
│   ├── scraper.py        ← Web scraping + synthetic fallback
├── models/
│   ├── mcmc.py           ← Bradley-Terry MCMC (Metropolis-Hastings)
├── utils/
│   ├── bracket.py        ← 64-team tournament simulator
└── analysis/
    └── visualizer.py     ← 6-panel matplotlib report
```

---

## MCMC Details

- **Algorithm**: Metropolis-Hastings with random-walk proposals
- **Target acceptance rate**: 25–45% (ideal for this problem)
- **Identifiability**: sum(α) = 0 constraint enforced at each step
- **Prior**: Informative Normal prior built from observable team stats
- **Convergence**: Trace plots and autocorrelation diagnostics included
- **Uncertainty propagation**: Each tournament simulation draws a fresh α sample, fully propagating posterior uncertainty into final probabilities
