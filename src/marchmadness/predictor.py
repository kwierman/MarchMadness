"""
predictor.py
------------
High-level API: data → MCMC → simulation → visualisation.

Quick start
-----------
    from march_madness import MarchMadnessPredictor

    predictor = MarchMadnessPredictor(season=2025)
    predictor.load_data()
    predictor.fit_mcmc(n_warmup=2000, n_samples=4000)
    results = predictor.run_tournament_simulation(n_simulations=10_000)
    predictor.report(results, save_path="march_madness_report.png")
"""

import pandas as pd
import numpy as np
from typing import Optional, TYPE_CHECKING

from .data.scraper import DataScraper
from .models.mcmc import MCMCModel
from .utils.bracket import BracketSimulator
from .analysis.visualizer import TournamentVisualizer

if TYPE_CHECKING:
    from .config import Config


class MarchMadnessPredictor:
    """
    End-to-end March Madness prediction pipeline.

    Parameters
    ----------
    season   : int   — season end year (default 2025)
    cache_dir: str   — local directory to cache fetched data
    verbose  : bool  — print progress messages
    config   : Config, optional — configuration object
    """

    def __init__(
        self,
        season: int = 2025,
        cache_dir: str = ".mm_cache",
        verbose: bool = True,
        config: Optional["Config"] = None,
    ):
        self._config = config

        if config is not None:
            self.season = config.season
            self.cache_dir = config.data.cache_dir
            self.verbose = verbose
            self._model_config = config.model
            self._visualizer_config = config.visualizer
        else:
            self.season = season
            self.cache_dir = cache_dir
            self.verbose = verbose
            self._model_config = None
            self._visualizer_config = None

        self._scraper: DataScraper | None = None
        self._mcmc: MCMCModel | None = None
        self._simulator: BracketSimulator | None = None
        self._viz: TournamentVisualizer | None = None

        self.data: dict[str, pd.DataFrame] | None = None
        self.teams_df: pd.DataFrame | None = None
        self.games_df: pd.DataFrame | None = None
        self.players_df: pd.DataFrame | None = None
        self.sim_results: dict | None = None

    # ── Step 1: Load data ────────────────────────────────────────────────

    def load_data(self, offline: Optional[bool] = None) -> "MarchMadnessPredictor":
        """
        Fetch team, game, and player data.
        Set offline=True to always use synthetic data (useful for testing).
        Uses config offline_fallback if not explicitly set.
        """
        if offline is None:
            offline = self._config.data.offline_fallback if self._config else False

        self._scraper = DataScraper(
            season=self.season,
            cache_dir=self.cache_dir,
            config=self._config,
        )
        self.data = self._scraper.fetch_all(offline=offline)
        self.teams_df = self.data["teams"]
        self.games_df = self.data["games"]
        self.players_df = self.data["players"]

        if self.verbose:
            print(
                f"[Predictor] Loaded {len(self.teams_df)} teams, "
                f"{len(self.games_df)} games, "
                f"{len(self.players_df)} player records."
            )
        return self

    # ── Step 2: Fit MCMC model ────────────────────────────────────────────

    def fit_mcmc(
        self,
        n_warmup: Optional[int] = None,
        n_samples: Optional[int] = None,
        step_size: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> "MarchMadnessPredictor":
        """
        Run Metropolis-Hastings MCMC to estimate team strength posteriors.
        Uses config values if parameters not explicitly provided.
        """
        if self.teams_df is None:
            raise RuntimeError("Call load_data() before fit_mcmc().")

        if self._model_config:
            n_warmup = (
                n_warmup if n_warmup is not None else self._model_config.mcmc_warmup
            )
            n_samples = (
                n_samples if n_samples is not None else self._model_config.mcmc_samples
            )
            step_size = (
                step_size
                if step_size is not None
                else self._model_config.mcmc_step_size
            )
            seed = seed if seed is not None else self._model_config.mcmc_seed
        else:
            n_warmup = n_warmup if n_warmup is not None else 2000
            n_samples = n_samples if n_samples is not None else 4000
            step_size = step_size if step_size is not None else 0.08
            seed = seed if seed is not None else 0

        self._mcmc = MCMCModel(
            teams_df=self.teams_df,
            games_df=self.games_df,
            n_warmup=n_warmup,
            n_samples=n_samples,
            step_size=step_size,
            seed=seed,
        )
        self._mcmc.fit(verbose=self.verbose)
        return self

    # ── Step 3: Run tournament simulation ────────────────────────────────

    def run_tournament_simulation(
        self,
        n_simulations: Optional[int] = None,
        seed: int = 42,
    ) -> dict:
        """
        Simulate the full 64-team tournament N times using posterior draws.
        Uses config n_simulations if not explicitly provided.

        Returns dict with championship/FF/E8/S16 probabilities.
        """
        if self._mcmc is None or self._mcmc.alpha_samples is None:
            raise RuntimeError("Call fit_mcmc() before run_tournament_simulation().")

        if n_simulations is None:
            n_simulations = (
                self._model_config.n_simulations if self._model_config else 10_000
            )

        self._simulator = BracketSimulator(
            teams_df=self.teams_df,
            alpha_samples=self._mcmc.alpha_samples,
            team_names=self._mcmc.team_names,
            seed=seed,
        )
        self.sim_results = self._simulator.run_simulations(
            n_simulations=n_simulations, verbose=self.verbose
        )
        return self.sim_results

    # ── Step 4: Reporting ─────────────────────────────────────────────────

    def report(
        self,
        sim_results: dict | None = None,
        save_path: str | None = "march_madness_2025.png",
        top_n: int = 16,
    ) -> None:
        """
        Print a text summary and save the visualisation PNG.
        """
        if sim_results is None:
            sim_results = self.sim_results
        if sim_results is None:
            raise RuntimeError(
                "No simulation results. Run run_tournament_simulation()."
            )

        self._print_summary(sim_results, top_n=top_n)

        posterior_df = self._mcmc.posterior_summary() if self._mcmc else pd.DataFrame()

        self._viz = TournamentVisualizer(
            teams_df=self.teams_df,
            sim_results=sim_results,
            posterior_df=posterior_df,
            alpha_samples=self._mcmc.alpha_samples,
            team_names=self._mcmc.team_names,
        )
        fig = self._viz.plot_all(save_path=save_path)
        return fig

    def _print_summary(self, sim_results: dict, top_n: int = 16) -> None:
        n = sim_results["n_simulations"]
        print("\n" + "═" * 62)
        print(f"  🏀  2025 NCAA MARCH MADNESS — MCMC PREDICTION REPORT")
        print(f"       Based on {n:,} Monte Carlo tournament simulations")
        print("═" * 62)

        print(
            f"\n{'RANK':<5} {'TEAM':<22} {'SEED':<6} {'CONF':<10} "
            f"{'CHAMPION':>10} {'FINAL4':>8} {'ELITE8':>8} {'SWEET16':>8}"
        )
        print("─" * 80)

        champ_probs = sim_results["championship_probs"]
        ff_probs = sim_results["final_four_probs"]
        e8_probs = sim_results["elite_eight_probs"]
        s16_probs = sim_results["sweet_16_probs"]

        seed_map = dict(zip(self.teams_df["team"], self.teams_df["seed"]))
        conf_map = dict(zip(self.teams_df["team"], self.teams_df["conference"]))

        for rank, team in enumerate(champ_probs.head(top_n).index, 1):
            seed = seed_map.get(team, "?")
            conf = conf_map.get(team, "?")
            cp = champ_probs.get(team, 0) * 100
            fp = ff_probs.get(team, 0) * 100
            ep = e8_probs.get(team, 0) * 100
            sp = s16_probs.get(team, 0) * 100
            print(
                f"  {rank:<4} {team:<22} {seed!s:<6} {conf:<10} "
                f"{cp:>9.1f}% {fp:>7.1f}% {ep:>7.1f}% {sp:>7.1f}%"
            )

        best = champ_probs.idxmax()
        best_p = champ_probs.max() * 100
        print("\n" + "═" * 62)
        print(f"  🏆  Predicted Champion: {best}  ({best_p:.1f}% probability)")
        print("═" * 62 + "\n")

    # ── Convenience: win probability lookup ───────────────────────────────

    def win_probability(self, team_a: str, team_b: str) -> float:
        """Return P(team_a beats team_b) from MCMC posterior."""
        if self._mcmc is None:
            raise RuntimeError("Call fit_mcmc() first.")
        return self._mcmc.win_probability(team_a, team_b)

    def posterior_summary(self) -> pd.DataFrame:
        """Return team strength posterior summary table."""
        if self._mcmc is None:
            raise RuntimeError("Call fit_mcmc() first.")
        return self._mcmc.posterior_summary()
