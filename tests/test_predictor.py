import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestMarchMadnessPredictor:
    """Tests for the MarchMadnessPredictor class."""

    def test_init_default(self):
        """Test MarchMadnessPredictor initialization with defaults."""
        from marchmadness import MarchMadnessPredictor

        predictor = MarchMadnessPredictor()

        assert predictor.season == 2025
        assert predictor.cache_dir == ".mm_cache"
        assert predictor.verbose is True

    def test_init_custom(self):
        """Test MarchMadnessPredictor with custom parameters."""
        from marchmadness import MarchMadnessPredictor

        predictor = MarchMadnessPredictor(
            season=2024, cache_dir="/tmp/cache", verbose=False
        )

        assert predictor.season == 2024
        assert predictor.cache_dir == "/tmp/cache"
        assert predictor.verbose is False

    @patch("marchmadness.predictor.DataScraper")
    def test_load_data(self, mock_scraper_class):
        """Test loading data."""
        from marchmadness import MarchMadnessPredictor

        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = {
            "teams": pd.DataFrame({"team": ["Duke"], "seed": [1]}),
            "games": pd.DataFrame(
                {"team1": ["Duke"], "team2": ["Auburn"], "winner": ["Duke"]}
            ),
            "players": pd.DataFrame({"player": ["Player1"], "team": ["Duke"]}),
        }
        mock_scraper_class.return_value = mock_scraper

        predictor = MarchMadnessPredictor(verbose=False)
        predictor.load_data(offline=True)

        assert predictor.teams_df is not None
        assert predictor.games_df is not None
        assert predictor.players_df is not None

    @patch("marchmadness.predictor.DataScraper")
    def test_load_data_verbose(self, mock_scraper_class):
        """Test loading data with verbose output."""
        from marchmadness import MarchMadnessPredictor

        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = {
            "teams": pd.DataFrame(
                {
                    "team": ["Duke"],
                    "seed": [1],
                    "region": ["East"],
                    "conference": ["ACC"],
                    "offensive_rating": [115.0],
                    "defensive_rating": [92.0],
                    "net_rating": [23.0],
                    "efg_pct": [0.55],
                    "to_pct": [15.0],
                    "sos": [10.0],
                    "win_pct": [0.9],
                    "kenpom_rank": [1],
                    "strength": [0.95],
                }
            ),
            "games": pd.DataFrame(
                {
                    "team1": ["Duke"],
                    "team2": ["Auburn"],
                    "winner": ["Duke"],
                    "loser": ["Auburn"],
                    "margin": [5],
                }
            ),
            "players": pd.DataFrame(
                {
                    "player": ["Player1"],
                    "team": ["Duke"],
                    "ppg": [15.0],
                    "rpg": [5.0],
                    "apg": [3.0],
                    "fg_pct": [0.5],
                    "three_pt_pct": [0.35],
                    "per": [20.0],
                    "minutes": [30.0],
                }
            ),
        }
        mock_scraper_class.return_value = mock_scraper

        predictor = MarchMadnessPredictor(verbose=True)
        predictor.load_data(offline=True)

    def test_load_data_raises_without_call(self):
        """Test that fit_mcmc raises if load_data not called."""
        from marchmadness import MarchMadnessPredictor

        predictor = MarchMadnessPredictor(verbose=False)

        with pytest.raises(RuntimeError, match="Call load_data"):
            predictor.fit_mcmc()

    @patch("marchmadness.predictor.MCMCModel")
    @patch("marchmadness.predictor.DataScraper")
    def test_fit_mcmc(self, mock_scraper_class, mock_mcmc_class):
        """Test fitting MCMC model."""
        from marchmadness import MarchMadnessPredictor

        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = {
            "teams": pd.DataFrame(
                {
                    "team": ["Duke", "Auburn"],
                    "seed": [1, 1],
                    "region": ["East", "West"],
                    "conference": ["ACC", "SEC"],
                    "offensive_rating": [115.0, 112.0],
                    "defensive_rating": [92.0, 94.0],
                    "net_rating": [23.0, 18.0],
                    "efg_pct": [0.55, 0.54],
                    "to_pct": [15.0, 15.5],
                    "sos": [10.0, 9.5],
                    "win_pct": [0.9, 0.88],
                    "kenpom_rank": [1, 2],
                    "strength": [0.95, 0.92],
                }
            ),
            "games": pd.DataFrame(
                {
                    "team1": ["Duke"],
                    "team2": ["Auburn"],
                    "winner": ["Duke"],
                    "loser": ["Auburn"],
                    "margin": [5],
                }
            ),
            "players": pd.DataFrame(
                {
                    "player": ["Player1"],
                    "team": ["Duke"],
                    "ppg": [15.0],
                    "rpg": [5.0],
                    "apg": [3.0],
                    "fg_pct": [0.5],
                    "three_pt_pct": [0.35],
                    "per": [20.0],
                    "minutes": [30.0],
                }
            ),
        }
        mock_scraper_class.return_value = mock_scraper

        mock_mcmc = MagicMock()
        mock_mcmc.fit.return_value = mock_mcmc
        mock_mcmc.alpha_samples = np.random.randn(100, 2)
        mock_mcmc.team_names = ["Duke", "Auburn"]
        mock_mcmc_class.return_value = mock_mcmc

        predictor = MarchMadnessPredictor(verbose=False)
        predictor.load_data(offline=True)
        predictor.fit_mcmc(n_warmup=10, n_samples=20)

        assert predictor._mcmc is not None

    @patch("marchmadness.predictor.BracketSimulator")
    @patch("marchmadness.predictor.MCMCModel")
    @patch("marchmadness.predictor.DataScraper")
    def test_run_tournament_simulation(
        self, mock_scraper_class, mock_mcmc_class, mock_sim_class
    ):
        """Test running tournament simulation."""
        from marchmadness import MarchMadnessPredictor

        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = {
            "teams": pd.DataFrame(
                {
                    "team": ["Duke", "Auburn"],
                    "seed": [1, 1],
                    "region": ["East", "West"],
                    "conference": ["ACC", "SEC"],
                    "offensive_rating": [115.0, 112.0],
                    "defensive_rating": [92.0, 94.0],
                    "net_rating": [23.0, 18.0],
                    "efg_pct": [0.55, 0.54],
                    "to_pct": [15.0, 15.5],
                    "sos": [10.0, 9.5],
                    "win_pct": [0.9, 0.88],
                    "kenpom_rank": [1, 2],
                    "strength": [0.95, 0.92],
                }
            ),
            "games": pd.DataFrame(
                {
                    "team1": ["Duke"],
                    "team2": ["Auburn"],
                    "winner": ["Duke"],
                    "loser": ["Auburn"],
                    "margin": [5],
                }
            ),
            "players": pd.DataFrame(
                {
                    "player": ["Player1"],
                    "team": ["Duke"],
                    "ppg": [15.0],
                    "rpg": [5.0],
                    "apg": [3.0],
                    "fg_pct": [0.5],
                    "three_pt_pct": [0.35],
                    "per": [20.0],
                    "minutes": [30.0],
                }
            ),
        }
        mock_scraper_class.return_value = mock_scraper

        mock_mcmc = MagicMock()
        mock_mcmc.fit.return_value = mock_mcmc
        mock_mcmc.alpha_samples = np.random.randn(100, 2)
        mock_mcmc.team_names = ["Duke", "Auburn"]
        mock_mcmc_class.return_value = mock_mcmc

        mock_sim = MagicMock()
        mock_sim.run_simulations.return_value = {
            "championship_probs": pd.Series({"Duke": 0.6, "Auburn": 0.4}),
            "final_four_probs": pd.Series({"Duke": 0.8, "Auburn": 0.6}),
            "elite_eight_probs": pd.Series({"Duke": 0.9, "Auburn": 0.7}),
            "sweet_16_probs": pd.Series({"Duke": 0.95, "Auburn": 0.85}),
            "round_32_probs": pd.Series({"Duke": 0.98, "Auburn": 0.95}),
            "win_counts": {},
            "raw_results": [],
            "n_simulations": 100,
        }
        mock_sim_class.return_value = mock_sim

        predictor = MarchMadnessPredictor(verbose=False)
        predictor.load_data(offline=True)
        predictor.fit_mcmc(n_warmup=10, n_samples=20)
        results = predictor.run_tournament_simulation(n_simulations=100)

        assert results is not None
        assert "championship_probs" in results

    def test_run_tournament_simulation_raises_without_fit(self):
        """Test that simulation raises if MCMC not fitted."""
        from marchmadness import MarchMadnessPredictor

        predictor = MarchMadnessPredictor(verbose=False)

        with pytest.raises(RuntimeError, match="Call fit_mcmc"):
            predictor.run_tournament_simulation()

    @patch("marchmadness.predictor.TournamentVisualizer")
    @patch("marchmadness.predictor.BracketSimulator")
    @patch("marchmadness.predictor.MCMCModel")
    @patch("marchmadness.predictor.DataScraper")
    def test_report_with_sim_results(
        self, mock_scraper_class, mock_mcmc_class, mock_sim_class, mock_viz_class
    ):
        """Test report with simulation results."""
        import sys

        sys.modules["matplotlib"] = MagicMock()
        sys.modules["matplotlib.pyplot"] = MagicMock()

        from marchmadness import MarchMadnessPredictor

        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = {
            "teams": pd.DataFrame(
                {
                    "team": ["Duke", "Auburn"],
                    "seed": [1, 1],
                    "region": ["East", "West"],
                    "conference": ["ACC", "SEC"],
                    "offensive_rating": [115.0, 112.0],
                    "defensive_rating": [92.0, 94.0],
                    "net_rating": [23.0, 18.0],
                    "efg_pct": [0.55, 0.54],
                    "to_pct": [15.0, 15.5],
                    "sos": [10.0, 9.5],
                    "win_pct": [0.9, 0.88],
                    "kenpom_rank": [1, 2],
                    "strength": [0.95, 0.92],
                }
            ),
            "games": pd.DataFrame(
                {
                    "team1": ["Duke"],
                    "team2": ["Auburn"],
                    "winner": ["Duke"],
                    "loser": ["Auburn"],
                    "margin": [5],
                }
            ),
            "players": pd.DataFrame(
                {
                    "player": ["Player1"],
                    "team": ["Duke"],
                    "ppg": [15.0],
                    "rpg": [5.0],
                    "apg": [3.0],
                    "fg_pct": [0.5],
                    "three_pt_pct": [0.35],
                    "per": [20.0],
                    "minutes": [30.0],
                }
            ),
        }
        mock_scraper_class.return_value = mock_scraper

        mock_mcmc = MagicMock()
        mock_mcmc.fit.return_value = mock_mcmc
        mock_mcmc.alpha_samples = np.random.randn(100, 2)
        mock_mcmc.team_names = ["Duke", "Auburn"]
        mock_mcmc.posterior_summary.return_value = pd.DataFrame(
            {
                "team": ["Duke", "Auburn"],
                "alpha_mean": [2.0, 1.5],
                "alpha_std": [0.3, 0.35],
                "alpha_p5": [1.5, 1.0],
                "alpha_p95": [2.5, 2.0],
                "strength_rank": [1, 2],
            }
        )
        mock_mcmc_class.return_value = mock_mcmc

        mock_sim = MagicMock()
        mock_sim.run_simulations.return_value = {
            "championship_probs": pd.Series({"Duke": 0.6, "Auburn": 0.4}),
            "final_four_probs": pd.Series({"Duke": 0.8, "Auburn": 0.6}),
            "elite_eight_probs": pd.Series({"Duke": 0.9, "Auburn": 0.7}),
            "sweet_16_probs": pd.Series({"Duke": 0.95, "Auburn": 0.85}),
            "round_32_probs": pd.Series({"Duke": 0.98, "Auburn": 0.95}),
            "win_counts": {},
            "raw_results": [],
            "n_simulations": 100,
        }
        mock_sim_class.return_value = mock_sim

        mock_viz = MagicMock()
        mock_viz.plot_all.return_value = MagicMock()
        mock_viz_class.return_value = mock_viz

        predictor = MarchMadnessPredictor(verbose=False)
        predictor.load_data(offline=True)
        predictor.fit_mcmc(n_warmup=10, n_samples=20)
        predictor.run_tournament_simulation(n_simulations=100)

        predictor.report()

    def test_report_raises_without_sim_results(self):
        """Test report raises if no simulation results."""
        from marchmadness import MarchMadnessPredictor

        predictor = MarchMadnessPredictor(verbose=False)

        with pytest.raises(RuntimeError, match="No simulation results"):
            predictor.report()

    @patch("marchmadness.predictor.MCMCModel")
    @patch("marchmadness.predictor.DataScraper")
    def test_win_probability(self, mock_scraper_class, mock_mcmc_class):
        """Test win probability calculation."""
        from marchmadness import MarchMadnessPredictor

        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = {
            "teams": pd.DataFrame(
                {
                    "team": ["Duke", "Auburn"],
                    "seed": [1, 1],
                    "region": ["East", "West"],
                    "conference": ["ACC", "SEC"],
                    "offensive_rating": [115.0, 112.0],
                    "defensive_rating": [92.0, 94.0],
                    "net_rating": [23.0, 18.0],
                    "efg_pct": [0.55, 0.54],
                    "to_pct": [15.0, 15.5],
                    "sos": [10.0, 9.5],
                    "win_pct": [0.9, 0.88],
                    "kenpom_rank": [1, 2],
                    "strength": [0.95, 0.92],
                }
            ),
            "games": pd.DataFrame(
                {
                    "team1": ["Duke"],
                    "team2": ["Auburn"],
                    "winner": ["Duke"],
                    "loser": ["Auburn"],
                    "margin": [5],
                }
            ),
            "players": pd.DataFrame(
                {
                    "player": ["Player1"],
                    "team": ["Duke"],
                    "ppg": [15.0],
                    "rpg": [5.0],
                    "apg": [3.0],
                    "fg_pct": [0.5],
                    "three_pt_pct": [0.35],
                    "per": [20.0],
                    "minutes": [30.0],
                }
            ),
        }
        mock_scraper_class.return_value = mock_scraper

        mock_mcmc = MagicMock()
        mock_mcmc.fit.return_value = mock_mcmc
        mock_mcmc.alpha_samples = np.random.randn(100, 2)
        mock_mcmc.team_names = ["Duke", "Auburn"]
        mock_mcmc.win_probability.return_value = 0.65
        mock_mcmc_class.return_value = mock_mcmc

        predictor = MarchMadnessPredictor(verbose=False)
        predictor.load_data(offline=True)
        predictor.fit_mcmc(n_warmup=10, n_samples=20)

        prob = predictor.win_probability("Duke", "Auburn")

        assert prob == 0.65

    def test_win_probability_raises_without_fit(self):
        """Test win probability raises if MCMC not fitted."""
        from marchmadness import MarchMadnessPredictor

        predictor = MarchMadnessPredictor(verbose=False)

        with pytest.raises(RuntimeError, match="Call fit_mcmc"):
            predictor.win_probability("Duke", "Auburn")

    @patch("marchmadness.predictor.MCMCModel")
    @patch("marchmadness.predictor.DataScraper")
    def test_posterior_summary(self, mock_scraper_class, mock_mcmc_class):
        """Test posterior summary."""
        from marchmadness import MarchMadnessPredictor

        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = {
            "teams": pd.DataFrame(
                {
                    "team": ["Duke", "Auburn"],
                    "seed": [1, 1],
                    "region": ["East", "West"],
                    "conference": ["ACC", "SEC"],
                    "offensive_rating": [115.0, 112.0],
                    "defensive_rating": [92.0, 94.0],
                    "net_rating": [23.0, 18.0],
                    "efg_pct": [0.55, 0.54],
                    "to_pct": [15.0, 15.5],
                    "sos": [10.0, 9.5],
                    "win_pct": [0.9, 0.88],
                    "kenpom_rank": [1, 2],
                    "strength": [0.95, 0.92],
                }
            ),
            "games": pd.DataFrame(
                {
                    "team1": ["Duke"],
                    "team2": ["Auburn"],
                    "winner": ["Duke"],
                    "loser": ["Auburn"],
                    "margin": [5],
                }
            ),
            "players": pd.DataFrame(
                {
                    "player": ["Player1"],
                    "team": ["Duke"],
                    "ppg": [15.0],
                    "rpg": [5.0],
                    "apg": [3.0],
                    "fg_pct": [0.5],
                    "three_pt_pct": [0.35],
                    "per": [20.0],
                    "minutes": [30.0],
                }
            ),
        }
        mock_scraper_class.return_value = mock_scraper

        expected_df = pd.DataFrame(
            {
                "team": ["Duke", "Auburn"],
                "alpha_mean": [2.0, 1.5],
            }
        )

        mock_mcmc = MagicMock()
        mock_mcmc.fit.return_value = mock_mcmc
        mock_mcmc.alpha_samples = np.random.randn(100, 2)
        mock_mcmc.team_names = ["Duke", "Auburn"]
        mock_mcmc.posterior_summary.return_value = expected_df
        mock_mcmc_class.return_value = mock_mcmc

        predictor = MarchMadnessPredictor(verbose=False)
        predictor.load_data(offline=True)
        predictor.fit_mcmc(n_warmup=10, n_samples=20)

        summary = predictor.posterior_summary()

        assert isinstance(summary, pd.DataFrame)

    def test_posterior_summary_raises_without_fit(self):
        """Test posterior summary raises if MCMC not fitted."""
        from marchmadness import MarchMadnessPredictor

        predictor = MarchMadnessPredictor(verbose=False)

        with pytest.raises(RuntimeError, match="Call fit_mcmc"):
            predictor.posterior_summary()
