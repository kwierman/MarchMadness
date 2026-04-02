import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch


class TestBracketSimulator:
    """Tests for the BracketSimulator class."""

    @pytest.fixture
    def full_teams_df(self):
        """Create a full 64-team bracket DataFrame."""
        regions = ["East", "West", "South", "Midwest"]
        teams = []
        for region in regions:
            for seed in range(1, 17):
                teams.append(
                    {
                        "team": f"Team_{region}_{seed}",
                        "seed": seed,
                        "region": region,
                        "conference": "TestConf",
                        "strength": 1.0 - (seed - 1) * 0.05,
                    }
                )
        return pd.DataFrame(teams)

    def test_init(self, full_teams_df, sample_alpha_samples):
        """Test BracketSimulator initialization."""
        from marchmadness.utils.bracket import BracketSimulator

        team_names = (
            [f"Team_East_{i}" for i in range(1, 17)]
            + [f"Team_West_{i}" for i in range(1, 17)]
            + [f"Team_South_{i}" for i in range(1, 17)]
            + [f"Team_Midwest_{i}" for i in range(1, 17)]
        )

        n_teams = len(team_names)
        alpha_samples = np.random.randn(100, n_teams)

        sim = BracketSimulator(
            teams_df=full_teams_df.head(64),
            alpha_samples=alpha_samples,
            team_names=team_names,
            seed=42,
        )

        assert sim.n_samples == 100

    def test_build_bracket(self, full_teams_df, sample_alpha_samples):
        """Test bracket building."""
        from marchmadness.utils.bracket import BracketSimulator

        team_names = list(full_teams_df["team"].values)
        alpha_samples = np.random.randn(100, 64)

        sim = BracketSimulator(
            teams_df=full_teams_df,
            alpha_samples=alpha_samples,
            team_names=team_names,
            seed=42,
        )

        assert hasattr(sim, "bracket")
        assert len(sim.bracket) == 4  # 4 regions
        for region in ["East", "West", "South", "Midwest"]:
            assert region in sim.bracket
            assert len(sim.bracket[region]) == 16

    def test_simulate_game_known_teams(self, sample_teams_df, sample_alpha_samples):
        """Test game simulation with known teams."""
        from marchmadness.utils.bracket import BracketSimulator

        team_names = list(sample_teams_df["team"].values)
        alpha_samples = sample_alpha_samples

        sim = BracketSimulator(
            teams_df=sample_teams_df,
            alpha_samples=alpha_samples,
            team_names=team_names,
            seed=42,
        )

        alpha = alpha_samples[0]
        winner = sim._simulate_game("Duke", "Auburn", alpha)

        assert winner in ["Duke", "Auburn"]

    def test_simulate_game_unknown_team(self, sample_teams_df, sample_alpha_samples):
        """Test game simulation with unknown team returns 50/50."""
        from marchmadness.utils.bracket import BracketSimulator

        team_names = list(sample_teams_df["team"].values)
        alpha_samples = sample_alpha_samples

        sim = BracketSimulator(
            teams_df=sample_teams_df,
            alpha_samples=alpha_samples,
            team_names=team_names,
            seed=42,
        )

        alpha = alpha_samples[0]
        winner = sim._simulate_game("Duke", "UnknownTeam", alpha)

        assert winner in ["Duke", "UnknownTeam"]

    def test_simulate_single_region(self, full_teams_df, sample_alpha_samples):
        """Test single region simulation."""
        from marchmadness.utils.bracket import BracketSimulator

        team_names = list(full_teams_df["team"].values)
        alpha_samples = np.random.randn(100, 64)

        sim = BracketSimulator(
            teams_df=full_teams_df,
            alpha_samples=alpha_samples,
            team_names=team_names,
            seed=42,
        )

        teams = [f"Team_East_{i}" for i in range(1, 17)]
        alpha = alpha_samples[0]
        result = sim._simulate_single_region(teams, alpha)

        assert "champion" in result
        assert "rounds" in result
        assert len(result["rounds"]) == 4

    def test_simulate_tournament(self, full_teams_df, sample_alpha_samples):
        """Test full tournament simulation."""
        from marchmadness.utils.bracket import BracketSimulator

        team_names = list(full_teams_df["team"].values)
        alpha_samples = np.random.randn(100, 64)

        sim = BracketSimulator(
            teams_df=full_teams_df,
            alpha_samples=alpha_samples,
            team_names=team_names,
            seed=42,
        )

        alpha = alpha_samples[0]
        result = sim._simulate_tournament(alpha)

        assert "regional" in result
        assert "final_four" in result
        assert "semifinal_1" in result
        assert "semifinal_2" in result
        assert "champion" in result

    def test_run_simulations(self, full_teams_df, sample_alpha_samples, capsys):
        """Test running tournament simulations."""
        from marchmadness.utils.bracket import BracketSimulator

        team_names = list(full_teams_df["team"].values)
        alpha_samples = np.random.randn(100, 64)

        sim = BracketSimulator(
            teams_df=full_teams_df,
            alpha_samples=alpha_samples,
            team_names=team_names,
            seed=42,
        )

        results = sim.run_simulations(n_simulations=100, verbose=False)

        assert "championship_probs" in results
        assert "final_four_probs" in results
        assert "elite_eight_probs" in results
        assert "sweet_16_probs" in results
        assert "round_32_probs" in results
        assert "win_counts" in results
        assert "raw_results" in results
        assert results["n_simulations"] == 100

        champ_probs = results["championship_probs"]
        assert champ_probs.sum() <= 1.0
        assert champ_probs.max() <= 1.0

    def test_run_simulations_verbose(self, full_teams_df, sample_alpha_samples, capsys):
        """Test running simulations with verbose output."""
        from marchmadness.utils.bracket import BracketSimulator

        team_names = list(full_teams_df["team"].values)
        alpha_samples = np.random.randn(100, 64)

        sim = BracketSimulator(
            teams_df=full_teams_df,
            alpha_samples=alpha_samples,
            team_names=team_names,
            seed=42,
        )

        results = sim.run_simulations(n_simulations=100, verbose=True)

        captured = capsys.readouterr()
        assert "BracketSimulator" in captured.out

    def test_run_simulations_probs_sum(self, full_teams_df, sample_alpha_samples):
        """Test that probabilities sum correctly."""
        from marchmadness.utils.bracket import BracketSimulator

        team_names = list(full_teams_df["team"].values)
        alpha_samples = np.random.randn(100, 64)

        sim = BracketSimulator(
            teams_df=full_teams_df,
            alpha_samples=alpha_samples,
            team_names=team_names,
            seed=42,
        )

        results = sim.run_simulations(n_simulations=1000, verbose=False)

        assert results["championship_probs"].sum() <= 1.0
        assert results["final_four_probs"].sum() <= 4.0

    def test_round_names(self):
        """Test ROUND_NAMES constant."""
        from marchmadness.utils.bracket import ROUND_NAMES

        assert 1 in ROUND_NAMES
        assert 6 in ROUND_NAMES
        assert ROUND_NAMES[1] == "Round of 64"
        assert ROUND_NAMES[6] == "Championship"
