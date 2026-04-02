import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch


class TestMCMCModel:
    """Tests for the MCMCModel class."""

    def test_init(self, sample_teams_df, sample_games_df):
        """Test MCMCModel initialization."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(
            teams_df=sample_teams_df,
            games_df=sample_games_df,
            n_warmup=100,
            n_samples=200,
            step_size=0.1,
            seed=42,
        )

        assert model.n_warmup == 100
        assert model.n_samples == 200
        assert model.step_size == 0.1
        assert len(model.team_names) == 6
        assert model.n_teams == 6

    def test_init_default_params(self, sample_teams_df, sample_games_df):
        """Test MCMCModel with default parameters."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(teams_df=sample_teams_df, games_df=sample_games_df)

        assert model.n_warmup == 2000
        assert model.n_samples == 4000
        assert model.step_size == 0.08

    def test_build_informative_prior(self, sample_teams_df, sample_games_df):
        """Test building informative prior from team stats."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(teams_df=sample_teams_df, games_df=sample_games_df)
        prior = model._build_informative_prior()

        assert isinstance(prior, np.ndarray)
        assert len(prior) == 6

    def test_build_informative_prior_minimal_columns(self):
        """Test building prior with minimal columns."""
        from marchmadness.models.mcmc import MCMCModel

        teams_df = pd.DataFrame(
            {
                "team": ["TeamA", "TeamB", "TeamC"],
                "seed": [1, 2, 3],
            }
        )
        games_df = pd.DataFrame(
            {
                "team1": ["TeamA"],
                "team2": ["TeamB"],
                "winner": ["TeamA"],
            }
        )

        model = MCMCModel(teams_df=teams_df, games_df=games_df)
        prior = model._build_informative_prior()

        assert isinstance(prior, np.ndarray)
        assert len(prior) == 3
        assert np.allclose(prior, 0)

    def test_log_likelihood(self, sample_teams_df, sample_games_df):
        """Test log likelihood calculation."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(teams_df=sample_teams_df, games_df=sample_games_df)
        alpha = np.zeros(6)
        ll = model._log_likelihood(alpha)

        assert isinstance(ll, float)
        assert np.isfinite(ll)

    def test_log_likelihood_unknown_team(self, sample_teams_df, sample_games_df):
        """Test log likelihood with unknown team."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(teams_df=sample_teams_df, games_df=sample_games_df)
        games_df_with_unknown = pd.DataFrame(
            {
                "team1": ["Duke", "UnknownTeam"],
                "team2": ["Auburn", "Houston"],
                "winner": ["Duke", "Duke"],
            }
        )
        model.games_df = games_df_with_unknown
        alpha = np.zeros(6)
        ll = model._log_likelihood(alpha)

        assert isinstance(ll, float)

    def test_log_prior(self, sample_teams_df, sample_games_df):
        """Test log prior calculation."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(teams_df=sample_teams_df, games_df=sample_games_df)
        alpha = np.zeros(6)
        prior_mean = np.zeros(6)
        lp = model._log_prior(alpha, prior_mean)

        assert isinstance(lp, float)

    def test_log_posterior(self, sample_teams_df, sample_games_df):
        """Test log posterior calculation."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(teams_df=sample_teams_df, games_df=sample_games_df)
        alpha = np.zeros(6)
        prior_mean = np.zeros(6)
        lp = model._log_posterior(alpha, prior_mean)

        assert isinstance(lp, float)
        assert np.isfinite(lp)

    def test_fit(self, sample_teams_df, sample_games_df, capsys):
        """Test MCMC model fitting."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(
            teams_df=sample_teams_df,
            games_df=sample_games_df,
            n_warmup=50,
            n_samples=100,
            seed=42,
        )
        model.fit(verbose=False)

        assert model.alpha_samples is not None
        assert model.alpha_samples.shape == (100, 6)
        assert 0 < model.acceptance_rate <= 1

    def test_fit_verbose(self, sample_teams_df, sample_games_df, capsys):
        """Test MCMC model fitting with verbose output."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(
            teams_df=sample_teams_df,
            games_df=sample_games_df,
            n_warmup=50,
            n_samples=100,
            seed=42,
        )
        model.fit(verbose=True)

        captured = capsys.readouterr()
        assert "MCMC" in captured.out

    def test_win_probability(self, sample_teams_df, sample_games_df):
        """Test win probability calculation."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(
            teams_df=sample_teams_df,
            games_df=sample_games_df,
            n_warmup=50,
            n_samples=100,
            seed=42,
        )
        model.fit(verbose=False)

        prob = model.win_probability("Duke", "Auburn")

        assert isinstance(prob, float)
        assert 0 <= prob <= 1

    def test_win_probability_before_fit(self, sample_teams_df, sample_games_df):
        """Test win probability raises error before fit."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(teams_df=sample_teams_df, games_df=sample_games_df)

        with pytest.raises(RuntimeError, match="Call .fit"):
            model.win_probability("Duke", "Auburn")

    def test_posterior_summary(self, sample_teams_df, sample_games_df):
        """Test posterior summary DataFrame."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(
            teams_df=sample_teams_df,
            games_df=sample_games_df,
            n_warmup=50,
            n_samples=100,
            seed=42,
        )
        model.fit(verbose=False)

        summary = model.posterior_summary()

        assert isinstance(summary, pd.DataFrame)
        assert "team" in summary.columns
        assert "alpha_mean" in summary.columns
        assert "alpha_std" in summary.columns
        assert "alpha_p5" in summary.columns
        assert "alpha_p95" in summary.columns
        assert "strength_rank" in summary.columns

    def test_posterior_summary_before_fit(self, sample_teams_df, sample_games_df):
        """Test posterior summary raises error before fit."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(teams_df=sample_teams_df, games_df=sample_games_df)

        with pytest.raises(RuntimeError, match="Call .fit"):
            model.posterior_summary()

    def test_posterior_summary_sorted(self, sample_teams_df, sample_games_df):
        """Test posterior summary is sorted by alpha_mean."""
        from marchmadness.models.mcmc import MCMCModel

        model = MCMCModel(
            teams_df=sample_teams_df,
            games_df=sample_games_df,
            n_warmup=50,
            n_samples=100,
            seed=42,
        )
        model.fit(verbose=False)

        summary = model.posterior_summary()

        assert summary["alpha_mean"].is_monotonic_decreasing
