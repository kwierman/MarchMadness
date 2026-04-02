import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestTournamentVisualizer:
    """Tests for the TournamentVisualizer class."""

    def test_init(
        self,
        sample_teams_df,
        sample_posterior_df,
        sample_alpha_samples,
        sample_sim_results,
    ):
        """Test TournamentVisualizer initialization."""
        import sys

        mock_plt = MagicMock()
        sys.modules["matplotlib"] = MagicMock()
        sys.modules["matplotlib.pyplot"] = mock_plt
        sys.modules["matplotlib.gridspec"] = MagicMock()
        sys.modules["matplotlib.colors"] = MagicMock()

        from marchmadness.analysis.visualizer import TournamentVisualizer

        viz = TournamentVisualizer(
            teams_df=sample_teams_df,
            sim_results=sample_sim_results,
            posterior_df=sample_posterior_df,
            alpha_samples=sample_alpha_samples,
            team_names=list(sample_teams_df["team"].values),
        )

        assert viz.teams_df is not None
        assert viz.sim_results is not None
        assert viz.posterior_df is not None
        assert len(viz.team_names) == 6

    def test_init_creates_seed_map(
        self,
        sample_teams_df,
        sample_posterior_df,
        sample_alpha_samples,
        sample_sim_results,
    ):
        """Test that seed map is created during init."""
        import sys

        mock_plt = MagicMock()
        sys.modules["matplotlib"] = MagicMock()
        sys.modules["matplotlib.pyplot"] = mock_plt
        sys.modules["matplotlib.gridspec"] = MagicMock()
        sys.modules["matplotlib.colors"] = MagicMock()

        from marchmadness.analysis.visualizer import TournamentVisualizer

        viz = TournamentVisualizer(
            teams_df=sample_teams_df,
            sim_results=sample_sim_results,
            posterior_df=sample_posterior_df,
            alpha_samples=sample_alpha_samples,
            team_names=list(sample_teams_df["team"].values),
        )

        assert hasattr(viz, "seed_map")
        assert viz.seed_map["Duke"] == 1

    def test_palette_constant(self):
        """Test PALETTE constant exists."""
        import sys

        mock_plt = MagicMock()
        sys.modules["matplotlib"] = MagicMock()
        sys.modules["matplotlib.pyplot"] = mock_plt
        sys.modules["matplotlib.gridspec"] = MagicMock()
        sys.modules["matplotlib.colors"] = MagicMock()

        from marchmadness.analysis.visualizer import PALETTE

        assert isinstance(PALETTE, list)
        assert len(PALETTE) > 0

    def test_bracket_colors_constant(self):
        """Test bracket color constants."""
        import sys

        mock_plt = MagicMock()
        sys.modules["matplotlib"] = MagicMock()
        sys.modules["matplotlib.pyplot"] = mock_plt
        sys.modules["matplotlib.gridspec"] = MagicMock()
        sys.modules["matplotlib.colors"] = MagicMock()

        from marchmadness.analysis.visualizer import BRACKET_ORANGE, BRACKET_BLUE

        assert isinstance(BRACKET_ORANGE, str)
        assert isinstance(BRACKET_BLUE, str)
