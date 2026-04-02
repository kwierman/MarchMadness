import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os


class TestDataScraper:
    """Tests for the DataScraper class."""

    def test_init_default(self):
        """Test DataScraper initialization with defaults."""
        from marchmadness.data.scraper import DataScraper

        scraper = DataScraper()
        assert scraper.season == 2025
        assert scraper.cache_dir == ".mm_cache"
        assert scraper.use_cache is True

    def test_init_custom(self):
        """Test DataScraper initialization with custom parameters."""
        from marchmadness.data.scraper import DataScraper

        scraper = DataScraper(season=2024, cache_dir="/tmp/cache", use_cache=False)
        assert scraper.season == 2024
        assert scraper.cache_dir == "/tmp/cache"
        assert scraper.use_cache is False

    @patch("marchmadness.data.scraper.requests.get")
    def test_network_available_true(self, mock_get):
        """Test _network_available returns True when network is available."""
        from marchmadness.data.scraper import DataScraper

        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        scraper = DataScraper()
        assert scraper._network_available() is True

    @patch("marchmadness.data.scraper.requests.get")
    def test_network_available_false(self, mock_get):
        """Test _network_available returns False when network is unavailable."""
        from marchmadness.data.scraper import DataScraper

        mock_get.side_effect = Exception("Network error")

        scraper = DataScraper()
        assert scraper._network_available() is False

    def test_synthetic_data(self):
        """Test synthetic data generation."""
        from marchmadness.data.scraper import DataScraper

        scraper = DataScraper()
        result = scraper._synthetic_data()

        assert "teams" in result
        assert "games" in result
        assert "players" in result

        teams_df = result["teams"]
        assert len(teams_df) == 64
        assert "team" in teams_df.columns
        assert "seed" in teams_df.columns
        assert "region" in teams_df.columns

        assert teams_df["seed"].min() == 1
        assert teams_df["seed"].max() == 16

    @patch("marchmadness.data.scraper.requests.get")
    def test_fetch_all_offline(self, mock_get):
        """Test fetch_all with offline=True."""
        from marchmadness.data.scraper import DataScraper

        scraper = DataScraper()
        result = scraper.fetch_all(offline=True)

        assert "teams" in result
        assert "games" in result
        assert "players" in result
        mock_get.assert_not_called()

    @patch("marchmadness.data.scraper.DataScraper._network_available")
    @patch("marchmadness.data.scraper.requests.get")
    def test_fetch_all_network_unavailable(self, mock_get, mock_network):
        """Test fetch_all when network is unavailable."""
        from marchmadness.data.scraper import DataScraper

        mock_network.return_value = False

        scraper = DataScraper()
        result = scraper.fetch_all(offline=False)

        assert "teams" in result
        mock_get.assert_not_called()

    @patch("marchmadness.data.scraper.DataScraper._network_available")
    @patch("marchmadness.data.scraper.DataScraper._fetch_team_stats")
    @patch("marchmadness.data.scraper.DataScraper._fetch_game_results")
    @patch("marchmadness.data.scraper.DataScraper._fetch_player_stats")
    def test_fetch_all_success(self, mock_player, mock_game, mock_team, mock_network):
        """Test fetch_all when network is available and fetch succeeds."""
        from marchmadness.data.scraper import DataScraper

        mock_network.return_value = True
        mock_team.return_value = pd.DataFrame({"team": ["Duke"]})
        mock_game.return_value = pd.DataFrame(
            {"team1": ["Duke"], "team2": ["Auburn"], "winner": ["Duke"]}
        )
        mock_player.return_value = pd.DataFrame(
            {"player": ["Player1"], "team": ["Duke"]}
        )

        scraper = DataScraper()
        result = scraper.fetch_all(offline=False)

        assert "teams" in result
        assert "games" in result
        assert "players" in result

    @patch("marchmadness.data.scraper.DataScraper._network_available")
    @patch("marchmadness.data.scraper.DataScraper._fetch_team_stats")
    def test_fetch_all_exception_fallback(self, mock_team, mock_network):
        """Test fetch_all falls back to synthetic data on exception."""
        from marchmadness.data.scraper import DataScraper

        mock_network.return_value = True
        mock_team.side_effect = Exception("Network error")

        scraper = DataScraper()
        result = scraper.fetch_all(offline=False)

        assert "teams" in result

    def test_fetch_team_stats_from_cache(self):
        """Test fetch_team_stats loads from cache."""
        from marchmadness.data.scraper import DataScraper

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            with patch("pandas.read_csv") as mock_read:
                mock_read.return_value = pd.DataFrame({"team": ["Duke"]})

                scraper = DataScraper(cache_dir="/tmp/cache")
                df = scraper._fetch_team_stats()

                assert isinstance(df, pd.DataFrame)

    def test_fetch_game_results_from_cache(self):
        """Test fetch_game_results loads from cache."""
        from marchmadness.data.scraper import DataScraper

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            with patch("pandas.read_csv") as mock_read:
                mock_read.return_value = pd.DataFrame(
                    {"team1": ["Duke"], "team2": ["Auburn"]}
                )

                scraper = DataScraper(cache_dir="/tmp/cache")
                df = scraper._fetch_game_results()

                assert isinstance(df, pd.DataFrame)

    def test_fetch_player_stats_from_cache(self):
        """Test fetch_player_stats loads from cache."""
        from marchmadness.data.scraper import DataScraper

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            with patch("pandas.read_csv") as mock_read:
                mock_read.return_value = pd.DataFrame(
                    {"player": ["Player1"], "team": ["Duke"]}
                )

                scraper = DataScraper(cache_dir="/tmp/cache")
                df = scraper._fetch_player_stats()

                assert isinstance(df, pd.DataFrame)
