import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def sample_teams_df():
    """Create a sample teams DataFrame for testing."""
    return pd.DataFrame(
        {
            "team": ["Duke", "Auburn", "Houston", "Florida", "Michigan St", "Alabama"],
            "seed": [1, 1, 1, 1, 2, 2],
            "region": ["East", "West", "South", "Midwest", "East", "West"],
            "conference": ["ACC", "SEC", "Big 12", "SEC", "Big Ten", "SEC"],
            "strength": [0.95, 0.92, 0.90, 0.88, 0.85, 0.82],
            "offensive_rating": [115.0, 112.0, 110.0, 108.0, 106.0, 104.0],
            "defensive_rating": [92.0, 94.0, 95.0, 96.0, 97.0, 98.0],
            "net_rating": [23.0, 18.0, 15.0, 12.0, 9.0, 6.0],
            "efg_pct": [0.55, 0.54, 0.53, 0.52, 0.51, 0.50],
            "to_pct": [15.0, 15.5, 16.0, 16.5, 17.0, 17.5],
            "sos": [10.0, 9.5, 9.0, 8.5, 8.0, 7.5],
            "win_pct": [0.90, 0.88, 0.85, 0.82, 0.80, 0.75],
            "kenpom_rank": [1, 2, 3, 4, 5, 6],
        }
    )


@pytest.fixture
def sample_games_df():
    """Create a sample games DataFrame for testing."""
    return pd.DataFrame(
        {
            "team1": ["Duke", "Auburn", "Houston", "Florida", "Michigan St"],
            "team2": ["Auburn", "Houston", "Florida", "Michigan St", "Alabama"],
            "winner": ["Duke", "Auburn", "Houston", "Florida", "Michigan St"],
            "loser": ["Auburn", "Houston", "Florida", "Michigan St", "Alabama"],
            "margin": [5, 3, 7, 2, 4],
        }
    )


@pytest.fixture
def sample_alpha_samples():
    """Create sample MCMC alpha samples for testing."""
    np.random.seed(42)
    n_samples = 100
    n_teams = 6
    return np.random.randn(n_samples, n_teams)


@pytest.fixture
def sample_posterior_df():
    """Create sample posterior summary DataFrame."""
    return pd.DataFrame(
        {
            "team": ["Duke", "Auburn", "Houston", "Florida", "Michigan St", "Alabama"],
            "alpha_mean": [2.0, 1.8, 1.5, 1.2, 0.9, 0.5],
            "alpha_std": [0.3, 0.35, 0.4, 0.42, 0.45, 0.5],
            "alpha_p5": [1.5, 1.3, 1.0, 0.5, 0.2, -0.3],
            "alpha_p95": [2.5, 2.3, 2.0, 1.8, 1.5, 1.2],
            "strength_rank": [1, 2, 3, 4, 5, 6],
        }
    )


@pytest.fixture
def sample_sim_results():
    """Create sample simulation results for testing."""
    teams = ["Duke", "Auburn", "Houston", "Florida", "Michigan St", "Alabama"]
    return {
        "championship_probs": pd.Series(
            {
                "Duke": 0.25,
                "Auburn": 0.20,
                "Houston": 0.18,
                "Florida": 0.15,
                "Michigan St": 0.12,
                "Alabama": 0.10,
            }
        ),
        "final_four_probs": pd.Series(
            {
                "Duke": 0.50,
                "Auburn": 0.45,
                "Houston": 0.40,
                "Florida": 0.35,
                "Michigan St": 0.30,
                "Alabama": 0.25,
            }
        ),
        "elite_eight_probs": pd.Series(
            {
                "Duke": 0.70,
                "Auburn": 0.65,
                "Houston": 0.60,
                "Florida": 0.55,
                "Michigan St": 0.50,
                "Alabama": 0.45,
            }
        ),
        "sweet_16_probs": pd.Series(
            {
                "Duke": 0.85,
                "Auburn": 0.80,
                "Houston": 0.75,
                "Florida": 0.70,
                "Michigan St": 0.65,
                "Alabama": 0.60,
            }
        ),
        "round_32_probs": pd.Series(
            {
                "Duke": 0.95,
                "Auburn": 0.92,
                "Houston": 0.90,
                "Florida": 0.88,
                "Michigan St": 0.85,
                "Alabama": 0.82,
            }
        ),
        "win_counts": {},
        "raw_results": [],
        "n_simulations": 1000,
    }
