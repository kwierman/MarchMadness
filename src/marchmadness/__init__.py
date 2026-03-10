"""
March Madness MCMC Predictor
============================
A Python package for predicting NCAA March Madness outcomes using
Markov Chain Monte Carlo simulation with real sports data.

Usage:
    from march_madness import MarchMadnessPredictor
    predictor = MarchMadnessPredictor()
    predictor.load_data()
    results = predictor.run_tournament_simulation(n_simulations=10000)
    predictor.plot_results(results)
"""

from .predictor import MarchMadnessPredictor
from .models.mcmc import MCMCModel
from .data.scraper import DataScraper
from .analysis.visualizer import TournamentVisualizer

__version__ = "1.0.0"
__all__ = ["MarchMadnessPredictor", "MCMCModel", "DataScraper", "TournamentVisualizer"]
