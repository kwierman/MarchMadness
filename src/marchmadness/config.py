"""
config.py
---------
Configuration management for March Madness predictor.

Loads settings from YAML files with support for:
- Data sources (URLs, cache directories)
- Model storage paths
- Logging configuration
- MCMC parameters

Usage:
    from marchmadness.config import Config

    # Load from default location
    config = Config.load()

    # Load from custom file
    config = Config.load(path="custom_config.yaml")

    # Access configuration
    config.data.cache_dir
    config.logging.level
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""

    sports_reference_url: str = "https://www.sports-reference.com/cbb"
    espn_api_url: str = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
    cache_dir: str = ".mm_cache"
    use_cache: bool = True
    offline_fallback: bool = True
    timeout_seconds: int = 15


@dataclass
class ModelConfig:
    """Configuration for model parameters."""

    mcmc_warmup: int = 2000
    mcmc_samples: int = 4000
    mcmc_step_size: float = 0.08
    mcmc_seed: int = 0
    n_simulations: int = 10000
    output_dir: str = "output"
    model_save_path: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    console: bool = True


@dataclass
class VisualizerConfig:
    """Configuration for visualization."""

    dpi: int = 150
    figure_size: tuple = (22, 26)
    theme: str = "default"
    output_format: str = "png"


@dataclass
class Config:
    """
    Main configuration class for March Madness predictor.

    Attributes:
        data: Data source configuration
        model: Model parameters
        logging: Logging configuration
        visualizer: Visualization settings
        season: Default season year
    """

    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    season: int = 2025

    _config_path: Optional[Path] = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            path: Path to config file. If None, searches for:
                  1. ./march_madness.yaml
                  2. ./march_madness.yml
                  3. ~/.march_madness.yaml
                  4. /etc/march_madness.yaml

        Returns:
            Config instance with loaded settings

        Raises:
            FileNotFoundError: If no config file found and no defaults
        """
        if path:
            config_path = Path(path)
            if config_path.exists():
                return cls._from_file(config_path)
            else:
                raise FileNotFoundError(f"Config file not found: {path}")

        search_paths = [
            Path("./march_madness.yaml"),
            Path("./march_madness.yml"),
            Path.home() / ".march_madness.yaml",
            Path("/etc/march_madness.yaml"),
        ]

        for config_path in search_paths:
            if config_path.exists():
                return cls._from_file(config_path)

        return cls()

    @classmethod
    def _from_file(cls, path: Path) -> "Config":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        if "data" in data:
            config.data = DataSourceConfig(**data["data"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        if "visualizer" in data:
            config.visualizer = VisualizerConfig(**data["visualizer"])
        if "season" in data:
            config.season = data["season"]

        config._config_path = path
        return config

    def save(self, path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path where to save the config
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "season": self.season,
            "data": {
                "sports_reference_url": self.data.sports_reference_url,
                "espn_api_url": self.data.espn_api_url,
                "cache_dir": self.data.cache_dir,
                "use_cache": self.data.use_cache,
                "offline_fallback": self.data.offline_fallback,
                "timeout_seconds": self.data.timeout_seconds,
            },
            "model": {
                "mcmc_warmup": self.model.mcmc_warmup,
                "mcmc_samples": self.model.mcmc_samples,
                "mcmc_step_size": self.model.mcmc_step_size,
                "mcmc_seed": self.model.mcmc_seed,
                "n_simulations": self.model.n_simulations,
                "output_dir": self.model.output_dir,
                "model_save_path": self.model.model_save_path,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file,
                "console": self.logging.console,
            },
            "visualizer": {
                "dpi": self.visualizer.dpi,
                "figure_size": list(self.visualizer.figure_size),
                "theme": self.visualizer.theme,
                "output_format": self.visualizer.output_format,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_env(cls) -> "Config":
        """
        Create configuration from environment variables.

        Environment variables:
            MM_CACHE_DIR: Cache directory
            MM_SEASON: Default season
            MM_LOG_LEVEL: Logging level
            MM_OFFLINE: Use offline mode
            MM_N_SIMS: Number of simulations
        """
        config = cls()

        if cache_dir := os.environ.get("MM_CACHE_DIR"):
            config.data.cache_dir = cache_dir

        if season := os.environ.get("MM_SEASON"):
            config.season = int(season)

        if log_level := os.environ.get("MM_LOG_LEVEL"):
            config.logging.level = log_level

        if offline := os.environ.get("MM_OFFLINE"):
            config.data.offline_fallback = offline.lower() == "true"

        if n_sims := os.environ.get("MM_N_SIMS"):
            config.model.n_simulations = int(n_sims)

        return config


def get_config(path: Optional[str] = None) -> Config:
    """
    Convenience function to get configuration.

    Loads from file if path provided, otherwise uses defaults.
    """
    if path:
        return Config.load(path)
    return Config.load() if path is None else Config.load(path)
