import pytest
import os
import tempfile
from pathlib import Path


class TestConfig:
    """Tests for the Config class."""

    def test_default_config(self):
        """Test creating default config."""
        from marchmadness.config import Config

        config = Config()

        assert config.season == 2025
        assert config.data.cache_dir == ".mm_cache"
        assert config.data.use_cache is True
        assert config.model.mcmc_warmup == 2000
        assert config.model.mcmc_samples == 4000
        assert config.logging.level == "INFO"
        assert config.visualizer.dpi == 150

    def test_load_from_file(self):
        """Test loading config from YAML file."""
        from marchmadness.config import Config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
season: 2024
data:
  cache_dir: "/tmp/test_cache"
  use_cache: false
model:
  mcmc_warmup: 1000
  mcmc_samples: 2000
logging:
  level: "DEBUG"
""")
            f.flush()

            config = Config.load(f.name)

            os.unlink(f.name)

        assert config.season == 2024
        assert config.data.cache_dir == "/tmp/test_cache"
        assert config.data.use_cache is False
        assert config.model.mcmc_warmup == 1000
        assert config.logging.level == "DEBUG"

    def test_save_to_file(self):
        """Test saving config to YAML file."""
        from marchmadness.config import Config

        config = Config()
        config.season = 2026
        config.data.cache_dir = "/tmp/my_cache"
        config.model.n_simulations = 5000

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            config.save(temp_path)

            loaded = Config.load(temp_path)

            assert loaded.season == 2026
            assert loaded.data.cache_dir == "/tmp/my_cache"
            assert loaded.model.n_simulations == 5000
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Test loading non-existent file raises error."""
        from marchmadness.config import Config

        with pytest.raises(FileNotFoundError):
            Config.load("/nonexistent/config.yaml")

    def test_data_source_config(self):
        """Test DataSourceConfig defaults."""
        from marchmadness.config import DataSourceConfig

        config = DataSourceConfig()

        assert config.sports_reference_url == "https://www.sports-reference.com/cbb"
        assert (
            config.espn_api_url
            == "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
        )
        assert config.cache_dir == ".mm_cache"
        assert config.use_cache is True
        assert config.offline_fallback is True
        assert config.timeout_seconds == 15

    def test_model_config(self):
        """Test ModelConfig defaults."""
        from marchmadness.config import ModelConfig

        config = ModelConfig()

        assert config.mcmc_warmup == 2000
        assert config.mcmc_samples == 4000
        assert config.mcmc_step_size == 0.08
        assert config.mcmc_seed == 0
        assert config.n_simulations == 10000
        assert config.output_dir == "output"
        assert config.model_save_path is None

    def test_logging_config(self):
        """Test LoggingConfig defaults."""
        from marchmadness.config import LoggingConfig

        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.file is None
        assert config.console is True

    def test_visualizer_config(self):
        """Test VisualizerConfig defaults."""
        from marchmadness.config import VisualizerConfig

        config = VisualizerConfig()

        assert config.dpi == 150
        assert config.figure_size == (22, 26)
        assert config.theme == "default"
        assert config.output_format == "png"

    def test_get_config_function(self):
        """Test get_config convenience function."""
        from marchmadness.config import get_config

        config = get_config()

        assert isinstance(config.season, int)
        assert config.data.cache_dir is not None

    def test_config_from_env_missing(self, monkeypatch):
        """Test Config.from_env with no env vars."""
        monkeypatch.delenv("MM_CACHE_DIR", raising=False)
        monkeypatch.delenv("MM_SEASON", raising=False)
        monkeypatch.delenv("MM_LOG_LEVEL", raising=False)

        from marchmadness.config import Config

        config = Config.from_env()

        assert config.data.cache_dir == ".mm_cache"
        assert config.season == 2025

    def test_config_from_env_with_values(self, monkeypatch):
        """Test Config.from_env with env vars."""
        monkeypatch.setenv("MM_CACHE_DIR", "/custom/cache")
        monkeypatch.setenv("MM_SEASON", "2024")
        monkeypatch.setenv("MM_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("MM_N_SIMS", "5000")

        from marchmadness.config import Config

        config = Config.from_env()

        assert config.data.cache_dir == "/custom/cache"
        assert config.season == 2024
        assert config.logging.level == "DEBUG"
        assert config.model.n_simulations == 5000
