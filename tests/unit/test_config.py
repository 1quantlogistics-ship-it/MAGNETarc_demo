"""
Unit tests for ARC configuration management.

Tests configuration loading, environment detection, path resolution,
and validation logic.
"""

import os
import pytest
import tempfile
from pathlib import Path

from config import (
    ARCSettings,
    get_settings,
    reset_settings_cache,
    get_dev_settings,
    get_test_settings,
    get_prod_settings,
    detect_environment,
    validate_configuration
)


# ============================================================================
# ARCSettings Tests
# ============================================================================

@pytest.mark.unit
class TestARCSettings:
    """Test ARCSettings configuration class."""

    def test_default_settings(self):
        """Test creating settings with defaults."""
        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")

        assert settings.environment == "dev"
        assert settings.arc_version == "1.1.0"
        assert settings.mode == "SEMI"
        assert settings.home == Path("/workspace/arc")

    def test_test_environment_uses_temp_dir(self):
        """Test that test environment uses temporary directory."""
        settings = ARCSettings(
            environment="test",
            llm_endpoint="http://localhost:8000/v1"
        )

        assert settings.environment == "test"
        assert "/tmp/" in str(settings.home) or "Temp" in str(settings.home)
        assert settings.llm_timeout == 10  # Shorter for tests

    def test_prod_environment_safety(self):
        """Test that prod environment enforces safety settings."""
        settings = ARCSettings(
            environment="prod",
            llm_endpoint="http://localhost:8000/v1"
        )

        assert settings.environment == "prod"
        assert settings.require_approval_for_train is True
        assert settings.require_approval_for_commands is True
        assert settings.mode == "SEMI"
        assert settings.api_debug is False

    def test_custom_paths(self, tmp_path):
        """Test settings with custom paths."""
        custom_home = tmp_path / "custom_arc"

        settings = ARCSettings(
            home=custom_home,
            llm_endpoint="http://localhost:8000/v1"
        )

        assert settings.home == custom_home
        assert settings.memory_dir == custom_home / "memory"
        assert settings.experiments_dir == custom_home / "experiments"

    def test_explicit_subdirectories(self, tmp_path):
        """Test explicitly setting subdirectories."""
        settings = ARCSettings(
            home=tmp_path / "arc",
            memory_dir=tmp_path / "custom_memory",
            llm_endpoint="http://localhost:8000/v1"
        )

        assert settings.memory_dir == tmp_path / "custom_memory"

    def test_environment_variable_override(self, monkeypatch, tmp_path):
        """Test overriding settings via environment variables."""
        monkeypatch.setenv("ARC_HOME", str(tmp_path / "env_arc"))
        monkeypatch.setenv("ARC_MODE", "AUTO")
        monkeypatch.setenv("ARC_LLM_TIMEOUT", "60")

        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")

        assert settings.home == tmp_path / "env_arc"
        assert settings.mode == "AUTO"
        assert settings.llm_timeout == 60

    def test_ensure_directories_creates_structure(self, tmp_path):
        """Test that ensure_directories creates all directories."""
        custom_home = tmp_path / "test_arc"

        settings = ARCSettings(
            home=custom_home,
            llm_endpoint="http://localhost:8000/v1"
        )

        # Directories shouldn't exist yet
        assert not settings.memory_dir.exists()

        # Create them
        settings.ensure_directories()

        # Now they should exist
        assert settings.home.exists()
        assert settings.memory_dir.exists()
        assert settings.experiments_dir.exists()
        assert settings.logs_dir.exists()
        assert settings.checkpoints_dir.exists()
        assert settings.snapshots_dir.exists()

    def test_get_memory_file_path(self):
        """Test memory file path helper."""
        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")
        path = settings.get_memory_file_path("directive.json")

        assert path.name == "directive.json"
        assert path.parent == settings.memory_dir

    def test_get_experiment_path(self):
        """Test experiment path helper."""
        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")
        path = settings.get_experiment_path("exp_1_1")

        assert path.name == "exp_1_1"
        assert path.parent == settings.experiments_dir

    def test_get_log_file_path(self):
        """Test log file path helper."""
        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")
        path = settings.get_log_file_path("arc")

        assert "arc_" in path.name
        assert ".log" in path.name
        assert path.parent == settings.logs_dir

    def test_get_snapshot_path(self):
        """Test snapshot path helper."""
        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")
        path = settings.get_snapshot_path("snapshot_001")

        assert path.name == "snapshot_001"
        assert path.parent == settings.snapshots_dir

    def test_to_dict(self):
        """Test converting settings to dictionary."""
        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")
        config_dict = settings.to_dict()

        assert isinstance(config_dict, dict)
        assert "environment" in config_dict
        assert "arc_version" in config_dict
        assert "mode" in config_dict


# ============================================================================
# Settings Factory Tests
# ============================================================================

@pytest.mark.unit
class TestSettingsFactory:
    """Test settings factory functions."""

    def test_get_settings_default(self):
        """Test get_settings with default environment."""
        reset_settings_cache()
        settings = get_settings()

        assert settings.environment == "dev"

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        reset_settings_cache()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2  # Same instance

    def test_get_dev_settings(self):
        """Test get_dev_settings helper."""
        reset_settings_cache()
        settings = get_dev_settings()

        assert settings.environment == "dev"

    def test_get_test_settings(self):
        """Test get_test_settings helper."""
        reset_settings_cache()
        settings = get_test_settings()

        assert settings.environment == "test"
        assert "/tmp/" in str(settings.home) or "Temp" in str(settings.home)

    def test_get_prod_settings(self):
        """Test get_prod_settings helper."""
        reset_settings_cache()
        settings = get_prod_settings()

        assert settings.environment == "prod"
        assert settings.require_approval_for_train is True

    def test_reset_settings_cache_clears(self):
        """Test that reset_settings_cache clears the cache."""
        reset_settings_cache()

        settings1 = get_settings()
        reset_settings_cache()
        settings2 = get_settings()

        # Different instances after reset
        assert settings1 is not settings2


# ============================================================================
# Environment Detection Tests
# ============================================================================

@pytest.mark.unit
class TestEnvironmentDetection:
    """Test environment detection logic."""

    def test_detect_environment_from_var(self, monkeypatch):
        """Test detecting environment from ARC_ENVIRONMENT variable."""
        monkeypatch.setenv("ARC_ENVIRONMENT", "prod")
        env = detect_environment()

        assert env == "prod"

    def test_detect_environment_pytest(self, monkeypatch):
        """Test detecting test environment during pytest."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_config.py::test_name")
        monkeypatch.delenv("ARC_ENVIRONMENT", raising=False)

        env = detect_environment()

        assert env == "test"

    def test_detect_environment_default(self, monkeypatch):
        """Test default environment detection."""
        monkeypatch.delenv("ARC_ENVIRONMENT", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("RUNPOD_POD_ID", raising=False)

        env = detect_environment()

        assert env == "dev"


# ============================================================================
# Configuration Validation Tests
# ============================================================================

@pytest.mark.unit
class TestConfigurationValidation:
    """Test configuration validation logic."""

    def test_validate_valid_configuration(self, tmp_path):
        """Test validating a valid configuration."""
        settings = ARCSettings(
            home=tmp_path / "arc",
            llm_endpoint="http://localhost:8000/v1",
            min_learning_rate=0.0001,
            max_learning_rate=0.1
        )

        # Create parent directory so validation passes
        settings.home.parent.mkdir(parents=True, exist_ok=True)

        is_valid, issues = validate_configuration(settings)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_invalid_llm_endpoint(self, tmp_path):
        """Test validation fails for invalid LLM endpoint."""
        settings = ARCSettings(
            home=tmp_path / "arc",
            llm_endpoint="not-a-url",  # Invalid
            min_learning_rate=0.0001,
            max_learning_rate=0.1
        )

        is_valid, issues = validate_configuration(settings)

        assert is_valid is False
        assert any("Invalid LLM endpoint" in issue for issue in issues)

    def test_validate_invalid_learning_rate_range(self, tmp_path):
        """Test validation fails for invalid learning rate range."""
        settings = ARCSettings(
            home=tmp_path / "arc",
            llm_endpoint="http://localhost:8000/v1",
            min_learning_rate=0.1,
            max_learning_rate=0.01  # min > max, invalid
        )

        is_valid, issues = validate_configuration(settings)

        assert is_valid is False
        assert any("min_learning_rate must be less than max_learning_rate" in issue for issue in issues)

    def test_validate_invalid_batch_size_range(self, tmp_path):
        """Test validation fails for invalid batch size range."""
        settings = ARCSettings(
            home=tmp_path / "arc",
            llm_endpoint="http://localhost:8000/v1",
            min_batch_size=64,
            max_batch_size=32  # min > max, invalid
        )

        is_valid, issues = validate_configuration(settings)

        assert is_valid is False
        assert any("min_batch_size must be less than or equal to max_batch_size" in issue for issue in issues)

    def test_validate_full_mode_in_prod_warning(self, tmp_path):
        """Test validation warns about FULL mode in production."""
        settings = ARCSettings(
            home=tmp_path / "arc",
            llm_endpoint="http://localhost:8000/v1",
            environment="prod",
            mode="FULL"  # Dangerous in prod
        )

        is_valid, issues = validate_configuration(settings)

        assert is_valid is False
        assert any("FULL mode is not recommended for production" in issue for issue in issues)


# ============================================================================
# Path Conversion Tests
# ============================================================================

@pytest.mark.unit
class TestPathConversion:
    """Test path string to Path object conversion."""

    def test_string_path_converted(self):
        """Test that string paths are converted to Path objects."""
        settings = ARCSettings(
            home="/workspace/arc",
            llm_endpoint="http://localhost:8000/v1"
        )

        assert isinstance(settings.home, Path)
        assert settings.home == Path("/workspace/arc")

    def test_path_object_accepted(self, tmp_path):
        """Test that Path objects are accepted directly."""
        settings = ARCSettings(
            home=tmp_path / "arc",
            llm_endpoint="http://localhost:8000/v1"
        )

        assert isinstance(settings.home, Path)
        assert settings.home == tmp_path / "arc"

    def test_tilde_expansion(self):
        """Test that ~ is expanded in paths."""
        settings = ARCSettings(
            home="~/arc",
            llm_endpoint="http://localhost:8000/v1"
        )

        assert "~" not in str(settings.home)
        assert settings.home.is_absolute()


# ============================================================================
# Constraint Defaults Tests
# ============================================================================

@pytest.mark.unit
class TestConstraintDefaults:
    """Test default constraint values."""

    def test_default_learning_rate_constraints(self):
        """Test default learning rate constraints."""
        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")

        assert settings.max_learning_rate == 1.0
        assert settings.min_learning_rate == 1e-7

    def test_default_batch_size_constraints(self):
        """Test default batch size constraints."""
        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")

        assert settings.min_batch_size == 1
        assert settings.max_batch_size == 512

    def test_default_command_allowlist(self):
        """Test default allowed commands."""
        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")

        assert "python" in settings.allowed_commands
        assert "git" in settings.allowed_commands
        assert "rm -rf" not in settings.allowed_commands  # Should not be there!


# ============================================================================
# Timeout Configuration Tests
# ============================================================================

@pytest.mark.unit
class TestTimeoutConfiguration:
    """Test timeout-related settings."""

    def test_default_timeouts(self):
        """Test default timeout values."""
        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")

        assert settings.llm_timeout == 120
        assert settings.command_timeout == 3600
        assert settings.max_training_time == 7200

    def test_test_environment_shorter_timeouts(self):
        """Test that test environment has shorter timeouts."""
        settings = ARCSettings(
            environment="test",
            llm_endpoint="http://localhost:8000/v1"
        )

        assert settings.llm_timeout == 10  # Much shorter
        assert settings.command_timeout == 30
        assert settings.max_training_time == 60


# ============================================================================
# Logging Configuration Tests
# ============================================================================

@pytest.mark.unit
class TestLoggingConfiguration:
    """Test logging-related settings."""

    def test_default_logging_settings(self):
        """Test default logging configuration."""
        settings = ARCSettings(llm_endpoint="http://localhost:8000/v1")

        assert settings.log_level == "INFO"
        assert settings.log_format == "json"
        assert settings.log_to_file is True
        assert settings.log_to_console is True

    def test_test_environment_debug_logging(self):
        """Test that test environment uses DEBUG logging."""
        settings = ARCSettings(
            environment="test",
            llm_endpoint="http://localhost:8000/v1"
        )

        assert settings.log_level == "DEBUG"

    def test_prod_environment_info_logging(self):
        """Test that prod environment uses INFO logging."""
        settings = ARCSettings(
            environment="prod",
            llm_endpoint="http://localhost:8000/v1"
        )

        assert settings.log_level == "INFO"
