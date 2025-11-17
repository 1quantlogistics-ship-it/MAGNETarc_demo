"""
Configuration Loader: Load and validate YAML configurations
============================================================

Loads agent registry, model endpoints, and consensus rules from YAML files.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """
    Load and validate ARC configuration files.

    Loads:
    - agents.yaml: Agent registry definitions
    - models.yaml: Model endpoint configurations
    - consensus.yaml: Voting and consensus rules
    """

    def __init__(self, config_dir: str = "/workspace/arc/config"):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_agents_config(self) -> Dict[str, Any]:
        """
        Load agent registry configuration.

        Returns:
            Agents config dictionary
        """
        if "agents" in self._cache:
            return self._cache["agents"]

        config_path = self.config_dir / "agents.yaml"

        if not config_path.exists():
            # Return default config
            config = self._get_default_agents_config()
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        self._cache["agents"] = config
        return config

    def load_models_config(self) -> Dict[str, Any]:
        """
        Load model endpoints configuration.

        Returns:
            Models config dictionary
        """
        if "models" in self._cache:
            return self._cache["models"]

        config_path = self.config_dir / "models.yaml"

        if not config_path.exists():
            # Return default config
            config = self._get_default_models_config()
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        self._cache["models"] = config
        return config

    def load_consensus_config(self) -> Dict[str, Any]:
        """
        Load consensus voting configuration.

        Returns:
            Consensus config dictionary
        """
        if "consensus" in self._cache:
            return self._cache["consensus"]

        config_path = self.config_dir / "consensus.yaml"

        if not config_path.exists():
            # Return default config
            config = self._get_default_consensus_config()
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        self._cache["consensus"] = config
        return config

    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files.

        Returns:
            Dictionary with all configs
        """
        return {
            "agents": self.load_agents_config(),
            "models": self.load_models_config(),
            "consensus": self.load_consensus_config()
        }

    def save_agents_config(self, config: Dict[str, Any]) -> bool:
        """
        Save agent registry configuration.

        Args:
            config: Config to save

        Returns:
            True if save successful
        """
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            config_path = self.config_dir / "agents.yaml"

            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            self._cache["agents"] = config
            return True
        except Exception as e:
            print(f"Error saving agents config: {e}")
            return False

    def save_models_config(self, config: Dict[str, Any]) -> bool:
        """
        Save models configuration.

        Args:
            config: Config to save

        Returns:
            True if save successful
        """
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            config_path = self.config_dir / "models.yaml"

            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            self._cache["models"] = config
            return True
        except Exception as e:
            print(f"Error saving models config: {e}")
            return False

    def save_consensus_config(self, config: Dict[str, Any]) -> bool:
        """
        Save consensus configuration.

        Args:
            config: Config to save

        Returns:
            True if save successful
        """
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            config_path = self.config_dir / "consensus.yaml"

            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            self._cache["consensus"] = config
            return True
        except Exception as e:
            print(f"Error saving consensus config: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._cache.clear()

    # Default configurations

    @staticmethod
    def _get_default_agents_config() -> Dict[str, Any]:
        """Get default agents configuration."""
        return {
            "agents": [
                {
                    "id": "director_001",
                    "role": "director",
                    "model": "claude-sonnet-4.5",
                    "priority": "high",
                    "capabilities": ["strategy", "planning"],
                    "voting_weight": 2.0,
                    "offline": False
                },
                {
                    "id": "architect_001",
                    "role": "architect",
                    "model": "deepseek-r1",
                    "priority": "medium",
                    "capabilities": ["proposal_generation"],
                    "voting_weight": 1.5,
                    "offline": False
                },
                {
                    "id": "critic_001",
                    "role": "critic",
                    "model": "qwen2.5-32b",
                    "priority": "high",
                    "capabilities": ["safety_review", "constraint_checking"],
                    "voting_weight": 2.0,
                    "offline": False
                },
                {
                    "id": "critic_secondary_001",
                    "role": "critic_secondary",
                    "model": "deepseek-r1",
                    "priority": "high",
                    "capabilities": ["safety_review", "validation"],
                    "voting_weight": 1.8,
                    "offline": False
                },
                {
                    "id": "historian_001",
                    "role": "historian",
                    "model": "deepseek-r1",
                    "priority": "medium",
                    "capabilities": ["memory_management"],
                    "voting_weight": 1.0,
                    "offline": False
                },
                {
                    "id": "executor_001",
                    "role": "executor",
                    "model": "deepseek-r1",
                    "priority": "medium",
                    "capabilities": ["execution"],
                    "voting_weight": 1.0,
                    "offline": False
                },
                {
                    "id": "explorer_001",
                    "role": "explorer",
                    "model": "qwen2.5-32b",
                    "priority": "medium",
                    "capabilities": ["exploration", "proposal_generation"],
                    "voting_weight": 1.2,
                    "offline": False
                },
                {
                    "id": "parameter_scientist_001",
                    "role": "parameter_scientist",
                    "model": "deepseek-r1",
                    "priority": "medium",
                    "capabilities": ["proposal_generation", "exploration"],
                    "voting_weight": 1.5,
                    "offline": False
                },
                {
                    "id": "supervisor_001",
                    "role": "supervisor",
                    "model": "llama-3-8b-local",
                    "priority": "critical",
                    "capabilities": ["supervision", "validation", "safety_review"],
                    "voting_weight": 3.0,
                    "offline": True
                }
            ]
        }

    @staticmethod
    def _get_default_models_config() -> Dict[str, Any]:
        """Get default models configuration."""
        return {
            "models": [
                {
                    "id": "deepseek-r1",
                    "name": "DeepSeek R1",
                    "endpoint": "http://localhost:8000/generate",
                    "provider": "vllm",
                    "context_window": 32768,
                    "max_output_tokens": 4096,
                    "offline": False
                },
                {
                    "id": "claude-sonnet-4.5",
                    "name": "Claude Sonnet 4.5",
                    "endpoint": "https://api.anthropic.com/v1/messages",
                    "provider": "anthropic",
                    "context_window": 200000,
                    "max_output_tokens": 8192,
                    "offline": False,
                    "api_key_required": True
                },
                {
                    "id": "qwen2.5-32b",
                    "name": "Qwen 2.5 32B",
                    "endpoint": "http://localhost:8001/generate",
                    "provider": "vllm",
                    "context_window": 32768,
                    "max_output_tokens": 4096,
                    "offline": False
                },
                {
                    "id": "llama-3-8b-local",
                    "name": "Llama 3 8B (Local)",
                    "endpoint": "http://localhost:8002/generate",
                    "provider": "local",
                    "context_window": 8192,
                    "max_output_tokens": 2048,
                    "offline": True
                },
                {
                    "id": "mock-llm",
                    "name": "Mock LLM (Offline)",
                    "endpoint": "mock://offline",
                    "provider": "mock",
                    "context_window": 100000,
                    "max_output_tokens": 10000,
                    "offline": True
                }
            ]
        }

    @staticmethod
    def _get_default_consensus_config() -> Dict[str, Any]:
        """Get default consensus configuration."""
        return {
            "voting": {
                "consensus_threshold": 0.66,
                "min_votes_required": 2,
                "enable_confidence_weighting": True,
                "allow_abstentions": True
            },
            "conflict_resolution": {
                "default_strategy": "conservative",
                "supervisor_override_threshold": 0.9,
                "enable_supervisor_veto": True,
                "enable_director_override": True
            },
            "decision_rules": {
                "approve_threshold": 0.66,
                "reject_threshold": -0.66,
                "revision_priority": True
            }
        }


# Global config loader instance
_global_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: Optional[str] = None) -> ConfigLoader:
    """
    Get global config loader instance.

    Args:
        config_dir: Config directory (None = use default)

    Returns:
        ConfigLoader instance
    """
    global _global_loader

    if _global_loader is None:
        if config_dir is None:
            config_dir = os.environ.get("ARC_CONFIG_DIR", "/workspace/arc/config")
        _global_loader = ConfigLoader(config_dir)

    return _global_loader
