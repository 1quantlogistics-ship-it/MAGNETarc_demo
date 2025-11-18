"""
Experiment Config Generator: Translate agent proposals into executable training configs
========================================================================================

Converts multi-agent proposals into validated training configurations that can be
submitted to the control plane for execution.

Key responsibilities:
- Merge baseline config with proposed changes
- Validate against training parameter schema
- Generate experiment-specific config files
- Ensure safe parameter ranges
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class BaselineConfig:
    """Default training configuration."""
    # Model architecture
    model: str = "efficientnet_b3"
    input_size: int = 512
    num_classes: int = 2

    # Training hyperparameters
    learning_rate: float = 0.0001
    batch_size: int = 8
    epochs: int = 10
    optimizer: str = "adam"
    loss: str = "focal"

    # Data augmentation
    augmentations: List[str] = None

    # Dataset
    dataset: str = "refuge2"
    train_split: float = 0.8
    val_split: float = 0.2

    # Regularization
    dropout: float = 0.2
    weight_decay: float = 0.0001

    # Training settings
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    lr_factor: float = 0.5

    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    mixed_precision: bool = True

    def __post_init__(self):
        if self.augmentations is None:
            self.augmentations = ["flip", "rotate", "crop"]


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


class ExperimentConfigGenerator:
    """
    Generates executable training configs from agent proposals.

    Workflow:
    1. Load baseline config
    2. Apply proposed changes from agents
    3. Validate against parameter schema
    4. Write config file to experiments directory
    5. Return config dict for control plane submission
    """

    def __init__(
        self,
        baseline_config: Optional[Dict[str, Any]] = None,
        experiments_dir: str = "/Users/bengibson/Desktop/ARC/arc_clean/experiments",
        memory_path: str = "/Users/bengibson/Desktop/ARC/arc_clean/memory"
    ):
        """
        Initialize config generator.

        Args:
            baseline_config: Optional custom baseline (otherwise use defaults)
            experiments_dir: Directory to store experiment configs
            memory_path: Path to memory for loading constraints
        """
        self.baseline = baseline_config or asdict(BaselineConfig())
        self.experiments_dir = Path(experiments_dir)
        self.memory_path = Path(memory_path)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Parameter schema for validation
        self.param_schema = {
            "learning_rate": {"type": float, "min": 1e-6, "max": 1.0},
            "batch_size": {"type": int, "min": 1, "max": 128},
            "epochs": {"type": int, "min": 1, "max": 100},
            "dropout": {"type": float, "min": 0.0, "max": 0.9},
            "weight_decay": {"type": float, "min": 0.0, "max": 0.1},
            "input_size": {"type": int, "min": 128, "max": 1024},
            "early_stopping_patience": {"type": int, "min": 1, "max": 20},
            "reduce_lr_patience": {"type": int, "min": 1, "max": 10},
            "lr_factor": {"type": float, "min": 0.1, "max": 0.9},
            "train_split": {"type": float, "min": 0.5, "max": 0.95},
            "val_split": {"type": float, "min": 0.05, "max": 0.5},
        }

        # Valid categorical values
        self.valid_values = {
            "model": ["efficientnet_b0", "efficientnet_b3", "efficientnet_b5", "resnet50", "vit_base"],
            "optimizer": ["adam", "adamw", "sgd", "rmsprop"],
            "loss": ["focal", "cross_entropy", "dice", "bce"],
            "dataset": ["refuge2", "drions", "rim_one", "combined"],
            "device": ["cuda", "cpu", "mps"],
        }

        # Valid augmentation types
        self.valid_augmentations = [
            "flip", "rotate", "crop", "brightness", "contrast",
            "gaussian_noise", "gaussian_blur", "elastic_transform",
            "grid_distortion", "coarse_dropout"
        ]

    def generate_config(
        self,
        experiment_id: str,
        proposal: Dict[str, Any],
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Generate training config from proposal.

        Args:
            experiment_id: Unique experiment identifier
            proposal: Agent proposal with config_changes dict
            validate: Whether to validate against schema

        Returns:
            Complete training config dict

        Raises:
            ConfigValidationError: If validation fails
        """
        # Start with baseline
        config = self.baseline.copy()

        # Apply proposed changes
        config_changes = proposal.get("changes", {})
        for param, value in config_changes.items():
            # Handle special cases
            if param == "add_layer":
                # Architecture changes need special handling
                config.setdefault("architecture_changes", [])
                config["architecture_changes"].append({
                    "type": "add_layer",
                    "layer": value,
                    "num_heads": config_changes.get("num_heads", 8)
                })
            elif param == "augmentations":
                # Augmentation changes
                if isinstance(value, list):
                    config["augmentations"] = value
                else:
                    config["augmentations"].append(value)
            else:
                # Standard parameter update
                config[param] = value

        # Add experiment metadata
        config["experiment_id"] = experiment_id
        config["proposal_type"] = proposal.get("type", "unknown")
        config["risk_level"] = proposal.get("risk_level", "medium")
        config["description"] = proposal.get("description", "")
        config["generated_at"] = datetime.now().isoformat()

        # Validate if requested
        if validate:
            self._validate_config(config)

        # Load and apply constraints
        try:
            constraints = self._load_constraints()
            self._apply_constraints(config, constraints)
        except Exception as e:
            # Constraints optional in early development
            print(f"Warning: Could not load constraints: {e}")

        # Write config file
        self._write_config_file(experiment_id, config)

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate config against parameter schema.

        Raises:
            ConfigValidationError: If validation fails
        """
        errors = []

        # Validate numeric parameters
        for param, schema in self.param_schema.items():
            if param in config:
                value = config[param]

                # Type check
                if not isinstance(value, schema["type"]):
                    errors.append(f"{param}: Expected {schema['type'].__name__}, got {type(value).__name__}")
                    continue

                # Range check
                if "min" in schema and value < schema["min"]:
                    errors.append(f"{param}={value} below minimum ({schema['min']})")
                if "max" in schema and value > schema["max"]:
                    errors.append(f"{param}={value} above maximum ({schema['max']})")

        # Validate categorical parameters
        for param, valid_options in self.valid_values.items():
            if param in config:
                value = config[param]
                if value not in valid_options:
                    errors.append(f"{param}='{value}' not in valid options: {valid_options}")

        # Validate augmentations
        if "augmentations" in config:
            augs = config["augmentations"]
            if isinstance(augs, list):
                for aug in augs:
                    if aug not in self.valid_augmentations:
                        errors.append(f"Invalid augmentation: '{aug}'. Valid: {self.valid_augmentations}")

        # Check split consistency
        if "train_split" in config and "val_split" in config:
            total = config["train_split"] + config["val_split"]
            if not (0.99 <= total <= 1.01):  # Allow small float errors
                errors.append(f"train_split + val_split = {total}, should equal 1.0")

        if errors:
            raise ConfigValidationError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    def _load_constraints(self) -> Dict[str, Any]:
        """Load constraints from memory."""
        constraints_path = self.memory_path / "constraints.json"
        if constraints_path.exists():
            with open(constraints_path, 'r') as f:
                return json.load(f)
        return {"forbidden_ranges": []}

    def _apply_constraints(self, config: Dict[str, Any], constraints: Dict[str, Any]) -> None:
        """
        Apply learned constraints to config.

        Raises:
            ConfigValidationError: If config violates hard constraints
        """
        forbidden_ranges = constraints.get("forbidden_ranges", [])
        violations = []

        for forbidden in forbidden_ranges:
            param = forbidden.get("param")
            if param in config:
                value = config[param]
                min_val = forbidden.get("min")
                max_val = forbidden.get("max")

                if min_val is not None and value < min_val:
                    violations.append(f"{param}={value} violates constraint (min={min_val})")
                if max_val is not None and value > max_val:
                    violations.append(f"{param}={value} violates constraint (max={max_val})")

        if violations:
            raise ConfigValidationError(f"Constraint violations:\n" + "\n".join(f"  - {v}" for v in violations))

    def _write_config_file(self, experiment_id: str, config: Dict[str, Any]) -> Path:
        """
        Write config to experiment directory.

        Args:
            experiment_id: Experiment identifier
            config: Complete config dict

        Returns:
            Path to written config file
        """
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Write YAML (human-readable)
        yaml_path = exp_dir / "config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Write JSON (machine-readable)
        json_path = exp_dir / "config.json"
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)

        return yaml_path

    def validate_proposal(self, proposal: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate proposal without generating full config.

        Args:
            proposal: Agent proposal to validate

        Returns:
            (is_valid, error_messages)
        """
        try:
            # Create temporary config
            temp_config = self.baseline.copy()
            for param, value in proposal.get("changes", {}).items():
                temp_config[param] = value

            # Validate
            self._validate_config(temp_config)
            return (True, [])

        except ConfigValidationError as e:
            return (False, str(e).split('\n'))

    def get_baseline_config(self) -> Dict[str, Any]:
        """Get current baseline configuration."""
        return self.baseline.copy()

    def update_baseline(self, successful_config: Dict[str, Any]) -> None:
        """
        Update baseline based on successful experiment.

        Args:
            successful_config: Config from successful experiment
        """
        # Extract core parameters (exclude metadata)
        exclude_keys = {"experiment_id", "proposal_type", "risk_level", "description", "generated_at"}

        for key, value in successful_config.items():
            if key not in exclude_keys and key in self.baseline:
                self.baseline[key] = value


def get_config_generator(
    baseline_config: Optional[Dict[str, Any]] = None,
    experiments_dir: Optional[str] = None,
    memory_path: Optional[str] = None
) -> ExperimentConfigGenerator:
    """
    Factory function to get config generator instance.

    Args:
        baseline_config: Optional custom baseline
        experiments_dir: Optional custom experiments directory
        memory_path: Optional custom memory path

    Returns:
        ExperimentConfigGenerator instance
    """
    kwargs = {}
    if baseline_config is not None:
        kwargs["baseline_config"] = baseline_config
    if experiments_dir is not None:
        kwargs["experiments_dir"] = experiments_dir
    if memory_path is not None:
        kwargs["memory_path"] = memory_path

    return ExperimentConfigGenerator(**kwargs)
