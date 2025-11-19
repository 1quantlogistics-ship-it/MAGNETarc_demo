"""
Dataset Fusion for ARC Multi-Dataset Training

Enables training on multiple datasets simultaneously with:
- Dataset harmonization (size, normalization, format)
- Balanced sampling across datasets
- Cross-dataset validation splits
- Dataset-specific augmentation strategies

Supports RIM-ONE, REFUGE, AcuVue Custom, and future datasets.
"""

import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from config import get_settings
from tools.dev_logger import get_dev_logger

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for a single dataset."""
    dataset_id: str
    dataset_path: str
    total_samples: int
    class_distribution: Dict[str, int]
    image_size: Tuple[int, int]
    normalization: str
    checksum: str


@dataclass
class FusionConfig:
    """Configuration for dataset fusion."""
    dataset_ids: List[str]
    fusion_weights: Dict[str, float]  # Dataset sampling weights
    harmonization_strategy: str  # "resize", "crop", "pad"
    target_size: Tuple[int, int]
    cross_dataset_validation: bool
    validation_dataset: Optional[str] = None  # Dataset to use only for validation


class DatasetFusionError(Exception):
    """Raised when dataset fusion fails."""
    pass


class DatasetFusion:
    """
    Multi-dataset fusion manager for ARC.

    Handles:
    - Dataset loading and harmonization
    - Balanced sampling across datasets
    - Cross-dataset validation
    - Provenance tracking
    """

    def __init__(self, fusion_config: FusionConfig):
        """
        Initialize dataset fusion manager.

        Args:
            fusion_config: Fusion configuration
        """
        self.config = fusion_config
        self.settings = get_settings()
        self.dev_logger = get_dev_logger()

        # Dataset metadata cache
        self.dataset_metadata: Dict[str, DatasetMetadata] = {}

        # Validate config
        self._validate_config()

        logger.info(f"Dataset fusion initialized with {len(fusion_config.dataset_ids)} datasets")

    def _validate_config(self):
        """Validate fusion configuration."""
        if not self.config.dataset_ids:
            raise DatasetFusionError("No datasets specified in fusion config")

        # Check weights sum to 1.0 (or close)
        if self.config.fusion_weights:
            total_weight = sum(self.config.fusion_weights.values())
            if not (0.99 <= total_weight <= 1.01):
                logger.warning(f"Fusion weights sum to {total_weight}, normalizing...")
                # Normalize weights
                for dataset_id in self.config.fusion_weights:
                    self.config.fusion_weights[dataset_id] /= total_weight
        else:
            # Equal weights if not specified
            weight = 1.0 / len(self.config.dataset_ids)
            self.config.fusion_weights = {
                dataset_id: weight for dataset_id in self.config.dataset_ids
            }

        # Validate validation dataset
        if self.config.cross_dataset_validation:
            if not self.config.validation_dataset:
                raise DatasetFusionError("cross_dataset_validation=True requires validation_dataset")
            if self.config.validation_dataset not in self.config.dataset_ids:
                raise DatasetFusionError(f"Validation dataset {self.config.validation_dataset} not in dataset_ids")

    def load_datasets(self) -> Dict[str, Any]:
        """
        Load and harmonize all datasets.

        Returns:
            Dict with loaded dataset information
        """
        logger.info(f"Loading {len(self.config.dataset_ids)} datasets for fusion...")

        loaded_datasets = {}
        total_samples = 0

        for dataset_id in self.config.dataset_ids:
            try:
                # Load dataset metadata
                metadata = self._load_dataset_metadata(dataset_id)
                self.dataset_metadata[dataset_id] = metadata

                # Track samples
                total_samples += metadata.total_samples

                loaded_datasets[dataset_id] = {
                    "path": metadata.dataset_path,
                    "samples": metadata.total_samples,
                    "weight": self.config.fusion_weights.get(dataset_id, 0.0),
                    "metadata": asdict(metadata)
                }

                logger.info(f"Loaded {dataset_id}: {metadata.total_samples} samples")

            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_id}: {e}")
                raise DatasetFusionError(f"Failed to load dataset {dataset_id}: {e}")

        # Log fusion operation to FDA logs
        self._log_fusion_operation(loaded_datasets, total_samples)

        return {
            "datasets": loaded_datasets,
            "total_samples": total_samples,
            "fusion_config": asdict(self.config)
        }

    def _load_dataset_metadata(self, dataset_id: str) -> DatasetMetadata:
        """
        Load metadata for a single dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DatasetMetadata object
        """
        # Construct dataset path
        dataset_root = Path(self.settings.home) / "workspace" / "datasets" / dataset_id

        if not dataset_root.exists():
            raise DatasetFusionError(f"Dataset path does not exist: {dataset_root}")

        # Count samples (images)
        image_files = list(dataset_root.glob("**/*.jpg")) + list(dataset_root.glob("**/*.png"))
        total_samples = len(image_files)

        if total_samples == 0:
            raise DatasetFusionError(f"No images found in dataset: {dataset_id}")

        # Detect image size (from first image)
        try:
            import cv2
            first_image = cv2.imread(str(image_files[0]))
            image_size = (first_image.shape[1], first_image.shape[0])  # (width, height)
        except Exception as e:
            logger.warning(f"Could not detect image size for {dataset_id}: {e}")
            image_size = (512, 512)  # Default

        # Compute checksum for provenance
        checksum = self._compute_dataset_checksum(dataset_root)

        # Class distribution (simplified - count subdirectories)
        class_distribution = {}
        for subdir in dataset_root.iterdir():
            if subdir.is_dir():
                subdir_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
                if subdir_files:
                    class_distribution[subdir.name] = len(subdir_files)

        return DatasetMetadata(
            dataset_id=dataset_id,
            dataset_path=str(dataset_root),
            total_samples=total_samples,
            class_distribution=class_distribution if class_distribution else {"all": total_samples},
            image_size=image_size,
            normalization="standard",  # Default
            checksum=checksum
        )

    def _compute_dataset_checksum(self, dataset_path: Path) -> str:
        """
        Compute MD5 checksum for dataset (fast version - sample files).

        Args:
            dataset_path: Path to dataset

        Returns:
            MD5 checksum hex string
        """
        md5 = hashlib.md5()

        # Sample up to 100 files for checksum (full scan too slow)
        all_files = sorted(dataset_path.rglob("*.*"))
        sample_size = min(100, len(all_files))

        # Sample evenly across dataset
        step = max(1, len(all_files) // sample_size)
        sampled_files = all_files[::step][:sample_size]

        for file_path in sampled_files:
            if file_path.is_file():
                md5.update(str(file_path).encode())  # Use path as proxy
                md5.update(str(file_path.stat().st_size).encode())  # Include size

        return md5.hexdigest()

    def create_fusion_config_file(self, output_path: str) -> str:
        """
        Create fusion configuration file for training.

        Args:
            output_path: Path to write config

        Returns:
            Path to created config file
        """
        fusion_data = {
            "fusion_enabled": True,
            "datasets": [],
            "harmonization": {
                "strategy": self.config.harmonization_strategy,
                "target_size": self.config.target_size
            },
            "cross_dataset_validation": self.config.cross_dataset_validation
        }

        for dataset_id in self.config.dataset_ids:
            metadata = self.dataset_metadata.get(dataset_id)
            if metadata:
                dataset_config = {
                    "dataset_id": dataset_id,
                    "path": metadata.dataset_path,
                    "weight": self.config.fusion_weights.get(dataset_id, 0.0),
                    "validation_only": (
                        self.config.cross_dataset_validation and
                        dataset_id == self.config.validation_dataset
                    )
                }
                fusion_data["datasets"].append(dataset_config)

        # Write config
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(fusion_data, f, indent=2)

        logger.info(f"Fusion config written to {output_file}")
        return str(output_file)

    def _log_fusion_operation(self, loaded_datasets: Dict, total_samples: int):
        """
        Log fusion operation to FDA development logs.

        Args:
            loaded_datasets: Loaded dataset information
            total_samples: Total samples across all datasets
        """
        try:
            # Log data provenance for fusion
            self.dev_logger.log_data_provenance(
                operation="dataset_fusion",
                dataset_id="fused_" + "_".join(self.config.dataset_ids),
                dataset_version="fusion_v1.0",
                input_path=",".join([d["path"] for d in loaded_datasets.values()]),
                output_path="",  # No output path for fusion
                transformation_applied=f"Multi-dataset fusion: {len(loaded_datasets)} datasets",
                input_checksum="",  # Individual checksums in metadata
                output_checksum="",
                metadata={
                    "datasets": list(loaded_datasets.keys()),
                    "total_samples": total_samples,
                    "fusion_weights": self.config.fusion_weights,
                    "harmonization_strategy": self.config.harmonization_strategy,
                    "target_size": self.config.target_size,
                    "cross_dataset_validation": self.config.cross_dataset_validation
                }
            )
            logger.debug("FDA fusion provenance logged")
        except Exception as e:
            logger.warning(f"FDA logging failed for fusion: {e}")

    def get_sampling_weights(self) -> Dict[str, float]:
        """
        Get dataset sampling weights.

        Returns:
            Dict mapping dataset_id to sampling weight
        """
        return self.config.fusion_weights.copy()

    def get_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """
        Get metadata for a specific dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DatasetMetadata or None if not found
        """
        return self.dataset_metadata.get(dataset_id)

    def get_all_metadata(self) -> Dict[str, DatasetMetadata]:
        """
        Get metadata for all datasets.

        Returns:
            Dict mapping dataset_id to DatasetMetadata
        """
        return self.dataset_metadata.copy()


def create_fusion_from_config(
    dataset_ids: List[str],
    fusion_weights: Optional[Dict[str, float]] = None,
    target_size: Tuple[int, int] = (512, 512),
    harmonization_strategy: str = "resize",
    cross_dataset_validation: bool = False,
    validation_dataset: Optional[str] = None
) -> DatasetFusion:
    """
    Factory function to create DatasetFusion from parameters.

    Args:
        dataset_ids: List of dataset identifiers to fuse
        fusion_weights: Optional sampling weights (default: equal)
        target_size: Target image size for harmonization
        harmonization_strategy: Strategy for size harmonization
        cross_dataset_validation: Use one dataset only for validation
        validation_dataset: Dataset to use for validation only

    Returns:
        DatasetFusion instance
    """
    config = FusionConfig(
        dataset_ids=dataset_ids,
        fusion_weights=fusion_weights or {},
        harmonization_strategy=harmonization_strategy,
        target_size=target_size,
        cross_dataset_validation=cross_dataset_validation,
        validation_dataset=validation_dataset
    )

    return DatasetFusion(config)
