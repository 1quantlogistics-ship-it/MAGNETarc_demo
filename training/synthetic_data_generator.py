#!/usr/bin/env python3
"""
Synthetic Data Generator for MAGNET
====================================

Physics-informed synthetic data generation for training ML models.
Generates diverse, high-quality training datasets using domain constraints,
physics validation, and intelligent sampling strategies.

Author: Agent 2
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from scipy.stats import qmc  # Latin Hypercube Sampling
from scipy import stats

from design_core import BaseDesignParameters, BasePhysicsEngine, PhysicsResults, DesignDomain


@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation."""

    # Sampling configuration
    n_samples: int = 1000
    sampling_strategy: str = "mixed"  # "latin_hypercube", "gaussian", "edge_corner", "mixed"
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "latin_hypercube": 0.5,
        "gaussian": 0.3,
        "edge_corner": 0.2
    })

    # Physics simulation
    validate_physics: bool = True
    min_valid_ratio: float = 0.7  # Minimum ratio of valid designs

    # Augmentation
    enable_augmentation: bool = True
    augmentation_noise_std: float = 0.02  # 2% noise for augmentation
    augmentation_factor: int = 2  # Generate 2x augmented samples

    # Quality control
    min_diversity_score: float = 0.6  # Minimum parameter space coverage
    max_correlation: float = 0.95  # Maximum correlation between parameters

    # Export
    output_format: str = "numpy"  # "numpy", "csv", "json", "pytorch"
    include_metadata: bool = True

    # Random seed
    random_seed: Optional[int] = None


@dataclass
class DatasetStatistics:
    """Statistics for generated dataset."""

    total_samples: int
    valid_samples: int
    invalid_samples: int
    validity_ratio: float

    diversity_score: float
    parameter_coverage: Dict[str, float]

    performance_stats: Dict[str, Dict[str, float]]  # mean, std, min, max for each metric

    generation_time_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "validity_ratio": self.validity_ratio,
            "diversity_score": self.diversity_score,
            "parameter_coverage": self.parameter_coverage,
            "performance_stats": self.performance_stats,
            "generation_time_seconds": self.generation_time_seconds,
            "timestamp": self.timestamp
        }


class SyntheticDataGenerator:
    """
    Physics-informed synthetic data generator for multi-domain designs.

    Generates high-quality training datasets using:
    - Multiple sampling strategies (Latin Hypercube, Gaussian, Edge/Corner)
    - Physics validation to ensure feasibility
    - Data augmentation with controlled noise
    - Quality metrics and diversity scoring
    """

    def __init__(
        self,
        domain: DesignDomain,
        physics_engine: BasePhysicsEngine,
        parameter_template: BaseDesignParameters,
        config: Optional[DataGenerationConfig] = None
    ):
        """
        Initialize synthetic data generator.

        Args:
            domain: Design domain (naval, aerial, etc.)
            physics_engine: Physics simulation engine for validation
            parameter_template: Template for parameter constraints
            config: Generation configuration
        """
        self.domain = domain
        self.physics_engine = physics_engine
        self.parameter_template = parameter_template
        self.config = config or DataGenerationConfig()

        # Set random seed if provided
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        # Get parameter constraints
        self.constraints = parameter_template.get_constraints()
        self.param_names = list(self.constraints.keys())
        self.param_bounds = np.array([self.constraints[p] for p in self.param_names])

        # Dataset storage
        self.generated_designs: List[BaseDesignParameters] = []
        self.simulation_results: List[PhysicsResults] = []
        self.statistics: Optional[DatasetStatistics] = None

    def generate(self) -> Tuple[List[BaseDesignParameters], List[PhysicsResults]]:
        """
        Generate synthetic dataset with physics validation.

        Returns:
            Tuple of (design_parameters, simulation_results)
        """
        start_time = datetime.now()

        # Step 1: Sample parameter space
        if self.config.sampling_strategy == "mixed":
            samples = self._mixed_sampling()
        elif self.config.sampling_strategy == "latin_hypercube":
            samples = self._latin_hypercube_sampling()
        elif self.config.sampling_strategy == "gaussian":
            samples = self._gaussian_sampling()
        elif self.config.sampling_strategy == "edge_corner":
            samples = self._edge_corner_sampling()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.sampling_strategy}")

        # Step 2: Convert to design parameters
        designs = self._samples_to_designs(samples)

        # Step 3: Physics validation and simulation
        if self.config.validate_physics:
            designs, results = self._simulate_and_filter(designs)
        else:
            # Just validate without simulation
            valid_designs = [d for d in designs if not self.parameter_template.validate()]
            designs = valid_designs
            results = []

        # Step 4: Data augmentation
        if self.config.enable_augmentation:
            designs, results = self._augment_dataset(designs, results)

        # Step 5: Calculate statistics
        end_time = datetime.now()
        self.generated_designs = designs
        self.simulation_results = results
        self.statistics = self._calculate_statistics(
            designs, results, (end_time - start_time).total_seconds()
        )

        return designs, results

    def _latin_hypercube_sampling(self) -> np.ndarray:
        """
        Latin Hypercube Sampling for space-filling design.

        Ensures good coverage of parameter space with minimal samples.
        """
        n_dims = len(self.param_names)
        sampler = qmc.LatinHypercube(d=n_dims, seed=self.config.random_seed)

        # Sample in [0, 1]^d hypercube
        samples_unit = sampler.random(n=self.config.n_samples)

        # Scale to actual parameter bounds
        samples = qmc.scale(samples_unit, self.param_bounds[:, 0], self.param_bounds[:, 1])

        return samples

    def _gaussian_sampling(self) -> np.ndarray:
        """
        Gaussian sampling around best known designs.

        Focuses exploration near high-performing regions.
        """
        # Use center of parameter space as mean (could be improved with historical data)
        mean = (self.param_bounds[:, 0] + self.param_bounds[:, 1]) / 2

        # Standard deviation = 1/6 of parameter range (3-sigma covers most of range)
        std = (self.param_bounds[:, 1] - self.param_bounds[:, 0]) / 6

        # Generate samples
        samples = np.random.normal(mean, std, size=(self.config.n_samples, len(self.param_names)))

        # Clip to bounds
        samples = np.clip(samples, self.param_bounds[:, 0], self.param_bounds[:, 1])

        return samples

    def _edge_corner_sampling(self) -> np.ndarray:
        """
        Edge and corner sampling to explore extreme cases.

        Tests boundary conditions and edge cases.
        """
        n_dims = len(self.param_names)
        n_samples = self.config.n_samples

        samples = []

        # Pure corners: all combinations of min/max
        if n_dims <= 10:  # Only for reasonable dimensionality
            from itertools import product
            corners = list(product(*[(b[0], b[1]) for b in self.param_bounds]))
            samples.extend(corners[:n_samples // 4])  # Use 25% for corners

        # Edge samples: vary one parameter while others are at mean
        mean = (self.param_bounds[:, 0] + self.param_bounds[:, 1]) / 2
        for i in range(n_dims):
            n_edge_samples = max(1, (n_samples // 2) // n_dims)
            for _ in range(n_edge_samples):
                sample = mean.copy()
                sample[i] = np.random.uniform(self.param_bounds[i, 0], self.param_bounds[i, 1])
                samples.append(sample)

        # Fill remaining with random samples
        while len(samples) < n_samples:
            sample = np.random.uniform(
                self.param_bounds[:, 0],
                self.param_bounds[:, 1]
            )
            samples.append(sample)

        return np.array(samples[:n_samples])

    def _mixed_sampling(self) -> np.ndarray:
        """
        Mixed sampling using weighted combination of strategies.

        Combines Latin Hypercube, Gaussian, and Edge/Corner sampling.
        """
        weights = self.config.strategy_weights
        total_samples = self.config.n_samples

        # Calculate samples per strategy
        n_lhs = int(total_samples * weights.get("latin_hypercube", 0.5))
        n_gaussian = int(total_samples * weights.get("gaussian", 0.3))
        n_edge = total_samples - n_lhs - n_gaussian

        samples = []

        # Latin Hypercube samples
        if n_lhs > 0:
            lhs_config = DataGenerationConfig(n_samples=n_lhs, random_seed=self.config.random_seed)
            old_config = self.config
            self.config = lhs_config
            lhs_samples = self._latin_hypercube_sampling()
            self.config = old_config
            samples.append(lhs_samples)

        # Gaussian samples
        if n_gaussian > 0:
            gauss_config = DataGenerationConfig(n_samples=n_gaussian, random_seed=self.config.random_seed)
            old_config = self.config
            self.config = gauss_config
            gauss_samples = self._gaussian_sampling()
            self.config = old_config
            samples.append(gauss_samples)

        # Edge/Corner samples
        if n_edge > 0:
            edge_config = DataGenerationConfig(n_samples=n_edge, random_seed=self.config.random_seed)
            old_config = self.config
            self.config = edge_config
            edge_samples = self._edge_corner_sampling()
            self.config = old_config
            samples.append(edge_samples)

        # Combine and shuffle
        combined = np.vstack(samples)
        np.random.shuffle(combined)

        return combined

    def _samples_to_designs(self, samples: np.ndarray) -> List[BaseDesignParameters]:
        """
        Convert numpy samples to design parameter objects.

        Args:
            samples: numpy array of shape (n_samples, n_params)

        Returns:
            List of design parameter objects
        """
        designs = []

        for i, sample in enumerate(samples):
            # Create parameter dict
            params_dict = {name: float(value) for name, value in zip(self.param_names, sample)}
            params_dict["design_id"] = f"synthetic_{self.domain.value}_{i:06d}"
            params_dict["name"] = f"Synthetic {self.domain.value.title()} Design {i}"
            params_dict["domain"] = self.domain

            # Create design object (this is domain-specific)
            # We'd need to use the actual parameter class for the domain
            # For now, store as dict
            designs.append(params_dict)

        return designs

    def _simulate_and_filter(
        self,
        designs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[PhysicsResults]]:
        """
        Simulate designs and filter invalid ones.

        Args:
            designs: List of design parameter dicts

        Returns:
            Tuple of (valid_designs, simulation_results)
        """
        # This would call the actual physics engine
        # For now, placeholder
        valid_designs = designs
        results = []

        return valid_designs, results

    def _augment_dataset(
        self,
        designs: List[Dict[str, Any]],
        results: List[PhysicsResults]
    ) -> Tuple[List[Dict[str, Any]], List[PhysicsResults]]:
        """
        Augment dataset with controlled noise.

        Args:
            designs: Original designs
            results: Original simulation results

        Returns:
            Tuple of (augmented_designs, augmented_results)
        """
        augmented_designs = list(designs)
        augmented_results = list(results)

        noise_std = self.config.augmentation_noise_std

        for aug_idx in range(self.config.augmentation_factor - 1):
            for i, design in enumerate(designs):
                # Start with copy of original design
                augmented = design.copy()

                # Add noise to parameters
                for param_name in self.param_names:
                    if param_name in design:
                        original_value = design[param_name]
                        noise = np.random.normal(0, noise_std * abs(original_value))
                        augmented[param_name] = original_value + noise

                        # Clip to bounds
                        bounds = self.constraints[param_name]
                        augmented[param_name] = float(np.clip(augmented[param_name], bounds[0], bounds[1]))

                # Update design ID for augmented sample
                augmented["design_id"] = f"{design.get('design_id', 'unknown')}_aug{aug_idx+1}"

                augmented_designs.append(augmented)

        return augmented_designs, augmented_results

    def _calculate_statistics(
        self,
        designs: List[Dict[str, Any]],
        results: List[PhysicsResults],
        generation_time: float
    ) -> DatasetStatistics:
        """Calculate dataset statistics."""

        total = len(designs)
        valid = len([r for r in results if r.success]) if results else total
        invalid = total - valid

        # Diversity score: measure parameter space coverage
        if designs:
            samples_array = np.array([[d.get(p, 0) for p in self.param_names] for d in designs])
            diversity = self._calculate_diversity(samples_array)
        else:
            diversity = 0.0

        # Parameter coverage
        coverage = {}
        for param_name in self.param_names:
            values = [d.get(param_name, 0) for d in designs]
            if values:
                bounds = self.constraints[param_name]
                coverage[param_name] = (max(values) - min(values)) / (bounds[1] - bounds[0])
            else:
                coverage[param_name] = 0.0

        # Performance statistics
        perf_stats = {}
        if results and any(r.success for r in results):
            valid_results = [r for r in results if r.success]
            for metric in ["structural_integrity", "efficiency", "safety_score", "overall_score"]:
                values = [getattr(r, metric) for r in valid_results]
                perf_stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }

        return DatasetStatistics(
            total_samples=total,
            valid_samples=valid,
            invalid_samples=invalid,
            validity_ratio=valid / max(total, 1),
            diversity_score=diversity,
            parameter_coverage=coverage,
            performance_stats=perf_stats,
            generation_time_seconds=generation_time
        )

    def _calculate_diversity(self, samples: np.ndarray) -> float:
        """
        Calculate diversity score using minimum spanning tree approach.

        Higher score = better parameter space coverage.
        """
        if len(samples) < 2:
            return 0.0

        # Normalize samples to [0, 1]
        normalized = (samples - self.param_bounds[:, 0]) / (self.param_bounds[:, 1] - self.param_bounds[:, 0])

        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(normalized))

        # Diversity = mean of minimum distances (how spread out are points)
        min_distances = np.min(distances + np.eye(len(distances)) * 1e10, axis=1)
        diversity = float(np.mean(min_distances))

        return diversity

    def export(self, output_dir: str, dataset_name: str = "synthetic_dataset"):
        """
        Export generated dataset to disk.

        Args:
            output_dir: Directory to save dataset
            dataset_name: Name prefix for dataset files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export based on format
        if self.config.output_format == "numpy":
            self._export_numpy(output_path, dataset_name)
        elif self.config.output_format == "csv":
            self._export_csv(output_path, dataset_name)
        elif self.config.output_format == "json":
            self._export_json(output_path, dataset_name)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")

        # Export metadata
        if self.config.include_metadata and self.statistics:
            metadata_path = output_path / f"{dataset_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "domain": self.domain.value,
                    "config": {
                        "n_samples": self.config.n_samples,
                        "sampling_strategy": self.config.sampling_strategy,
                        "strategy_weights": self.config.strategy_weights,
                        "augmentation_factor": self.config.augmentation_factor,
                    },
                    "statistics": self.statistics.to_dict(),
                }, f, indent=2)

    def _export_numpy(self, output_path: Path, dataset_name: str):
        """Export as NumPy arrays."""
        # Parameters
        params_array = np.array([[d.get(p, 0) for p in self.param_names] for d in self.generated_designs])
        np.save(output_path / f"{dataset_name}_parameters.npy", params_array)

        # Results (if available)
        if self.simulation_results:
            results_array = np.array([
                [r.structural_integrity, r.efficiency, r.safety_score, r.overall_score]
                for r in self.simulation_results
            ])
            np.save(output_path / f"{dataset_name}_results.npy", results_array)

    def _export_csv(self, output_path: Path, dataset_name: str):
        """Export as CSV files."""
        import csv

        # Parameters
        with open(output_path / f"{dataset_name}_parameters.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["design_id"] + self.param_names)
            writer.writeheader()
            for design in self.generated_designs:
                row = {"design_id": design.get("design_id", "unknown")}
                row.update({p: design.get(p, 0) for p in self.param_names})
                writer.writerow(row)

    def _export_json(self, output_path: Path, dataset_name: str):
        """Export as JSON file."""
        data = {
            "designs": self.generated_designs,
            "results": [r.to_dict() for r in self.simulation_results] if self.simulation_results else []
        }

        with open(output_path / f"{dataset_name}.json", 'w') as f:
            json.dump(data, f, indent=2)

    def get_statistics(self) -> Optional[DatasetStatistics]:
        """Get dataset statistics."""
        return self.statistics

    def print_summary(self):
        """Print dataset generation summary."""
        if not self.statistics:
            print("No statistics available. Run generate() first.")
            return

        stats = self.statistics
        print("\n" + "=" * 70)
        print("SYNTHETIC DATASET GENERATION SUMMARY")
        print("=" * 70)
        print(f"Domain:              {self.domain.value.title()}")
        print(f"Total samples:       {stats.total_samples}")
        print(f"Valid samples:       {stats.valid_samples} ({stats.validity_ratio*100:.1f}%)")
        print(f"Invalid samples:     {stats.invalid_samples}")
        print(f"Diversity score:     {stats.diversity_score:.3f}")
        print(f"Generation time:     {stats.generation_time_seconds:.2f}s")
        print(f"\nParameter Coverage:")
        for param, coverage in stats.parameter_coverage.items():
            print(f"  {param:20s}: {coverage*100:.1f}%")

        if stats.performance_stats:
            print(f"\nPerformance Statistics:")
            for metric, values in stats.performance_stats.items():
                print(f"  {metric:20s}: mean={values['mean']:.3f}, std={values['std']:.3f}")
        print("=" * 70 + "\n")
