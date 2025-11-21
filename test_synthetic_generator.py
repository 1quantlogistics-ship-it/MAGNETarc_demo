#!/usr/bin/env python3
"""
Quick test of the Synthetic Data Generator
==========================================

Tests basic functionality without full domain implementations.
"""

import numpy as np
from training.synthetic_data_generator import (
    SyntheticDataGenerator,
    DataGenerationConfig
)
from design_core import DesignDomain, BaseDesignParameters
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass
class MockDesignParameters(BaseDesignParameters):
    """Mock design parameters for testing."""

    length: float = 10.0
    width: float = 5.0
    height: float = 3.0

    def validate(self) -> List[str]:
        """Validate parameters."""
        errors = []
        if self.length <= 0:
            errors.append("Length must be positive")
        if self.width <= 0:
            errors.append("Width must be positive")
        if self.height <= 0:
            errors.append("Height must be positive")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "design_id": self.design_id,
            "name": self.name,
            "domain": self.domain.value,
            "mass": self.mass,
            "length": self.length,
            "width": self.width,
            "height": self.height
        }

    def get_constraints(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter constraints."""
        return {
            "length": (5.0, 20.0),
            "width": (2.0, 10.0),
            "height": (1.0, 6.0),
            "mass": (100.0, 1000.0)
        }


def test_sampling_strategies():
    """Test different sampling strategies."""
    print("\n" + "="*70)
    print("TEST: Sampling Strategies")
    print("="*70)

    template = MockDesignParameters(
        domain=DesignDomain.NAVAL,
        design_id="template",
        name="Template",
        mass=500.0
    )

    strategies = ["latin_hypercube", "gaussian", "edge_corner", "mixed"]

    for strategy in strategies:
        print(f"\nTesting {strategy}...")

        config = DataGenerationConfig(
            n_samples=50,
            sampling_strategy=strategy,
            validate_physics=False,  # Skip physics for speed
            enable_augmentation=False,
            random_seed=42
        )

        generator = SyntheticDataGenerator(
            domain=DesignDomain.NAVAL,
            physics_engine=None,  # Not needed without validation
            parameter_template=template,
            config=config
        )

        designs, results = generator.generate()

        stats = generator.get_statistics()
        print(f"  Generated {stats.total_samples} samples")
        print(f"  Diversity score: {stats.diversity_score:.3f}")
        print(f"  Parameter coverage:")
        for param, coverage in stats.parameter_coverage.items():
            print(f"    {param}: {coverage*100:.1f}%")

        assert stats.total_samples == 50, f"Expected 50 samples, got {stats.total_samples}"
        print(f"  ✓ {strategy} sampling working correctly")

    print("\n✅ All sampling strategies working!")


def test_augmentation():
    """Test data augmentation."""
    print("\n" + "="*70)
    print("TEST: Data Augmentation")
    print("="*70)

    template = MockDesignParameters(
        domain=DesignDomain.NAVAL,
        design_id="template",
        name="Template",
        mass=500.0
    )

    config = DataGenerationConfig(
        n_samples=20,
        sampling_strategy="latin_hypercube",
        validate_physics=False,
        enable_augmentation=True,
        augmentation_factor=3,
        augmentation_noise_std=0.02,
        random_seed=42
    )

    generator = SyntheticDataGenerator(
        domain=DesignDomain.NAVAL,
        physics_engine=None,
        parameter_template=template,
        config=config
    )

    designs, results = generator.generate()

    stats = generator.get_statistics()
    expected_total = 20 * 3  # base * augmentation_factor

    print(f"  Base samples: 20")
    print(f"  Augmentation factor: 3")
    print(f"  Expected total: {expected_total}")
    print(f"  Actual total: {stats.total_samples}")

    assert stats.total_samples == expected_total, f"Expected {expected_total} samples, got {stats.total_samples}"

    print("  ✓ Augmentation working correctly")
    print("\n✅ Data augmentation working!")


def test_export():
    """Test dataset export."""
    print("\n" + "="*70)
    print("TEST: Dataset Export")
    print("="*70)

    import tempfile
    import shutil
    from pathlib import Path

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        template = MockDesignParameters(
            domain=DesignDomain.NAVAL,
            design_id="template",
            name="Template",
            mass=500.0
        )

        config = DataGenerationConfig(
            n_samples=30,
            sampling_strategy="mixed",
            validate_physics=False,
            enable_augmentation=False,
            output_format="numpy",
            include_metadata=True,
            random_seed=42
        )

        generator = SyntheticDataGenerator(
            domain=DesignDomain.NAVAL,
            physics_engine=None,
            parameter_template=template,
            config=config
        )

        designs, results = generator.generate()
        generator.export(temp_dir, "test_dataset")

        # Check files exist
        files = list(Path(temp_dir).glob("*"))
        print(f"  Generated files: {[f.name for f in files]}")

        params_file = Path(temp_dir) / "test_dataset_parameters.npy"
        metadata_file = Path(temp_dir) / "test_dataset_metadata.json"

        assert params_file.exists(), "Parameters file not found"
        assert metadata_file.exists(), "Metadata file not found"

        # Load and verify
        params_array = np.load(params_file)
        print(f"  Parameters shape: {params_array.shape}")

        assert params_array.shape[0] == 30, f"Expected 30 samples, got {params_array.shape[0]}"
        assert params_array.shape[1] == 4, f"Expected 4 parameters, got {params_array.shape[1]}"

        print("  ✓ Export working correctly")
        print("\n✅ Dataset export working!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_statistics():
    """Test statistics calculation."""
    print("\n" + "="*70)
    print("TEST: Statistics Calculation")
    print("="*70)

    template = MockDesignParameters(
        domain=DesignDomain.NAVAL,
        design_id="template",
        name="Template",
        mass=500.0
    )

    config = DataGenerationConfig(
        n_samples=100,
        sampling_strategy="mixed",
        validate_physics=False,
        enable_augmentation=False,
        random_seed=42
    )

    generator = SyntheticDataGenerator(
        domain=DesignDomain.NAVAL,
        physics_engine=None,
        parameter_template=template,
        config=config
    )

    designs, results = generator.generate()
    stats = generator.get_statistics()

    print(f"  Total samples: {stats.total_samples}")
    print(f"  Diversity score: {stats.diversity_score:.3f}")
    print(f"  Generation time: {stats.generation_time_seconds:.3f}s")

    # Verify diversity score is reasonable
    assert 0.0 <= stats.diversity_score <= 1.0, f"Diversity score out of range: {stats.diversity_score}"

    # Verify parameter coverage
    for param, coverage in stats.parameter_coverage.items():
        print(f"  {param} coverage: {coverage*100:.1f}%")
        assert 0.0 <= coverage <= 1.0, f"Coverage out of range for {param}: {coverage}"

    print("  ✓ Statistics calculation working correctly")
    print("\n✅ Statistics working!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATOR TESTS")
    print("="*70)

    try:
        test_sampling_strategies()
        test_augmentation()
        test_export()
        test_statistics()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70 + "\n")

    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70 + "\n")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
