"""
Training Data Generation Package
=================================

Provides synthetic data generation capabilities for training machine learning models.

Key modules:
- synthetic_data_generator: Physics-informed data generation with multiple sampling strategies
"""

from training.synthetic_data_generator import (
    SyntheticDataGenerator,
    DataGenerationConfig,
    DatasetStatistics
)

__all__ = [
    'SyntheticDataGenerator',
    'DataGenerationConfig',
    'DatasetStatistics',
]
