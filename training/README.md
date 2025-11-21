# Training Data Generation System

Physics-informed synthetic data generation for training machine learning models on multi-domain design optimization.

## Overview

The training data generation system creates high-quality, diverse datasets for ML model training by:
- **Physics-informed sampling**: Respects domain constraints and physical feasibility
- **Multiple sampling strategies**: Latin Hypercube, Gaussian, Edge/Corner, and mixed approaches
- **Data augmentation**: Controlled noise addition for increased dataset size
- **Quality metrics**: Diversity scoring, parameter coverage analysis
- **Multi-domain support**: Works across naval, aerial, ground vehicle, and structural domains

## Quick Start

```python
from training import SyntheticDataGenerator, DataGenerationConfig
from design_core import DesignDomain
from naval_domain import CatamaranParameters, NavalPhysicsEngine

# Configure generation
config = DataGenerationConfig(
    n_samples=1000,
    sampling_strategy="mixed",
    enable_augmentation=True,
    augmentation_factor=2
)

# Create generator
template = CatamaranParameters(
    design_id="template",
    name="Template",
    domain=DesignDomain.NAVAL,
    mass=1000.0,
    length_overall=18.0,
    beam=6.0,
    hull_spacing=4.5,
    hull_depth=2.5
)

physics_engine = NavalPhysicsEngine(use_gpu=False)

generator = SyntheticDataGenerator(
    domain=DesignDomain.NAVAL,
    physics_engine=physics_engine,
    parameter_template=template,
    config=config
)

# Generate dataset
designs, results = generator.generate()

# Export dataset
generator.export("datasets/naval_training", "naval_v1")

# Print summary
generator.print_summary()
```

## Sampling Strategies

### Latin Hypercube Sampling (LHS)
- **Purpose**: Maximum parameter space coverage with minimum samples
- **Best for**: Initial exploration, baseline datasets
- **Characteristics**: Space-filling, low-discrepancy sequence

```python
config = DataGenerationConfig(
    n_samples=500,
    sampling_strategy="latin_hypercube"
)
```

### Gaussian Sampling
- **Purpose**: Focus on high-performing regions
- **Best for**: Exploitation around known good designs
- **Characteristics**: Concentrated around means, tunable exploration radius

```python
config = DataGenerationConfig(
    n_samples=500,
    sampling_strategy="gaussian"
)
```

### Edge/Corner Sampling
- **Purpose**: Explore boundary conditions and extreme cases
- **Best for**: Robustness testing, edge case coverage
- **Characteristics**: Tests parameter limits, boundary interactions

```python
config = DataGenerationConfig(
    n_samples=500,
    sampling_strategy="edge_corner"
)
```

### Mixed Sampling (Recommended)
- **Purpose**: Balanced exploration and exploitation
- **Best for**: General-purpose training datasets
- **Characteristics**: Combines all strategies with configurable weights

```python
config = DataGenerationConfig(
    n_samples=1000,
    sampling_strategy="mixed",
    strategy_weights={
        "latin_hypercube": 0.5,  # 50% LHS
        "gaussian": 0.3,          # 30% Gaussian
        "edge_corner": 0.2        # 20% Edge/Corner
    }
)
```

## Data Augmentation

Increase dataset size with controlled noise while preserving physical validity.

```python
config = DataGenerationConfig(
    n_samples=500,
    enable_augmentation=True,
    augmentation_factor=3,  # 3x the original size
    augmentation_noise_std=0.02  # 2% noise
)
```

**How it works**:
1. Generates base dataset using chosen sampling strategy
2. For each valid design, creates `augmentation_factor - 1` variations
3. Adds Gaussian noise to each parameter (scaled by noise_std)
4. Clips augmented parameters to valid bounds
5. Optionally re-validates augmented designs

**Benefits**:
- Increases dataset size without expensive simulations
- Adds robustness to ML models (noise tolerance)
- Smooth coverage around high-quality designs

## Quality Metrics

### Diversity Score
Measures how well the dataset covers the parameter space.

```python
stats = generator.get_statistics()
print(f"Diversity score: {stats.diversity_score:.3f}")  # Higher = better coverage
```

**Calculation**: Mean of minimum distances between all sample pairs (normalized).

### Parameter Coverage
Percentage of parameter range explored for each dimension.

```python
for param, coverage in stats.parameter_coverage.items():
    print(f"{param}: {coverage*100:.1f}%")
```

**Target**: > 80% coverage for all parameters indicates good exploration.

### Validity Ratio
Percentage of generated designs that are physically valid.

```python
print(f"Valid: {stats.valid_samples}/{stats.total_samples} ({stats.validity_ratio*100:.1f}%)")
```

**Typical values**:
- LHS: 70-90% (good space coverage, some invalid edges)
- Gaussian: 85-95% (focused on valid regions)
- Edge/Corner: 40-70% (intentionally tests boundaries)
- Mixed: 70-85% (balanced)

## Export Formats

### NumPy (Default)
Best for Python ML workflows (scikit-learn, PyTorch, TensorFlow).

```python
generator.export("datasets/naval", dataset_name="naval_v1")
# Creates:
#   datasets/naval/naval_v1_parameters.npy
#   datasets/naval/naval_v1_results.npy
#   datasets/naval/naval_v1_metadata.json
```

**Loading**:
```python
import numpy as np
params = np.load("datasets/naval/naval_v1_parameters.npy")
results = np.load("datasets/naval/naval_v1_results.npy")
```

### CSV
Best for data analysis, visualization, and non-Python tools.

```python
config.output_format = "csv"
generator.export("datasets/naval", dataset_name="naval_v1")
# Creates:
#   datasets/naval/naval_v1_parameters.csv
#   datasets/naval/naval_v1_results.csv
#   datasets/naval/naval_v1_metadata.json
```

### JSON
Best for web applications, APIs, and language-agnostic workflows.

```python
config.output_format = "json"
generator.export("datasets/naval", dataset_name="naval_v1")
# Creates:
#   datasets/naval/naval_v1.json
#   datasets/naval/naval_v1_metadata.json
```

## Use Cases

### 1. Training Surrogate Models
Generate large datasets to train fast approximations of physics simulations.

```python
# Generate 10,000 samples for surrogate training
config = DataGenerationConfig(
    n_samples=5000,
    sampling_strategy="mixed",
    enable_augmentation=True,
    augmentation_factor=2  # = 10,000 total
)

designs, results = generator.generate()
generator.export("datasets/surrogate_training", "naval_surrogate_v1")
```

### 2. Transfer Learning Initialization
Create diverse baseline dataset for new domains.

```python
# Initial dataset for aerial domain
from aerial_domain import AircraftParameters, AerialPhysicsEngine

config = DataGenerationConfig(
    n_samples=1000,
    sampling_strategy="latin_hypercube",  # Maximum coverage
    enable_augmentation=False  # Keep unique samples only
)

generator = SyntheticDataGenerator(
    domain=DesignDomain.AERIAL,
    physics_engine=AerialPhysicsEngine(),
    parameter_template=AircraftParameters(...),
    config=config
)

designs, results = generator.generate()
```

### 3. Data Augmentation for Imbalanced Datasets
Increase samples in under-represented regions.

```python
# Focus on specific parameter region
config = DataGenerationConfig(
    n_samples=200,
    sampling_strategy="gaussian",  # Focused sampling
    enable_augmentation=True,
    augmentation_factor=5  # Heavy augmentation
)

designs, results = generator.generate()
```

### 4. Robustness Testing Datasets
Generate edge cases for testing ML model robustness.

```python
config = DataGenerationConfig(
    n_samples=500,
    sampling_strategy="edge_corner",  # Boundary exploration
    validate_physics=True,  # Ensure validity
    min_valid_ratio=0.5  # Accept lower validity for edge cases
)

designs, results = generator.generate()
generator.export("datasets/robustness_test", "naval_edges_v1")
```

## Configuration Reference

### DataGenerationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | int | 1000 | Number of base samples to generate |
| `sampling_strategy` | str | "mixed" | Strategy: "latin_hypercube", "gaussian", "edge_corner", "mixed" |
| `strategy_weights` | dict | {lhs: 0.5, gauss: 0.3, edge: 0.2} | Weights for mixed sampling |
| `validate_physics` | bool | True | Run physics validation on generated samples |
| `min_valid_ratio` | float | 0.7 | Minimum acceptable ratio of valid designs |
| `enable_augmentation` | bool | True | Enable data augmentation |
| `augmentation_noise_std` | float | 0.02 | Noise level for augmentation (2%) |
| `augmentation_factor` | int | 2 | Multiplier for augmented dataset size |
| `min_diversity_score` | float | 0.6 | Minimum acceptable diversity score |
| `max_correlation` | float | 0.95 | Maximum parameter correlation allowed |
| `output_format` | str | "numpy" | Export format: "numpy", "csv", "json" |
| `include_metadata` | bool | True | Export statistics and config metadata |
| `random_seed` | int | None | Random seed for reproducibility |

## Performance

**Generation Speed** (CPU, naval domain):
- LHS: ~2000 samples/sec (sampling) + physics simulation time
- Gaussian: ~5000 samples/sec (sampling) + physics simulation time
- Edge/Corner: ~3000 samples/sec (sampling) + physics simulation time
- Mixed: ~3000 samples/sec (sampling) + physics simulation time

**Physics Simulation** (bottleneck):
- CPU mode: ~500-1000 designs/sec (naval), ~300-500 designs/sec (aerial)
- GPU mode: ~5000-10000 designs/sec (batch processing)

**Tips for performance**:
1. Use GPU mode for physics simulation when available
2. Disable physics validation for initial sampling experiments
3. Use augmentation to increase dataset size without additional simulations
4. Generate base dataset once, then augment multiple times with different noise levels

## Examples

See `examples/` directory for complete examples:
- `generate_naval_dataset.py`: Naval vessel training data
- `generate_aerial_dataset.py`: Aircraft training data
- `augmentation_example.py`: Data augmentation techniques
- `quality_metrics_demo.py`: Dataset quality analysis

## Integration with MAGNET

The synthetic data generator integrates with:
- **Knowledge Base**: Can use historical experiment results as priors for Gaussian sampling
- **LLM Agents**: Provides training data for ML-guided hypothesis generation
- **Physics Engines**: Uses domain-specific validation and simulation
- **Multi-Domain System**: Works across all supported design domains

## Future Enhancements

Planned features for future versions:
- **Active Learning**: Prioritize samples with high uncertainty
- **Importance Sampling**: Focus on high-performance regions
- **Constrained Sampling**: Enforce multi-parameter constraints
- **Pareto-aware Sampling**: Generate balanced multi-objective datasets
- **Historical Priors**: Use experiment database to guide sampling

## Testing

Run tests for synthetic data generation:

```bash
pytest tests/training/ -v
```

Test specific functionality:

```bash
pytest tests/training/test_sampling_strategies.py -v
pytest tests/training/test_augmentation.py -v
pytest tests/training/test_quality_metrics.py -v
```

## Troubleshooting

### Low Validity Ratio
**Problem**: < 50% of samples are physically valid

**Solutions**:
1. Check parameter constraints are reasonable
2. Use Gaussian sampling instead of Edge/Corner
3. Reduce augmentation noise level
4. Review physics engine validation logic

### Low Diversity Score
**Problem**: Samples clustered in small region of parameter space

**Solutions**:
1. Use Latin Hypercube or mixed sampling
2. Increase number of base samples
3. Reduce Gaussian concentration (increase std)
4. Disable augmentation (it reduces diversity)

### Generation Too Slow
**Problem**: Taking too long to generate dataset

**Solutions**:
1. Enable GPU mode for physics engine
2. Reduce number of samples
3. Disable physics validation for sampling only
4. Use batch simulation instead of per-design simulation

## References

- Latin Hypercube Sampling: McKay et al. (1979)
- Quality Diversity: Pugh et al. (2016)
- Data Augmentation for Physics-Informed Learning: Raissi et al. (2019)
