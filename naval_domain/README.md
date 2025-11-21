# MAGNET Naval Domain Physics

High-fidelity twin-hull (catamaran) physics simulation for autonomous design research.

## Overview

The `naval_domain` module provides physics-based simulation of catamaran hull designs using industry-standard naval architecture formulas.

### Key Components

1. **PhysicsEngine** - Single-design CPU simulation
2. **ParallelPhysicsEngine** - GPU-accelerated batch processing
3. **HullParameters** - Validated design parameter schema
4. **baseline_designs** - Curated starting designs
5. **hull_generator** - Procedural design generation

---

## Physics Calculations

### Core Formulas

The physics engine implements:

1. **Displacement & Buoyancy** (Archimedes Principle)
   - Volumetric displacement calculation
   - Prismatic and block coefficients
   - Draft and freeboard relationships

2. **Metacentric Height (GM)** - Stability Analysis
   - GM = (I/∇) - BG
   - Waterline moment of inertia
   - Center of buoyancy calculation

3. **Resistance** (ITTC-1957 + Residuary)
   - Frictional resistance (ITTC-1957 model line)
   - Residuary resistance (Froude-based)
   - Form factor for twin hulls

4. **Power Requirements**
   - Effective horsepower (EHP)
   - Shaft horsepower (SHP) with efficiency
   - Speed-power curves

### Scoring System

Designs evaluated on three objectives (each 0-100):

1. **Stability Score** (40% weight)
   - Based on GM (metacentric height)
   - Higher GM = better stability
   - Threshold: GM > 1.0m for good score

2. **Speed Score** (30% weight)
   - Based on Froude number and speed achievement
   - Efficiency at target speed
   - Balance between speed and power

3. **Efficiency Score** (30% weight)
   - Power-to-weight ratio
   - Resistance-to-displacement
   - Hydrodynamic efficiency

**Overall Score** = 0.4×Stability + 0.3×Speed + 0.3×Efficiency

---

## HullParameters

### Schema

```python
from naval_domain.hull_parameters import HullParameters

hp = HullParameters(
    # Primary dimensions (meters)
    length_overall=18.0,
    beam=2.0,
    hull_depth=2.2,
    hull_spacing=5.4,

    # Hull form
    deadrise_angle=12.0,
    freeboard=1.4,
    lcb_position=48.0,  # % from stern

    # Coefficients
    prismatic_coefficient=0.60,
    waterline_beam=1.8,
    block_coefficient=0.42,

    # Performance
    design_speed=25.0,  # knots
    displacement=35.0,  # tonnes
    draft=0.8           # meters (optional, auto-calculated if None)
)
```

### Parameter Ranges

| Parameter | Min | Max | Unit | Description |
|-----------|-----|-----|------|-------------|
| length_overall | 8.0 | 50.0 | m | Overall length |
| beam | 1.0 | 5.0 | m | Single hull beam |
| hull_depth | 1.0 | 8.0 | m | Hull depth |
| hull_spacing | 2.0 | 20.0 | m | Center-to-center spacing |
| deadrise_angle | 0.0 | 30.0 | deg | V-angle of hull |
| freeboard | 0.5 | 4.0 | m | Deck to waterline |
| lcb_position | 40.0 | 60.0 | % | Longitudinal center of buoyancy |
| prismatic_coefficient | 0.50 | 0.75 | - | Volume distribution |
| waterline_beam | 0.8 | 3.5 | m | Beam at waterline |
| block_coefficient | 0.30 | 0.55 | - | Volume vs box |
| design_speed | 10.0 | 50.0 | kts | Target speed |
| displacement | 5.0 | 500.0 | t | Total weight |
| draft | 0.3 | 5.0 | m | Submerged depth |

### Validation

All parameters validated on construction:

```python
try:
    hp = HullParameters(length_overall=5.0)  # Too small
except ValueError as e:
    print(e)  # "length_overall must be between 8.0 and 50.0"
```

---

## PhysicsEngine

### Basic Usage

```python
from naval_domain.physics_engine import PhysicsEngine
from naval_domain.hull_parameters import HullParameters

# Initialize engine
physics = PhysicsEngine(verbose=True)

# Create design
hp = HullParameters(
    length_overall=18.0,
    beam=2.0,
    # ... other parameters
)

# Simulate
result = physics.simulate(hp)

# Access results
print(f"Overall Score: {result.overall_score:.1f}/100")
print(f"Stability: {result.stability_score:.1f}")
print(f"Speed: {result.speed_score:.1f}")
print(f"Efficiency: {result.efficiency_score:.1f}")
print(f"GM: {result.metacentric_height:.2f}m")
print(f"Total Resistance: {result.total_resistance:.1f}kN")
```

### Performance

**CPU (Mac M1):**
- Single design: ~1ms
- Sequential 100 designs: ~100ms
- Throughput: ~1000 designs/sec

**GPU (2x A40, batch=100):**
- Parallel: ~25ms
- Throughput: ~4000 designs/sec

---

## Baseline Designs

### Available Baselines

```python
from naval_domain.baseline_designs import (
    get_baseline_general,
    get_baseline_high_speed,
    get_baseline_stability,
    get_baseline_efficiency,
    get_baseline_compact,
    get_baseline_large,
    get_all_baselines
)

# Get specific baseline
general = get_baseline_general()

# Get all 6 baselines
all_designs = get_all_baselines()
```

### Baseline Specifications

| Name | LOA | Beam | Spacing | Speed | Displacement | Focus |
|------|-----|------|---------|-------|--------------|-------|
| General | 18m | 2.0m | 5.4m | 25kt | 35t | Balanced |
| High-Speed | 22m | 2.2m | 6.0m | 35kt | 45t | Speed |
| Stability | 16m | 1.9m | 7.0m | 22kt | 30t | Stability |
| Efficiency | 20m | 2.1m | 5.8m | 28kt | 40t | Efficiency |
| Compact | 12m | 1.7m | 4.5m | 20kt | 20t | Harbor ops |
| Large | 30m | 2.5m | 8.0m | 30kt | 80t | Extended ops |

---

## Testing

### Unit Tests

```bash
# Run all naval physics tests
pytest tests/naval/test_physics_engine.py -v

# Expected: 21/21 tests passing
```

---

## See Also

- [Memory Systems Documentation](../memory/README.md)
- [Orchestration Guide](../orchestration/README.md)
- [Agent System](../agents/README.md)

---

## References

1. ITTC Recommended Procedures and Guidelines (7.5-02-02-01)
2. Principles of Naval Architecture (SNAME, 1988)
3. "High Speed Small Craft" - Peter Du Cane
4. "Fast Ferry Design" - Alistair Greig
