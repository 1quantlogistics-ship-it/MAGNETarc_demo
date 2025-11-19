"""
Augmentation Policy Schema for ARC.

Defines safe augmentation operation space for glaucoma detection model training.
Enables structured augmentation policy search while maintaining clinical safety
(DRI constraint preservation).

Key Components:
- Safe operation whitelist (geometric, intensity, noise transforms)
- Forbidden operations that risk DRI corruption
- Policy composition and evolution strategies
- Magnitude bounds for clinical appropriateness

Clinical Safety Constraints:
- DRI (Disc Relevance Index) must remain ≥ 0.6 after augmentation
- No operations that corrupt optic disc structure (cutout, strong elastic)
- No color space changes (hue/saturation shifts)
- Conservative magnitude bounds for all transforms

Author: ARC Team (Dev 1)
Created: 2025-11-18
Version: 1.0
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, validator


class AugmentationOpType(str, Enum):
    """
    Safe augmentation operation types for glaucoma fundus images.

    **Geometric Transforms** (Safe - preserve optic disc structure):
    - ROTATE: Rotation within ±15° (fundus images can vary in orientation)
    - HORIZONTAL_FLIP: Mirror flip (safe for fundus)
    - VERTICAL_FLIP: Vertical mirror (use with caution)
    - SCALE: Zoom/scale (0.9-1.1 range)
    - TRANSLATE: Shift image (small offsets only)

    **Intensity Transforms** (Careful - preserve contrast):
    - BRIGHTNESS: Adjust brightness (±10% safe range)
    - CONTRAST: Adjust contrast (±10% safe range)
    - GAMMA: Gamma correction (0.9-1.1 range)
    - EQUALIZE_HIST: Histogram equalization (adaptive)

    **Noise/Blur Transforms** (Careful - low magnitude only):
    - GAUSSIAN_NOISE: Add Gaussian noise (low σ ≤ 0.01)
    - GAUSSIAN_BLUR: Gaussian blur (small kernel ≤ 3)

    **Forbidden Operations** (DO NOT USE):
    - Color jitter (hue/saturation) - corrupts fundus color space
    - Cutout / random erasing - removes critical disc structure
    - Elastic deformation (strong) - distorts optic disc shape
    - CLAHE with aggressive parameters - over-enhances artifacts
    """
    # Geometric (safe)
    ROTATE = "rotate"
    HORIZONTAL_FLIP = "horizontal_flip"
    VERTICAL_FLIP = "vertical_flip"
    SCALE = "scale"
    TRANSLATE = "translate"

    # Intensity (careful)
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    GAMMA = "gamma"
    EQUALIZE_HIST = "equalize_hist"

    # Noise/Blur (careful)
    GAUSSIAN_NOISE = "gaussian_noise"
    GAUSSIAN_BLUR = "gaussian_blur"


class PolicyEvolutionStrategy(str, Enum):
    """
    Strategy for evolving augmentation policies during ARC research cycles.

    - RANDOM: Random sampling from operation space (baseline exploration)
    - GRADIENT: Gradient-based optimization using DRI feedback
    - RL: Reinforcement learning (policy gradient methods)
    - EVOLUTIONARY: Genetic algorithm (mutation + crossover)
    - BAYESIAN: Bayesian optimization over policy space
    """
    RANDOM = "random"
    GRADIENT = "gradient"
    RL = "rl"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"


class AugmentationOp(BaseModel):
    """
    Single augmentation operation with magnitude and probability.

    Example:
        {
            "op_type": "rotate",
            "magnitude": 10.0,  # Degrees
            "probability": 0.5
        }
    """
    op_type: AugmentationOpType = Field(
        description="Type of augmentation operation"
    )

    magnitude: float = Field(
        description="Operation magnitude (interpretation depends on op_type)"
    )

    probability: float = Field(
        ge=0.0, le=1.0,
        description="Probability of applying this operation (0.0 to 1.0)"
    )

    @validator('magnitude')
    def validate_magnitude_bounds(cls, v, values):
        """
        Validate magnitude is within safe clinical bounds for each operation.

        Bounds designed to preserve DRI ≥ 0.6 constraint.
        """
        op_type = values.get('op_type')

        # Define safe magnitude bounds
        bounds = {
            # Geometric
            AugmentationOpType.ROTATE: (-15.0, 15.0),  # Degrees
            AugmentationOpType.HORIZONTAL_FLIP: (0.0, 1.0),  # Binary (0 or 1)
            AugmentationOpType.VERTICAL_FLIP: (0.0, 1.0),  # Binary
            AugmentationOpType.SCALE: (0.9, 1.1),  # Scale factor
            AugmentationOpType.TRANSLATE: (-0.1, 0.1),  # Fraction of image size

            # Intensity
            AugmentationOpType.BRIGHTNESS: (-0.1, 0.1),  # ±10%
            AugmentationOpType.CONTRAST: (-0.1, 0.1),  # ±10%
            AugmentationOpType.GAMMA: (0.9, 1.1),  # Gamma correction factor
            AugmentationOpType.EQUALIZE_HIST: (0.0, 1.0),  # Blend factor

            # Noise/Blur
            AugmentationOpType.GAUSSIAN_NOISE: (0.0, 0.01),  # Noise σ
            AugmentationOpType.GAUSSIAN_BLUR: (0.0, 3.0),  # Kernel size
        }

        if op_type in bounds:
            min_val, max_val = bounds[op_type]
            if not (min_val <= v <= max_val):
                raise ValueError(
                    f"{op_type} magnitude {v} outside safe bounds [{min_val}, {max_val}]"
                )

        return v


class AugmentationPolicy(BaseModel):
    """
    Complete augmentation policy as sequence of operations.

    An augmentation policy defines a pipeline of augmentation operations
    to be applied during training. Each operation is applied with a certain
    probability and magnitude.

    Clinical Safety:
    - All operations must be from safe whitelist
    - Magnitudes constrained to preserve DRI ≥ 0.6
    - No forbidden operations allowed

    Example:
        {
            "name": "mild_geometric",
            "operations": [
                {"op_type": "rotate", "magnitude": 10.0, "probability": 0.5},
                {"op_type": "horizontal_flip", "magnitude": 1.0, "probability": 0.5},
                {"op_type": "brightness", "magnitude": 0.05, "probability": 0.3}
            ],
            "evolution_strategy": "evolutionary"
        }
    """
    name: str = Field(
        description="Human-readable policy name (e.g., 'mild_geometric', 'intensity_boost')"
    )

    operations: List[AugmentationOp] = Field(
        min_items=1, max_items=10,
        description="Sequence of augmentation operations (1-10 ops)"
    )

    evolution_strategy: PolicyEvolutionStrategy = Field(
        default=PolicyEvolutionStrategy.RANDOM,
        description="Strategy used to evolve this policy"
    )

    dri_constraint: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum DRI threshold policy must maintain (default 0.6)"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata (parent policies, generation, fitness, etc.)"
    )

    @validator('operations')
    def validate_no_forbidden_ops(cls, v):
        """
        Ensure no forbidden operations are present.

        This is a safety check - forbidden ops should not be in AugmentationOpType,
        but we double-check here for defense-in-depth.
        """
        forbidden_keywords = ["cutout", "erasing", "elastic", "color_jitter", "hue", "saturation"]

        for op in v:
            op_name = op.op_type.value.lower()
            if any(keyword in op_name for keyword in forbidden_keywords):
                raise ValueError(
                    f"Forbidden operation detected: {op.op_type}. "
                    f"This operation risks corrupting DRI."
                )

        return v

    @validator('operations')
    def validate_policy_length(cls, v):
        """
        Validate policy is not too complex (overfitting risk).

        Policies with >10 operations risk overfitting and are hard to interpret.
        """
        if len(v) > 10:
            raise ValueError(
                f"Policy has {len(v)} operations (max 10). "
                f"Overly complex policies risk overfitting."
            )

        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary for serialization."""
        return self.dict()

    @classmethod
    def baseline_policy(cls) -> "AugmentationPolicy":
        """
        Create baseline augmentation policy (current production).

        Returns:
            Conservative policy with basic geometric transforms
        """
        return cls(
            name="baseline_geometric",
            operations=[
                AugmentationOp(
                    op_type=AugmentationOpType.ROTATE,
                    magnitude=10.0,
                    probability=0.5
                ),
                AugmentationOp(
                    op_type=AugmentationOpType.HORIZONTAL_FLIP,
                    magnitude=1.0,
                    probability=0.5
                )
            ],
            evolution_strategy=PolicyEvolutionStrategy.RANDOM
        )

    @classmethod
    def example_intensity_policy(cls) -> "AugmentationPolicy":
        """Example: Intensity augmentation policy."""
        return cls(
            name="intensity_boost",
            operations=[
                AugmentationOp(
                    op_type=AugmentationOpType.BRIGHTNESS,
                    magnitude=0.08,
                    probability=0.4
                ),
                AugmentationOp(
                    op_type=AugmentationOpType.CONTRAST,
                    magnitude=0.08,
                    probability=0.4
                ),
                AugmentationOp(
                    op_type=AugmentationOpType.GAMMA,
                    magnitude=1.05,
                    probability=0.3
                )
            ],
            evolution_strategy=PolicyEvolutionStrategy.EVOLUTIONARY
        )

    @classmethod
    def example_comprehensive_policy(cls) -> "AugmentationPolicy":
        """Example: Comprehensive policy combining multiple transform types."""
        return cls(
            name="comprehensive_v1",
            operations=[
                # Geometric
                AugmentationOp(
                    op_type=AugmentationOpType.ROTATE,
                    magnitude=12.0,
                    probability=0.5
                ),
                AugmentationOp(
                    op_type=AugmentationOpType.HORIZONTAL_FLIP,
                    magnitude=1.0,
                    probability=0.5
                ),
                AugmentationOp(
                    op_type=AugmentationOpType.SCALE,
                    magnitude=1.05,
                    probability=0.3
                ),
                # Intensity
                AugmentationOp(
                    op_type=AugmentationOpType.BRIGHTNESS,
                    magnitude=0.06,
                    probability=0.3
                ),
                AugmentationOp(
                    op_type=AugmentationOpType.CONTRAST,
                    magnitude=0.06,
                    probability=0.3
                ),
                # Noise (subtle)
                AugmentationOp(
                    op_type=AugmentationOpType.GAUSSIAN_NOISE,
                    magnitude=0.005,
                    probability=0.2
                )
            ],
            evolution_strategy=PolicyEvolutionStrategy.BAYESIAN,
            metadata={
                "generation": 1,
                "parent_policies": ["baseline_geometric", "intensity_boost"]
            }
        )


def validate_policy_safety(policy: AugmentationPolicy) -> Tuple[bool, str]:
    """
    Validate augmentation policy for clinical safety.

    Additional validation beyond Pydantic model validators.
    Used by Critic agent for policy proposal review.

    Args:
        policy: Augmentation policy to validate

    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: No operations with excessive probability (risks overfitting)
    high_prob_ops = [op for op in policy.operations if op.probability > 0.9]
    if len(high_prob_ops) > 2:
        return False, (
            f"Too many high-probability operations ({len(high_prob_ops)}). "
            f"Risks overfitting and reduced diversity."
        )

    # Check 2: Rotation magnitudes should not exceed ±15°
    rotate_ops = [
        op for op in policy.operations
        if op.op_type == AugmentationOpType.ROTATE
    ]
    for op in rotate_ops:
        if abs(op.magnitude) > 15.0:
            return False, (
                f"Rotation magnitude {op.magnitude}° exceeds safe limit (±15°). "
                f"Large rotations risk DRI corruption."
            )

    # Check 3: Combined intensity adjustments should not be too aggressive
    intensity_ops = [
        op for op in policy.operations
        if op.op_type in [
            AugmentationOpType.BRIGHTNESS,
            AugmentationOpType.CONTRAST,
            AugmentationOpType.GAMMA
        ]
    ]

    if len(intensity_ops) > 3:
        # Check cumulative intensity change
        total_intensity_change = sum(
            abs(op.magnitude) * op.probability
            for op in intensity_ops
        )

        if total_intensity_change > 0.3:
            return False, (
                f"Cumulative intensity change ({total_intensity_change:.2f}) too high. "
                f"Risks over-processing fundus images."
            )

    # Check 4: Gaussian noise should be very conservative
    noise_ops = [
        op for op in policy.operations
        if op.op_type == AugmentationOpType.GAUSSIAN_NOISE
    ]

    for op in noise_ops:
        if op.magnitude > 0.01:
            return False, (
                f"Gaussian noise σ={op.magnitude} too high (max 0.01). "
                f"High noise corrupts fine optic disc details."
            )

    # Check 5: Policy should not be empty or trivial
    active_ops = [op for op in policy.operations if op.probability > 0.0]
    if len(active_ops) == 0:
        return False, "Policy has no active operations (all probabilities = 0)"

    # All checks passed
    return True, ""


def mutate_policy(
    policy: AugmentationPolicy,
    mutation_rate: float = 0.3,
    magnitude_delta: float = 0.1
) -> AugmentationPolicy:
    """
    Mutate augmentation policy for evolutionary search.

    Applies random mutations to policy operations while maintaining
    clinical safety constraints.

    Mutation types:
    - Adjust operation magnitude (±magnitude_delta)
    - Adjust operation probability (±0.1)
    - Add new operation (if policy < 10 ops)
    - Remove operation (if policy > 1 op)

    Args:
        policy: Policy to mutate
        mutation_rate: Probability of mutating each component
        magnitude_delta: Maximum change in magnitude

    Returns:
        Mutated policy (new instance)
    """
    import random
    import copy

    mutated = copy.deepcopy(policy)
    mutated.name = f"{policy.name}_mutated"

    # Update metadata
    if mutated.metadata is None:
        mutated.metadata = {}
    mutated.metadata["parent"] = policy.name
    mutated.metadata["mutation_rate"] = mutation_rate

    # Mutate existing operations
    for op in mutated.operations:
        if random.random() < mutation_rate:
            # Mutate magnitude
            if random.random() < 0.5:
                delta = random.uniform(-magnitude_delta, magnitude_delta)

                # Get bounds for this operation
                if op.op_type == AugmentationOpType.ROTATE:
                    bounds = (-15.0, 15.0)
                elif op.op_type in [AugmentationOpType.BRIGHTNESS, AugmentationOpType.CONTRAST]:
                    bounds = (-0.1, 0.1)
                elif op.op_type == AugmentationOpType.GAMMA:
                    bounds = (0.9, 1.1)
                elif op.op_type == AugmentationOpType.SCALE:
                    bounds = (0.9, 1.1)
                else:
                    bounds = None

                if bounds:
                    new_magnitude = max(bounds[0], min(bounds[1], op.magnitude + delta))
                    op.magnitude = new_magnitude

            # Mutate probability
            else:
                delta = random.uniform(-0.1, 0.1)
                new_prob = max(0.0, min(1.0, op.probability + delta))
                op.probability = new_prob

    # Maybe add new operation
    if len(mutated.operations) < 10 and random.random() < mutation_rate:
        available_ops = list(AugmentationOpType)
        new_op_type = random.choice(available_ops)

        # Generate safe magnitude
        if new_op_type == AugmentationOpType.ROTATE:
            magnitude = random.uniform(-10.0, 10.0)
        elif new_op_type in [AugmentationOpType.BRIGHTNESS, AugmentationOpType.CONTRAST]:
            magnitude = random.uniform(-0.08, 0.08)
        elif new_op_type == AugmentationOpType.GAMMA:
            magnitude = random.uniform(0.95, 1.05)
        elif new_op_type == AugmentationOpType.SCALE:
            magnitude = random.uniform(0.95, 1.05)
        elif new_op_type == AugmentationOpType.GAUSSIAN_NOISE:
            magnitude = random.uniform(0.001, 0.008)
        else:
            magnitude = random.uniform(0.0, 1.0)

        new_op = AugmentationOp(
            op_type=new_op_type,
            magnitude=magnitude,
            probability=random.uniform(0.2, 0.6)
        )
        mutated.operations.append(new_op)

    # Maybe remove operation
    if len(mutated.operations) > 1 and random.random() < mutation_rate * 0.5:
        mutated.operations.pop(random.randint(0, len(mutated.operations) - 1))

    return mutated


def crossover_policies(
    policy1: AugmentationPolicy,
    policy2: AugmentationPolicy
) -> AugmentationPolicy:
    """
    Crossover two augmentation policies for evolutionary search.

    Creates offspring policy by combining operations from two parent policies.

    Args:
        policy1: First parent policy
        policy2: Second parent policy

    Returns:
        Offspring policy combining both parents
    """
    import random

    # Randomly select operations from both parents
    all_ops = policy1.operations + policy2.operations

    # Sample operations (up to 10)
    num_ops = min(10, random.randint(len(policy1.operations), len(all_ops)))
    selected_ops = random.sample(all_ops, num_ops)

    offspring = AugmentationPolicy(
        name=f"crossover_{policy1.name}_{policy2.name}",
        operations=selected_ops,
        evolution_strategy=PolicyEvolutionStrategy.EVOLUTIONARY,
        metadata={
            "parents": [policy1.name, policy2.name],
            "crossover_method": "uniform_selection"
        }
    )

    return offspring
