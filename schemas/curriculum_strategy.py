"""
Curriculum Learning Strategy Schema for ARC.

Defines curriculum learning strategies for progressive training on glaucoma detection.
Enables structured curriculum design that gradually increases task difficulty while
maintaining clinical safety.

Key Components:
- Difficulty metrics (image quality, disc visibility, disease severity)
- Curriculum stages with progression criteria
- Pacing strategies (linear, exponential, adaptive)
- Safety constraints for clinical appropriateness

Clinical Considerations:
- Start with clear, high-quality images (easy cases)
- Progress to challenging cases (poor quality, early-stage glaucoma)
- Maintain sensitivity ≥ 0.85 throughout curriculum
- Adaptive pacing based on model performance

Author: ARC Team (Dev 1)
Created: 2025-11-18
Version: 1.0
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class DifficultyMetric(str, Enum):
    """
    Metrics for assessing sample difficulty in glaucoma detection.

    **Image Quality Metrics**:
    - IMAGE_QUALITY: Overall image quality score (clarity, focus, illumination)
    - DISC_VISIBILITY: Optic disc visibility/clarity score

    **Clinical Difficulty Metrics**:
    - DISEASE_SEVERITY: Glaucoma severity (mild → severe)
    - CDR_RATIO: Cup-to-Disc Ratio (low CDR easier than high)
    - PREDICTION_CONFIDENCE: Model confidence (low conf = hard sample)

    **Combined Metrics**:
    - COMPOSITE: Weighted combination of multiple metrics
    """
    # Image quality
    IMAGE_QUALITY = "image_quality"
    DISC_VISIBILITY = "disc_visibility"

    # Clinical difficulty
    DISEASE_SEVERITY = "disease_severity"
    CDR_RATIO = "cdr_ratio"
    PREDICTION_CONFIDENCE = "prediction_confidence"

    # Combined
    COMPOSITE = "composite"


class PacingStrategy(str, Enum):
    """
    Strategy for curriculum pacing (how fast to progress through stages).

    - LINEAR: Fixed progression rate (e.g., every N epochs)
    - EXPONENTIAL: Accelerating progression (slow start, fast later)
    - ROOT: Decelerating progression (fast start, slow later)
    - ADAPTIVE: Performance-based (progress when validation AUC > threshold)
    - STEP: Discrete jumps at predefined epochs
    """
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    ROOT = "root"
    ADAPTIVE = "adaptive"
    STEP = "step"


class CurriculumStage(BaseModel):
    """
    Single stage in curriculum learning progression.

    Example:
        {
            "stage_id": 1,
            "name": "easy_cases",
            "difficulty_range": [0.0, 0.3],
            "min_epochs": 5,
            "progression_criterion": {"validation_auc": 0.80}
        }
    """
    stage_id: int = Field(
        ge=0,
        description="Stage identifier (0=easiest, increasing with difficulty)"
    )

    name: str = Field(
        description="Human-readable stage name (e.g., 'easy_cases', 'moderate_difficulty')"
    )

    difficulty_range: tuple[float, float] = Field(
        description="Range of difficulty scores for this stage [min, max] (0.0 to 1.0)"
    )

    min_epochs: int = Field(
        default=1, ge=1, le=50,
        description="Minimum epochs to spend in this stage"
    )

    max_epochs: Optional[int] = Field(
        default=None, ge=1, le=100,
        description="Maximum epochs for this stage (None=no limit)"
    )

    progression_criterion: Optional[Dict[str, float]] = Field(
        default=None,
        description="Criteria to progress to next stage (e.g., {'validation_auc': 0.80})"
    )

    sample_weight: float = Field(
        default=1.0, ge=0.1, le=10.0,
        description="Sample weight multiplier for this stage (higher=more emphasis)"
    )

    @validator('difficulty_range')
    def validate_difficulty_range(cls, v):
        """Validate difficulty range is valid."""
        if not (0.0 <= v[0] < v[1] <= 1.0):
            raise ValueError(
                f"Difficulty range {v} must satisfy: 0 ≤ min < max ≤ 1"
            )
        return v


class CurriculumStrategy(BaseModel):
    """
    Complete curriculum learning strategy.

    Defines a multi-stage curriculum that progressively trains on harder samples.

    Clinical Safety:
    - Must include all difficulty levels (0.0 to 1.0) across stages
    - Cannot skip difficulty ranges (progressive increase required)
    - Sensitivity must remain ≥ 0.85 throughout

    Example:
        {
            "name": "quality_based_curriculum",
            "difficulty_metric": "image_quality",
            "stages": [
                {"stage_id": 0, "name": "high_quality", "difficulty_range": [0.0, 0.3], ...},
                {"stage_id": 1, "name": "medium_quality", "difficulty_range": [0.3, 0.7], ...},
                {"stage_id": 2, "name": "low_quality", "difficulty_range": [0.7, 1.0], ...}
            ],
            "pacing_strategy": "adaptive"
        }
    """
    name: str = Field(
        description="Human-readable curriculum name (e.g., 'quality_based', 'severity_progressive')"
    )

    difficulty_metric: DifficultyMetric = Field(
        description="Metric used to assess sample difficulty"
    )

    stages: List[CurriculumStage] = Field(
        min_items=2, max_items=10,
        description="Ordered curriculum stages (2-10 stages)"
    )

    pacing_strategy: PacingStrategy = Field(
        description="Strategy for progressing through curriculum stages"
    )

    pacing_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parameters for pacing strategy (e.g., learning_rate, threshold)"
    )

    warmup_epochs: int = Field(
        default=0, ge=0, le=20,
        description="Number of warmup epochs with all data before curriculum starts"
    )

    final_mix_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Ratio of hard samples to mix in at all stages (0=pure curriculum, 1=all data)"
    )

    min_sensitivity_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Minimum sensitivity to maintain throughout curriculum"
    )

    @validator('stages')
    def validate_stage_ordering(cls, v):
        """Validate stages are properly ordered by difficulty."""
        # Check stage IDs are sequential
        stage_ids = [stage.stage_id for stage in v]
        expected_ids = list(range(len(v)))

        if stage_ids != expected_ids:
            raise ValueError(
                f"Stage IDs must be sequential starting from 0. Got: {stage_ids}, Expected: {expected_ids}"
            )

        # Check difficulty ranges are progressive (non-overlapping)
        for i in range(len(v) - 1):
            curr_max = v[i].difficulty_range[1]
            next_min = v[i+1].difficulty_range[0]

            if curr_max != next_min:
                raise ValueError(
                    f"Difficulty ranges must be contiguous. "
                    f"Stage {i} ends at {curr_max}, stage {i+1} starts at {next_min}"
                )

        # Check first stage starts at 0.0
        if v[0].difficulty_range[0] != 0.0:
            raise ValueError(
                f"First stage must start at difficulty 0.0, got {v[0].difficulty_range[0]}"
            )

        # Check last stage ends at 1.0
        if v[-1].difficulty_range[1] != 1.0:
            raise ValueError(
                f"Last stage must end at difficulty 1.0, got {v[-1].difficulty_range[1]}"
            )

        return v

    @validator('pacing_params')
    def validate_pacing_params(cls, v, values):
        """Validate pacing params match pacing strategy."""
        pacing_strategy = values.get('pacing_strategy')

        if pacing_strategy == PacingStrategy.ADAPTIVE:
            if v is None or 'performance_threshold' not in v:
                raise ValueError(
                    "Adaptive pacing requires 'performance_threshold' in pacing_params"
                )

        if pacing_strategy == PacingStrategy.STEP:
            if v is None or 'step_epochs' not in v:
                raise ValueError(
                    "Step pacing requires 'step_epochs' list in pacing_params"
                )

        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict()

    @classmethod
    def baseline_no_curriculum(cls) -> "CurriculumStrategy":
        """
        Create baseline (no curriculum - all data from start).

        Returns:
            Single-stage curriculum with all difficulty levels
        """
        return cls(
            name="baseline_no_curriculum",
            difficulty_metric=DifficultyMetric.IMAGE_QUALITY,
            stages=[
                CurriculumStage(
                    stage_id=0,
                    name="all_data",
                    difficulty_range=(0.0, 1.0),
                    min_epochs=1
                )
            ],
            pacing_strategy=PacingStrategy.LINEAR
        )

    @classmethod
    def example_quality_based(cls) -> "CurriculumStrategy":
        """Example: Image quality-based curriculum (3 stages)."""
        return cls(
            name="quality_based_3stage",
            difficulty_metric=DifficultyMetric.IMAGE_QUALITY,
            stages=[
                CurriculumStage(
                    stage_id=0,
                    name="high_quality",
                    difficulty_range=(0.0, 0.33),
                    min_epochs=5,
                    progression_criterion={"validation_auc": 0.75}
                ),
                CurriculumStage(
                    stage_id=1,
                    name="medium_quality",
                    difficulty_range=(0.33, 0.67),
                    min_epochs=5,
                    progression_criterion={"validation_auc": 0.80}
                ),
                CurriculumStage(
                    stage_id=2,
                    name="low_quality",
                    difficulty_range=(0.67, 1.0),
                    min_epochs=10
                )
            ],
            pacing_strategy=PacingStrategy.ADAPTIVE,
            pacing_params={"performance_threshold": 0.75},
            warmup_epochs=2,
            final_mix_ratio=0.1
        )

    @classmethod
    def example_severity_based(cls) -> "CurriculumStrategy":
        """Example: Disease severity-based curriculum (easy → hard)."""
        return cls(
            name="severity_progressive",
            difficulty_metric=DifficultyMetric.DISEASE_SEVERITY,
            stages=[
                CurriculumStage(
                    stage_id=0,
                    name="severe_cases",
                    difficulty_range=(0.0, 0.25),
                    min_epochs=3,
                    sample_weight=1.0
                ),
                CurriculumStage(
                    stage_id=1,
                    name="moderate_cases",
                    difficulty_range=(0.25, 0.5),
                    min_epochs=5,
                    sample_weight=1.2
                ),
                CurriculumStage(
                    stage_id=2,
                    name="mild_cases",
                    difficulty_range=(0.5, 0.75),
                    min_epochs=5,
                    sample_weight=1.5
                ),
                CurriculumStage(
                    stage_id=3,
                    name="early_stage",
                    difficulty_range=(0.75, 1.0),
                    min_epochs=10,
                    sample_weight=2.0
                )
            ],
            pacing_strategy=PacingStrategy.LINEAR,
            pacing_params={"epochs_per_stage": 5},
            final_mix_ratio=0.2
        )


def validate_curriculum_safety(curriculum: CurriculumStrategy) -> tuple[bool, str]:
    """
    Validate curriculum strategy for clinical safety.

    Additional validation beyond Pydantic model validators.
    Used by Critic agent for curriculum proposal review.

    Args:
        curriculum: Curriculum strategy to validate

    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: Curriculum should not have too many stages (complexity)
    if len(curriculum.stages) > 8:
        return False, (
            f"Too many curriculum stages ({len(curriculum.stages)}). "
            f"Max 8 recommended to avoid over-complication."
        )

    # Check 2: Each stage should have reasonable min_epochs
    for stage in curriculum.stages:
        if stage.min_epochs < 2:
            return False, (
                f"Stage '{stage.name}' has min_epochs={stage.min_epochs} (too low). "
                f"Minimum 2 epochs required for stable learning."
            )

    # Check 3: Sensitivity threshold should be reasonable
    if curriculum.min_sensitivity_threshold < 0.80:
        return False, (
            f"Minimum sensitivity threshold ({curriculum.min_sensitivity_threshold:.2f}) too low. "
            f"Clinical safety requires ≥ 0.80 sensitivity."
        )

    # Check 4: Final mix ratio should not be too high (defeats curriculum purpose)
    if curriculum.final_mix_ratio > 0.5:
        return False, (
            f"Final mix ratio ({curriculum.final_mix_ratio:.2f}) too high (max 0.5). "
            f"High mix ratio defeats the purpose of curriculum learning."
        )

    # Check 5: Warmup should not be too long
    if curriculum.warmup_epochs > 10:
        return False, (
            f"Warmup epochs ({curriculum.warmup_epochs}) too long (max 10). "
            f"Excessive warmup delays curriculum benefits."
        )

    # Check 6: Sample weights should not be too extreme
    for stage in curriculum.stages:
        if stage.sample_weight > 5.0:
            return False, (
                f"Stage '{stage.name}' sample weight ({stage.sample_weight}) too high (max 5.0). "
                f"Extreme weights risk training instability."
            )

    # All checks passed
    return True, ""


def compute_difficulty_score(
    sample: Dict[str, Any],
    metric: DifficultyMetric
) -> float:
    """
    Compute difficulty score for a sample based on specified metric.

    Args:
        sample: Sample with metadata (image_quality, cdr_ratio, severity, etc.)
        metric: Difficulty metric to use

    Returns:
        Difficulty score (0.0 = easiest, 1.0 = hardest)
    """
    if metric == DifficultyMetric.IMAGE_QUALITY:
        # Higher quality = easier (invert)
        quality = sample.get('image_quality', 0.5)
        return 1.0 - quality

    elif metric == DifficultyMetric.DISC_VISIBILITY:
        # Higher visibility = easier (invert)
        visibility = sample.get('disc_visibility', 0.5)
        return 1.0 - visibility

    elif metric == DifficultyMetric.DISEASE_SEVERITY:
        # Severe cases easier to detect (invert severity)
        severity = sample.get('disease_severity', 0.5)  # 0=normal, 1=severe
        return 1.0 - severity

    elif metric == DifficultyMetric.CDR_RATIO:
        # High CDR = severe = easier to detect (invert)
        cdr = sample.get('cdr_ratio', 0.3)
        # Normalize CDR (0.3 = normal, 0.8 = severe)
        normalized_cdr = (cdr - 0.3) / (0.8 - 0.3)
        normalized_cdr = max(0.0, min(1.0, normalized_cdr))
        return 1.0 - normalized_cdr

    elif metric == DifficultyMetric.PREDICTION_CONFIDENCE:
        # Low confidence = hard sample
        confidence = sample.get('prediction_confidence', 0.5)
        return 1.0 - confidence

    elif metric == DifficultyMetric.COMPOSITE:
        # Weighted combination
        quality_score = 1.0 - sample.get('image_quality', 0.5)
        severity_score = 1.0 - sample.get('disease_severity', 0.5)
        confidence_score = 1.0 - sample.get('prediction_confidence', 0.5)

        # Equal weighting by default
        return (quality_score + severity_score + confidence_score) / 3.0

    else:
        return 0.5  # Default middle difficulty
