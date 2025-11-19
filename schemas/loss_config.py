"""
Loss Configuration Schema for ARC.

Defines structured loss function composition for glaucoma detection training.
Enables exploration of loss engineering strategies (focal loss, class weighting,
auxiliary tasks) while maintaining clinical safety.

Key Components:
- Base loss types (BCE, Focal, Dice, Tversky)
- Multi-task loss composition (classification + auxiliary tasks)
- Class weighting strategies
- Loss hyperparameters with safe bounds

Clinical Considerations:
- Optimize for AUC while maintaining sensitivity ≥ 0.85
- Handle class imbalance (glaucoma prevalence ~2-3%)
- Auxiliary tasks: DRI prediction, ISNT ratio prediction
- Avoid over-optimization on specificity at expense of sensitivity

Author: ARC Team (Dev 1)
Created: 2025-11-18
Version: 1.0
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class LossType(str, Enum):
    """
    Base loss function types for glaucoma classification.

    **Classification Losses**:
    - BCE: Binary Cross-Entropy (standard baseline)
    - FOCAL: Focal Loss (handles class imbalance, γ parameter controls focus on hard examples)
    - WEIGHTED_BCE: BCE with class weights (simple imbalance handling)

    **Segmentation-Inspired Losses** (for pixel-level tasks if applicable):
    - DICE: Dice Loss (IoU-based, good for imbalanced segmentation)
    - TVERSKY: Tversky Loss (generalization of Dice, controls FP/FN trade-off)

    **Combined Losses**:
    - BCE_DICE: Combination of BCE + Dice (hybrid classification-segmentation)
    """
    # Classification
    BCE = "bce"
    FOCAL = "focal"
    WEIGHTED_BCE = "weighted_bce"

    # Segmentation-inspired
    DICE = "dice"
    TVERSKY = "tversky"

    # Combined
    BCE_DICE = "bce_dice"


class AuxiliaryTask(str, Enum):
    """
    Auxiliary tasks for multi-task learning.

    Auxiliary tasks provide additional supervision signals that can improve
    primary glaucoma classification performance.

    - DRI_PREDICTION: Predict Disc Relevance Index (continuous regression)
    - ISNT_PREDICTION: Predict ISNT ratio (continuous regression)
    - CDR_PREDICTION: Predict Cup-to-Disc Ratio (continuous regression)
    - VESSEL_DENSITY: Predict retinal vessel density (continuous regression)
    """
    DRI_PREDICTION = "dri_prediction"
    ISNT_PREDICTION = "isnt_prediction"
    CDR_PREDICTION = "cdr_prediction"
    VESSEL_DENSITY = "vessel_density"


class ClassWeightingStrategy(str, Enum):
    """
    Strategy for computing class weights to handle imbalance.

    - NONE: No class weighting (baseline)
    - BALANCED: Inverse class frequency (sklearn-style)
    - EFFECTIVE_SAMPLES: Effective number of samples (Class-Balanced Loss)
    - CUSTOM: User-specified class weights
    """
    NONE = "none"
    BALANCED = "balanced"
    EFFECTIVE_SAMPLES = "effective_samples"
    CUSTOM = "custom"


class LossHyperparameters(BaseModel):
    """
    Hyperparameters for specific loss functions.

    Different loss types require different hyperparameters:
    - Focal Loss: gamma (focus parameter), alpha (class balance)
    - Tversky Loss: alpha (FP weight), beta (FN weight)
    - BCE+Dice: combination weight
    """
    # Focal loss parameters
    focal_gamma: Optional[float] = Field(
        default=None, ge=0.0, le=5.0,
        description="Focal loss gamma (focus on hard examples, 0=BCE, typical: 2.0)"
    )

    focal_alpha: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Focal loss alpha (class balance weight, 0.5=no balance)"
    )

    # Tversky loss parameters
    tversky_alpha: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Tversky alpha (FP weight, higher=penalize FP more)"
    )

    tversky_beta: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Tversky beta (FN weight, higher=penalize FN more)"
    )

    # Combined loss weight
    combination_weight: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Weight for combining losses (e.g., BCE vs Dice)"
    )

    @validator('tversky_alpha', 'tversky_beta')
    def validate_tversky_sum(cls, v, values):
        """Tversky alpha + beta should sum to 1.0."""
        if 'tversky_alpha' in values and 'tversky_beta' in values:
            alpha = values.get('tversky_alpha', 0.0)
            beta = values.get('tversky_beta', 0.0)
            if v is not None and abs(alpha + beta - 1.0) > 0.01:
                raise ValueError(
                    f"Tversky alpha ({alpha}) + beta ({beta}) should sum to 1.0"
                )
        return v


class AuxiliaryTaskConfig(BaseModel):
    """
    Configuration for a single auxiliary task.

    Example:
        {
            "task_type": "dri_prediction",
            "weight": 0.3,
            "loss_type": "mse"
        }
    """
    task_type: AuxiliaryTask = Field(
        description="Type of auxiliary task"
    )

    weight: float = Field(
        ge=0.0, le=1.0,
        description="Task weight in multi-task loss (0.0 to 1.0)"
    )

    loss_type: str = Field(
        default="mse",
        description="Loss function for auxiliary task (mse, mae, huber)"
    )

    @validator('loss_type')
    def validate_auxiliary_loss_type(cls, v):
        """Validate auxiliary task loss type."""
        allowed = ["mse", "mae", "huber", "smooth_l1"]
        if v not in allowed:
            raise ValueError(
                f"Auxiliary loss type '{v}' not in allowed list: {allowed}"
            )
        return v


class LossConfig(BaseModel):
    """
    Complete loss function configuration.

    Defines the loss function composition for training, including:
    - Primary classification loss
    - Optional auxiliary tasks
    - Class weighting strategy
    - Loss-specific hyperparameters

    Clinical Safety:
    - Primary task weight should be ≥ 0.6 (classification is primary goal)
    - Auxiliary task weights should sum to ≤ 0.4
    - Avoid over-weighting specificity at expense of sensitivity

    Example:
        {
            "name": "focal_with_dri",
            "primary_loss": "focal",
            "primary_weight": 0.7,
            "auxiliary_tasks": [
                {"task_type": "dri_prediction", "weight": 0.3, "loss_type": "mse"}
            ],
            "class_weighting": "balanced",
            "hyperparameters": {"focal_gamma": 2.0, "focal_alpha": 0.75}
        }
    """
    name: str = Field(
        description="Human-readable loss config name (e.g., 'focal_gamma2', 'bce_dri_aux')"
    )

    primary_loss: LossType = Field(
        description="Primary classification loss function"
    )

    primary_weight: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Weight for primary classification loss (should be ≥ 0.6 for clinical safety)"
    )

    auxiliary_tasks: Optional[List[AuxiliaryTaskConfig]] = Field(
        default=None,
        description="Optional auxiliary tasks for multi-task learning"
    )

    class_weighting: ClassWeightingStrategy = Field(
        default=ClassWeightingStrategy.NONE,
        description="Strategy for class weight computation"
    )

    custom_class_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom class weights (required if class_weighting='custom')"
    )

    hyperparameters: Optional[LossHyperparameters] = Field(
        default=None,
        description="Loss-specific hyperparameters"
    )

    label_smoothing: float = Field(
        default=0.0, ge=0.0, le=0.2,
        description="Label smoothing factor (0.0 = no smoothing, typical: 0.1)"
    )

    @validator('primary_weight')
    def validate_primary_weight(cls, v):
        """Primary task weight should be ≥ 0.6 for clinical safety."""
        if v < 0.6:
            raise ValueError(
                f"Primary weight ({v}) should be ≥ 0.6 to prioritize classification. "
                f"Auxiliary tasks are supplementary, not primary goals."
            )
        return v

    @validator('auxiliary_tasks')
    def validate_auxiliary_weights(cls, v, values):
        """Auxiliary task weights should sum to ≤ 0.4."""
        if v is None:
            return v

        primary_weight = values.get('primary_weight', 1.0)
        aux_weight_sum = sum(task.weight for task in v)

        total_weight = primary_weight + aux_weight_sum

        if total_weight > 1.0 + 1e-6:  # Allow small floating point error
            raise ValueError(
                f"Total loss weight ({total_weight:.3f}) exceeds 1.0. "
                f"Primary: {primary_weight}, Auxiliary: {aux_weight_sum:.3f}"
            )

        if aux_weight_sum > 0.4:
            raise ValueError(
                f"Auxiliary task weights sum to {aux_weight_sum:.3f} (max 0.4). "
                f"Primary task must remain dominant for clinical safety."
            )

        return v

    @validator('custom_class_weights')
    def validate_custom_weights(cls, v, values):
        """If class_weighting='custom', custom_class_weights must be provided."""
        weighting = values.get('class_weighting')
        if weighting == ClassWeightingStrategy.CUSTOM and v is None:
            raise ValueError(
                "custom_class_weights must be provided when class_weighting='custom'"
            )
        return v

    @validator('hyperparameters')
    def validate_hyperparameters_match_loss(cls, v, values):
        """Validate hyperparameters match primary loss type."""
        primary_loss = values.get('primary_loss')

        if primary_loss == LossType.FOCAL:
            if v is None or v.focal_gamma is None:
                raise ValueError(
                    "Focal loss requires focal_gamma hyperparameter"
                )

        if primary_loss == LossType.TVERSKY:
            if v is None or v.tversky_alpha is None or v.tversky_beta is None:
                raise ValueError(
                    "Tversky loss requires tversky_alpha and tversky_beta hyperparameters"
                )

        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict()

    @classmethod
    def baseline_bce(cls) -> "LossConfig":
        """
        Create baseline BCE loss (current production).

        Returns:
            Simple BCE loss with no auxiliary tasks
        """
        return cls(
            name="baseline_bce",
            primary_loss=LossType.BCE,
            primary_weight=1.0,
            class_weighting=ClassWeightingStrategy.NONE
        )

    @classmethod
    def example_focal_balanced(cls) -> "LossConfig":
        """Example: Focal loss with balanced class weights."""
        return cls(
            name="focal_gamma2_balanced",
            primary_loss=LossType.FOCAL,
            primary_weight=1.0,
            class_weighting=ClassWeightingStrategy.BALANCED,
            hyperparameters=LossHyperparameters(
                focal_gamma=2.0,
                focal_alpha=0.75
            )
        )

    @classmethod
    def example_multitask_dri(cls) -> "LossConfig":
        """Example: BCE with DRI auxiliary task."""
        return cls(
            name="bce_dri_aux",
            primary_loss=LossType.BCE,
            primary_weight=0.7,
            auxiliary_tasks=[
                AuxiliaryTaskConfig(
                    task_type=AuxiliaryTask.DRI_PREDICTION,
                    weight=0.3,
                    loss_type="mse"
                )
            ],
            class_weighting=ClassWeightingStrategy.BALANCED
        )

    @classmethod
    def example_comprehensive(cls) -> "LossConfig":
        """Example: Focal loss with multiple auxiliary tasks."""
        return cls(
            name="focal_multi_aux",
            primary_loss=LossType.FOCAL,
            primary_weight=0.6,
            auxiliary_tasks=[
                AuxiliaryTaskConfig(
                    task_type=AuxiliaryTask.DRI_PREDICTION,
                    weight=0.2,
                    loss_type="mse"
                ),
                AuxiliaryTaskConfig(
                    task_type=AuxiliaryTask.CDR_PREDICTION,
                    weight=0.2,
                    loss_type="smooth_l1"
                )
            ],
            class_weighting=ClassWeightingStrategy.EFFECTIVE_SAMPLES,
            hyperparameters=LossHyperparameters(
                focal_gamma=2.0,
                focal_alpha=0.75
            ),
            label_smoothing=0.1
        )


def validate_loss_safety(loss_config: LossConfig) -> tuple[bool, str]:
    """
    Validate loss configuration for clinical safety.

    Additional validation beyond Pydantic model validators.
    Used by Critic agent for loss config proposal review.

    Args:
        loss_config: Loss configuration to validate

    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: Primary weight must be dominant
    if loss_config.primary_weight < 0.6:
        return False, (
            f"Primary weight ({loss_config.primary_weight}) too low. "
            f"Classification must remain primary goal (≥ 0.6)."
        )

    # Check 2: Focal gamma should not be too high (risks instability)
    if loss_config.hyperparameters and loss_config.hyperparameters.focal_gamma:
        if loss_config.hyperparameters.focal_gamma > 3.0:
            return False, (
                f"Focal gamma ({loss_config.hyperparameters.focal_gamma}) too high (max 3.0). "
                f"High gamma risks training instability."
            )

    # Check 3: Too many auxiliary tasks risks diluting primary objective
    if loss_config.auxiliary_tasks and len(loss_config.auxiliary_tasks) > 3:
        return False, (
            f"Too many auxiliary tasks ({len(loss_config.auxiliary_tasks)}). "
            f"Maximum 3 recommended to maintain focus on primary classification."
        )

    # Check 4: Label smoothing should be conservative
    if loss_config.label_smoothing > 0.15:
        return False, (
            f"Label smoothing ({loss_config.label_smoothing}) too high (max 0.15). "
            f"Excessive smoothing may degrade calibration."
        )

    # Check 5: For Tversky loss, beta should be ≥ alpha (prioritize FN over FP)
    if loss_config.hyperparameters:
        if (loss_config.hyperparameters.tversky_alpha is not None and
            loss_config.hyperparameters.tversky_beta is not None):
            alpha = loss_config.hyperparameters.tversky_alpha
            beta = loss_config.hyperparameters.tversky_beta

            if beta < alpha:
                return False, (
                    f"Tversky beta ({beta}) < alpha ({alpha}). "
                    f"For clinical safety, should prioritize recall (beta ≥ alpha) "
                    f"to avoid missing glaucoma cases (FN)."
                )

    # All checks passed
    return True, ""


def compute_effective_class_weights(
    class_counts: Dict[str, int],
    beta: float = 0.9999
) -> Dict[str, float]:
    """
    Compute class weights using effective number of samples.

    Based on "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019).

    Args:
        class_counts: Dictionary of class counts {"negative": N_neg, "positive": N_pos}
        beta: Hyperparameter controlling re-weighting (default: 0.9999)

    Returns:
        Dictionary of class weights
    """
    weights = {}

    for class_name, count in class_counts.items():
        effective_num = (1.0 - beta ** count) / (1.0 - beta)
        weights[class_name] = 1.0 / effective_num

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    return weights
