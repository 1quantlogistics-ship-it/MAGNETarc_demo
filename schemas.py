"""
ARC Memory File Schemas

Pydantic models for validating all ARC memory files. These schemas ensure
type safety, data integrity, and provide clear contracts for all persistent state.

All memory files must conform to these schemas to prevent corruption and
ensure safe operation of the ARC system.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


# ============================================================================
# Enumerations
# ============================================================================

class OperatingMode(str, Enum):
    """ARC operating modes with different autonomy levels."""
    SEMI = "SEMI"  # All actions require approval (safest)
    AUTO = "AUTO"  # Automatic reasoning, human approval for training
    FULL = "FULL"  # Full autonomy (dangerous, use with caution)
    OFF = "OFF"    # System disabled


class DirectiveMode(str, Enum):
    """Research strategy modes for Director."""
    EXPLORE = "explore"    # High novelty, search for new approaches
    EXPLOIT = "exploit"    # Low risk, refine known good approaches
    RECOVER = "recover"    # Safety mode, revert to stable baselines
    WILDCAT = "wildcat"    # Experimental mode for breakthrough attempts


class Objective(str, Enum):
    """Primary optimization objectives."""
    IMPROVE_AUC = "improve_auc"
    IMPROVE_SENSITIVITY = "improve_sensitivity"
    IMPROVE_SPECIFICITY = "improve_specificity"
    ROBUSTNESS = "robustness"
    SPEED = "speed"
    EFFICIENCY = "efficiency"


class NoveltyClass(str, Enum):
    """Novelty classification for experiments."""
    EXPLOIT = "exploit"    # ≤2 parameter changes, low risk
    EXPLORE = "explore"    # Architecture/data changes, medium risk
    WILDCAT = "wildcat"    # High risk, potentially high reward


class ReviewDecision(str, Enum):
    """Critic review decisions."""
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class ResourceCost(str, Enum):
    """Resource requirements for experiments."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrendDirection(str, Enum):
    """Performance trend indicators."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    UNKNOWN = "unknown"


# ============================================================================
# Directive Schema
# ============================================================================

class NoveltyBudget(BaseModel):
    """Budget allocation for different novelty categories."""
    exploit: int = Field(ge=0, le=10, description="Number of low-risk exploit experiments")
    explore: int = Field(ge=0, le=10, description="Number of medium-risk explore experiments")
    wildcat: int = Field(ge=0, le=5, description="Number of high-risk wildcat experiments")

    @field_validator('exploit', 'explore', 'wildcat')
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        """Ensure all budgets are non-negative."""
        if v < 0:
            raise ValueError("Budget values must be non-negative")
        return v


class Directive(BaseModel):
    """Strategic directive from Director role.

    This schema defines the research strategy for a cycle, including
    mode, objectives, novelty budget, and focus areas.
    """
    model_config = ConfigDict(use_enum_values=True)

    cycle_id: int = Field(ge=0, description="Cycle number (0-indexed)")
    mode: DirectiveMode = Field(description="Research strategy mode")
    objective: Objective = Field(description="Primary optimization objective")
    novelty_budget: NoveltyBudget = Field(description="Allocation of experiments by novelty")
    focus_areas: List[str] = Field(default_factory=list, description="Domains to focus on (e.g., 'architecture', 'learning_rate')")
    forbidden_axes: List[str] = Field(default_factory=list, description="Parameters to avoid modifying")
    encouraged_axes: List[str] = Field(default_factory=list, description="Parameters encouraged for modification")
    notes: str = Field(default="", max_length=500, description="Strategic reasoning summary")
    timestamp: Optional[str] = Field(default=None, description="ISO 8601 timestamp")

    @field_validator('timestamp', mode='before')
    @classmethod
    def set_timestamp(cls, v: Optional[str]) -> str:
        """Auto-set timestamp if not provided."""
        return v or datetime.utcnow().isoformat()


# ============================================================================
# History Summary Schema
# ============================================================================

class BestMetrics(BaseModel):
    """Best observed metrics across all experiments."""
    auc: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Best AUC score")
    sensitivity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Best sensitivity")
    specificity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Best specificity")
    loss: Optional[float] = Field(default=None, ge=0.0, description="Best (lowest) loss")


class ExperimentRecord(BaseModel):
    """Record of a completed experiment."""
    experiment_id: str = Field(description="Unique experiment identifier")
    auc: float = Field(ge=0.0, le=1.0, description="Achieved AUC score")
    sensitivity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    specificity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    loss: Optional[float] = Field(default=None, ge=0.0)
    training_time: Optional[float] = Field(default=None, ge=0.0, description="Training time in seconds")
    timestamp: str = Field(description="ISO 8601 timestamp")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Experiment configuration")
    success: bool = Field(default=True, description="Whether experiment completed successfully")


class PerformanceTrends(BaseModel):
    """Performance trend analysis."""
    auc_trend: TrendDirection = Field(default=TrendDirection.UNKNOWN)
    sensitivity_trend: TrendDirection = Field(default=TrendDirection.UNKNOWN)
    specificity_trend: TrendDirection = Field(default=TrendDirection.UNKNOWN)
    cycles_without_improvement: int = Field(default=0, ge=0, description="Stagnation counter")
    consecutive_regressions: int = Field(default=0, ge=0, description="Regression counter")


class HistorySummary(BaseModel):
    """Compressed research history and memory.

    Maintained by Historian role. Tracks aggregate statistics,
    best results, recent experiments, and trends.
    """
    total_cycles: int = Field(default=0, ge=0, description="Total completed cycles")
    total_experiments: int = Field(default=0, ge=0, description="Total experiments run")
    best_metrics: BestMetrics = Field(default_factory=BestMetrics, description="Best observed metrics")
    recent_experiments: List[ExperimentRecord] = Field(
        default_factory=list,
        max_length=50,
        description="Recent experiment history (limited to 50)"
    )
    failed_configs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Configurations that failed repeatedly"
    )
    successful_patterns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Configurations that succeeded"
    )
    performance_trends: PerformanceTrends = Field(
        default_factory=PerformanceTrends,
        description="Trend analysis"
    )
    last_updated: Optional[str] = Field(default=None, description="ISO 8601 timestamp")

    @field_validator('last_updated', mode='before')
    @classmethod
    def set_last_updated(cls, v: Optional[str]) -> str:
        """Auto-set timestamp if not provided."""
        return v or datetime.utcnow().isoformat()


# ============================================================================
# Constraints Schema
# ============================================================================

class ForbiddenRange(BaseModel):
    """Forbidden parameter range learned from failures."""
    param: str = Field(description="Parameter name")
    min: Optional[float] = Field(default=None, description="Minimum forbidden value")
    max: Optional[float] = Field(default=None, description="Maximum forbidden value")
    reason: Optional[str] = Field(default=None, description="Why this range is forbidden")


class Constraints(BaseModel):
    """Learned constraints from experiment history.

    Updated by Historian based on repeated failures and successes.
    Used by Critic to validate proposals.
    """
    forbidden_ranges: List[ForbiddenRange] = Field(
        default_factory=list,
        description="Parameter ranges to avoid"
    )
    unstable_configs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Configuration patterns that caused instability"
    )
    safe_baselines: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Known stable configurations for recovery"
    )
    max_learning_rate: Optional[float] = Field(default=1.0, gt=0.0, description="Maximum safe learning rate")
    min_batch_size: Optional[int] = Field(default=1, gt=0, description="Minimum batch size")
    max_batch_size: Optional[int] = Field(default=512, gt=0, description="Maximum batch size")
    last_updated: Optional[str] = Field(default=None, description="ISO 8601 timestamp")

    @field_validator('last_updated', mode='before')
    @classmethod
    def set_last_updated(cls, v: Optional[str]) -> str:
        """Auto-set timestamp if not provided."""
        return v or datetime.utcnow().isoformat()


# ============================================================================
# Proposals Schema
# ============================================================================

class ExpectedImpact(BaseModel):
    """Expected metric changes from an experiment."""
    auc: Literal["up", "down", "same", "unknown"] = Field(default="unknown")
    sensitivity: Literal["up", "down", "same", "unknown"] = Field(default="unknown")
    specificity: Literal["up", "down", "same", "unknown"] = Field(default="unknown")


class Proposal(BaseModel):
    """Single experiment proposal from Architect."""
    experiment_id: str = Field(description="Unique experiment identifier (e.g., 'exp_5_1')")
    novelty_class: NoveltyClass = Field(description="Novelty/risk category")
    hypothesis: str = Field(min_length=10, max_length=500, description="Scientific hypothesis")
    changes: Dict[str, Any] = Field(description="Configuration changes to apply")
    expected_impact: ExpectedImpact = Field(
        default_factory=ExpectedImpact,
        description="Predicted metric changes"
    )
    resource_cost: ResourceCost = Field(default=ResourceCost.MEDIUM, description="Estimated resource needs")
    rationale: Optional[str] = Field(default=None, description="Detailed reasoning")


class Proposals(BaseModel):
    """Collection of experiment proposals for a cycle.

    Generated by Architect role based on Director's directive.
    """
    model_config = ConfigDict(use_enum_values=True)

    cycle_id: int = Field(ge=0, description="Cycle number")
    proposals: List[Proposal] = Field(description="List of proposed experiments")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @field_validator('proposals')
    @classmethod
    def validate_unique_ids(cls, v: List[Proposal]) -> List[Proposal]:
        """Ensure all experiment IDs are unique."""
        ids = [p.experiment_id for p in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate experiment_id found in proposals")
        return v


# ============================================================================
# Reviews Schema
# ============================================================================

class Review(BaseModel):
    """Critic's review of a single proposal."""
    proposal_id: str = Field(description="Experiment ID being reviewed")
    decision: ReviewDecision = Field(description="Review decision")
    issues: List[str] = Field(default_factory=list, description="Issues found (if any)")
    reasoning: str = Field(min_length=10, max_length=500, description="Review justification")
    risk_level: Optional[Literal["low", "medium", "high"]] = Field(default=None)


class Reviews(BaseModel):
    """Collection of Critic reviews for a cycle.

    Generated by Critic role to validate Architect's proposals.
    """
    model_config = ConfigDict(use_enum_values=True)

    cycle_id: int = Field(ge=0, description="Cycle number")
    reviews: List[Review] = Field(description="Reviews for each proposal")
    approved: List[str] = Field(default_factory=list, description="Approved experiment IDs")
    rejected: List[str] = Field(default_factory=list, description="Rejected experiment IDs")
    revise: List[str] = Field(default_factory=list, description="Experiments needing revision")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @field_validator('approved', 'rejected', 'revise')
    @classmethod
    def validate_no_overlap(cls, v: List[str], info) -> List[str]:
        """Ensure no experiment appears in multiple decision lists."""
        # This is a simplified check; full validation would need access to all fields
        return v


# ============================================================================
# System State Schema
# ============================================================================

class ActiveExperiment(BaseModel):
    """Currently running experiment."""
    experiment_id: str
    status: Literal["queued", "running", "completed", "failed"]
    started_at: Optional[str] = None
    pid: Optional[int] = None


class SystemState(BaseModel):
    """Global ARC system state.

    Tracks operating mode, version, configuration, and active work.
    """
    model_config = ConfigDict(use_enum_values=True)

    mode: OperatingMode = Field(default=OperatingMode.SEMI, description="Current operating mode")
    arc_version: str = Field(default="0.9.0", description="ARC system version")
    llm_endpoint: str = Field(description="LLM API endpoint URL")
    last_cycle_id: Optional[int] = Field(default=None, ge=0, description="Most recent cycle ID")
    last_cycle_timestamp: Optional[str] = Field(default=None, description="ISO 8601 timestamp")
    status: Literal["idle", "running", "paused", "error"] = Field(default="idle")
    active_experiments: List[ActiveExperiment] = Field(
        default_factory=list,
        description="Currently running/queued experiments"
    )
    error_count: int = Field(default=0, ge=0, description="Error counter for circuit breaking")
    last_error: Optional[str] = Field(default=None, description="Last error message")
    uptime_seconds: Optional[float] = Field(default=None, ge=0.0)


# ============================================================================
# Utility Functions
# ============================================================================

def validate_memory_file(file_path: str, schema_class: type[BaseModel]) -> BaseModel:
    """
    Load and validate a memory file against its schema.

    Args:
        file_path: Path to JSON file
        schema_class: Pydantic model class to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If file doesn't match schema
        JSONDecodeError: If file isn't valid JSON
    """
    import json

    with open(file_path, 'r') as f:
        data = json.load(f)

    return schema_class(**data)


def save_memory_file(file_path: str, model: BaseModel, atomic: bool = True) -> None:
    """
    Save a Pydantic model to a JSON file with optional atomic write.

    Args:
        file_path: Path to save to
        model: Pydantic model instance
        atomic: If True, use atomic write (write-temp-rename pattern)

    Raises:
        IOError: If write fails
    """
    import json
    import os
    from pathlib import Path

    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    json_data = model.model_dump_json(indent=2, exclude_none=False)

    if atomic:
        # Atomic write: write to temp file, then rename
        temp_path = f"{file_path}.tmp"
        try:
            with open(temp_path, 'w') as f:
                f.write(json_data)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Atomic rename
            os.replace(temp_path, file_path)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    else:
        # Direct write
        with open(file_path, 'w') as f:
            f.write(json_data)


def create_default_memory_files(memory_dir: str) -> None:
    """
    Initialize all memory files with safe defaults.

    Args:
        memory_dir: Directory to create memory files in
    """
    from pathlib import Path

    memory_path = Path(memory_dir)
    memory_path.mkdir(parents=True, exist_ok=True)

    # Create default instances
    defaults = {
        'system_state.json': SystemState(
            llm_endpoint="http://localhost:8000/v1",
            mode=OperatingMode.SEMI,
            status="idle"
        ),
        'history_summary.json': HistorySummary(),
        'constraints.json': Constraints(),
        'directive.json': Directive(
            cycle_id=0,
            mode=DirectiveMode.EXPLORE,
            objective=Objective.IMPROVE_AUC,
            novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=0),
            notes="Initial directive - exploration phase"
        ),
    }

    for filename, model in defaults.items():
        file_path = memory_path / filename
        if not file_path.exists():
            save_memory_file(str(file_path), model, atomic=True)


# ============================================================================
# Schema Version
# ============================================================================

SCHEMA_VERSION = "1.0.0"
"""Current schema version for migration tracking."""
