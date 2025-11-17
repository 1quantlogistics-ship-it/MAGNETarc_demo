"""
Unit tests for ARC schema validation.

Tests all Pydantic models in schemas.py to ensure proper validation,
serialization, and error handling.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from pydantic import ValidationError

from schemas import (
    # Enums
    OperatingMode, DirectiveMode, Objective, NoveltyClass,
    ReviewDecision, ResourceCost, TrendDirection,
    # Models
    NoveltyBudget, Directive, BestMetrics, ExperimentRecord,
    PerformanceTrends, HistorySummary, ForbiddenRange, Constraints,
    SystemState, ExpectedImpact, Proposal, Proposals,
    Review, Reviews, ActiveExperiment,
    # Functions
    validate_memory_file, save_memory_file, create_default_memory_files,
    SCHEMA_VERSION
)


# ============================================================================
# NoveltyBudget Tests
# ============================================================================

@pytest.mark.unit
class TestNoveltyBudget:
    """Test NoveltyBudget model validation."""

    def test_valid_budget(self):
        """Test creating valid novelty budget."""
        budget = NoveltyBudget(exploit=3, explore=2, wildcat=1)
        assert budget.exploit == 3
        assert budget.explore == 2
        assert budget.wildcat == 1

    def test_zero_budget(self):
        """Test budget with zero values."""
        budget = NoveltyBudget(exploit=0, explore=0, wildcat=0)
        assert budget.exploit == 0

    def test_negative_budget_fails(self):
        """Test that negative budgets are rejected."""
        with pytest.raises(ValidationError):
            NoveltyBudget(exploit=-1, explore=2, wildcat=1)

    def test_excessive_budget_fails(self):
        """Test that budgets exceeding limits are rejected."""
        with pytest.raises(ValidationError):
            NoveltyBudget(exploit=11, explore=2, wildcat=1)


# ============================================================================
# Directive Tests
# ============================================================================

@pytest.mark.unit
class TestDirective:
    """Test Directive model validation."""

    def test_valid_directive(self):
        """Test creating valid directive."""
        directive = Directive(
            cycle_id=1,
            mode=DirectiveMode.EXPLORE,
            objective=Objective.IMPROVE_AUC,
            novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1),
            focus_areas=["learning_rate"],
            notes="Test directive"
        )
        assert directive.cycle_id == 1
        assert directive.mode == DirectiveMode.EXPLORE
        assert len(directive.focus_areas) == 1

    def test_directive_auto_timestamp(self):
        """Test that timestamp is auto-generated."""
        directive = Directive(
            cycle_id=1,
            mode=DirectiveMode.EXPLORE,
            objective=Objective.IMPROVE_AUC,
            novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1)
        )
        assert directive.timestamp is not None
        # Should be valid ISO format
        datetime.fromisoformat(directive.timestamp)

    def test_directive_negative_cycle_fails(self):
        """Test that negative cycle_id is rejected."""
        with pytest.raises(ValidationError):
            Directive(
                cycle_id=-1,
                mode=DirectiveMode.EXPLORE,
                objective=Objective.IMPROVE_AUC,
                novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1)
            )

    def test_directive_notes_too_long_fails(self):
        """Test that notes exceeding max length are rejected."""
        with pytest.raises(ValidationError):
            Directive(
                cycle_id=1,
                mode=DirectiveMode.EXPLORE,
                objective=Objective.IMPROVE_AUC,
                novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1),
                notes="x" * 501  # Exceeds 500 char limit
            )

    def test_directive_serialization(self):
        """Test directive can be serialized to JSON."""
        directive = Directive(
            cycle_id=1,
            mode=DirectiveMode.EXPLORE,
            objective=Objective.IMPROVE_AUC,
            novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1)
        )
        json_str = directive.model_dump_json()
        data = json.loads(json_str)
        assert data["cycle_id"] == 1
        assert data["mode"] == "explore"


# ============================================================================
# HistorySummary Tests
# ============================================================================

@pytest.mark.unit
class TestHistorySummary:
    """Test HistorySummary model validation."""

    def test_empty_history(self):
        """Test creating empty history summary."""
        history = HistorySummary()
        assert history.total_cycles == 0
        assert history.total_experiments == 0
        assert len(history.recent_experiments) == 0

    def test_history_with_metrics(self):
        """Test history with best metrics."""
        history = HistorySummary(
            total_cycles=5,
            total_experiments=15,
            best_metrics=BestMetrics(auc=0.85, sensitivity=0.82, specificity=0.88)
        )
        assert history.best_metrics.auc == 0.85

    def test_history_with_experiments(self):
        """Test history with experiment records."""
        exp = ExperimentRecord(
            experiment_id="exp_1_1",
            auc=0.85,
            timestamp=datetime.utcnow().isoformat()
        )
        history = HistorySummary(
            recent_experiments=[exp]
        )
        assert len(history.recent_experiments) == 1

    def test_history_auto_timestamp(self):
        """Test auto-generated last_updated timestamp."""
        history = HistorySummary()
        assert history.last_updated is not None

    def test_invalid_metric_range(self):
        """Test that metrics outside [0,1] are rejected."""
        with pytest.raises(ValidationError):
            BestMetrics(auc=1.5)  # Invalid: >1.0


# ============================================================================
# Constraints Tests
# ============================================================================

@pytest.mark.unit
class TestConstraints:
    """Test Constraints model validation."""

    def test_empty_constraints(self):
        """Test creating empty constraints."""
        constraints = Constraints()
        assert len(constraints.forbidden_ranges) == 0
        assert constraints.max_learning_rate == 1.0

    def test_constraints_with_forbidden_ranges(self):
        """Test constraints with forbidden parameter ranges."""
        constraints = Constraints(
            forbidden_ranges=[
                ForbiddenRange(
                    param="learning_rate",
                    min=0.1,
                    max=1.0,
                    reason="Causes instability"
                )
            ]
        )
        assert len(constraints.forbidden_ranges) == 1
        assert constraints.forbidden_ranges[0].param == "learning_rate"

    def test_constraints_batch_size_validation(self):
        """Test batch size constraint validation."""
        constraints = Constraints(
            min_batch_size=8,
            max_batch_size=256
        )
        assert constraints.min_batch_size == 8
        assert constraints.max_batch_size == 256

    def test_invalid_learning_rate_fails(self):
        """Test that invalid learning rates are rejected."""
        with pytest.raises(ValidationError):
            Constraints(max_learning_rate=-0.1)  # Negative

        with pytest.raises(ValidationError):
            Constraints(min_batch_size=0)  # Zero batch size


# ============================================================================
# Proposals Tests
# ============================================================================

@pytest.mark.unit
class TestProposals:
    """Test Proposals model validation."""

    def test_single_proposal(self):
        """Test proposals with one experiment."""
        proposals = Proposals(
            cycle_id=5,
            proposals=[
                Proposal(
                    experiment_id="exp_5_1",
                    novelty_class=NoveltyClass.EXPLOIT,
                    hypothesis="Test hypothesis",
                    changes={"learning_rate": 0.001}
                )
            ]
        )
        assert len(proposals.proposals) == 1
        assert proposals.proposals[0].experiment_id == "exp_5_1"

    def test_multiple_proposals(self):
        """Test proposals with multiple experiments."""
        proposals = Proposals(
            cycle_id=5,
            proposals=[
                Proposal(
                    experiment_id="exp_5_1",
                    novelty_class=NoveltyClass.EXPLOIT,
                    hypothesis="Hypothesis 1",
                    changes={"learning_rate": 0.001}
                ),
                Proposal(
                    experiment_id="exp_5_2",
                    novelty_class=NoveltyClass.EXPLORE,
                    hypothesis="Hypothesis 2",
                    changes={"dropout": 0.3}
                )
            ]
        )
        assert len(proposals.proposals) == 2

    def test_duplicate_experiment_ids_fail(self):
        """Test that duplicate experiment IDs are rejected."""
        with pytest.raises(ValidationError, match="Duplicate experiment_id"):
            Proposals(
                cycle_id=5,
                proposals=[
                    Proposal(
                        experiment_id="exp_5_1",
                        novelty_class=NoveltyClass.EXPLOIT,
                        hypothesis="Hypothesis 1",
                        changes={"learning_rate": 0.001}
                    ),
                    Proposal(
                        experiment_id="exp_5_1",  # Duplicate!
                        novelty_class=NoveltyClass.EXPLORE,
                        hypothesis="Hypothesis 2",
                        changes={"dropout": 0.3}
                    )
                ]
            )

    def test_hypothesis_too_short_fails(self):
        """Test that short hypotheses are rejected."""
        with pytest.raises(ValidationError):
            Proposal(
                experiment_id="exp_5_1",
                novelty_class=NoveltyClass.EXPLOIT,
                hypothesis="Short",  # Less than 10 chars
                changes={"learning_rate": 0.001}
            )


# ============================================================================
# Reviews Tests
# ============================================================================

@pytest.mark.unit
class TestReviews:
    """Test Reviews model validation."""

    def test_approved_reviews(self):
        """Test reviews with approved proposals."""
        reviews = Reviews(
            cycle_id=5,
            reviews=[
                Review(
                    proposal_id="exp_5_1",
                    decision=ReviewDecision.APPROVE,
                    issues=[],
                    reasoning="Good proposal"
                )
            ],
            approved=["exp_5_1"]
        )
        assert len(reviews.approved) == 1
        assert "exp_5_1" in reviews.approved

    def test_mixed_reviews(self):
        """Test reviews with mixed decisions."""
        reviews = Reviews(
            cycle_id=5,
            reviews=[
                Review(
                    proposal_id="exp_5_1",
                    decision=ReviewDecision.APPROVE,
                    issues=[],
                    reasoning="Good"
                ),
                Review(
                    proposal_id="exp_5_2",
                    decision=ReviewDecision.REJECT,
                    issues=["Too risky"],
                    reasoning="High risk"
                ),
                Review(
                    proposal_id="exp_5_3",
                    decision=ReviewDecision.REVISE,
                    issues=["Needs changes"],
                    reasoning="Good idea, needs refinement"
                )
            ],
            approved=["exp_5_1"],
            rejected=["exp_5_2"],
            revise=["exp_5_3"]
        )
        assert len(reviews.reviews) == 3
        assert len(reviews.approved) == 1
        assert len(reviews.rejected) == 1
        assert len(reviews.revise) == 1


# ============================================================================
# SystemState Tests
# ============================================================================

@pytest.mark.unit
class TestSystemState:
    """Test SystemState model validation."""

    def test_default_system_state(self):
        """Test creating system state with defaults."""
        state = SystemState(llm_endpoint="http://localhost:8000/v1")
        assert state.mode == OperatingMode.SEMI
        assert state.status == "idle"
        assert len(state.active_experiments) == 0

    def test_system_state_with_active_experiments(self):
        """Test system state with active experiments."""
        state = SystemState(
            llm_endpoint="http://localhost:8000/v1",
            active_experiments=[
                ActiveExperiment(
                    experiment_id="exp_1_1",
                    status="running",
                    pid=12345
                )
            ]
        )
        assert len(state.active_experiments) == 1
        assert state.active_experiments[0].pid == 12345


# ============================================================================
# File I/O Tests
# ============================================================================

@pytest.mark.unit
class TestFileIO:
    """Test file I/O utility functions."""

    def test_save_and_load_memory_file(self, tmp_path):
        """Test saving and loading memory files."""
        file_path = tmp_path / "directive.json"

        # Create and save directive
        directive = Directive(
            cycle_id=1,
            mode=DirectiveMode.EXPLORE,
            objective=Objective.IMPROVE_AUC,
            novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1)
        )
        save_memory_file(str(file_path), directive)

        # Load and validate
        loaded = validate_memory_file(str(file_path), Directive)
        assert loaded.cycle_id == 1
        assert loaded.mode == DirectiveMode.EXPLORE

    def test_atomic_write(self, tmp_path):
        """Test atomic write (no temp file left behind)."""
        file_path = tmp_path / "test.json"
        directive = Directive(
            cycle_id=1,
            mode=DirectiveMode.EXPLORE,
            objective=Objective.IMPROVE_AUC,
            novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1)
        )

        save_memory_file(str(file_path), directive, atomic=True)

        # Verify no temp file exists
        assert not (tmp_path / "test.json.tmp").exists()
        assert file_path.exists()

    def test_create_default_memory_files(self, tmp_path):
        """Test creating default memory files."""
        memory_dir = tmp_path / "memory"
        create_default_memory_files(str(memory_dir))

        # Verify all files created
        assert (memory_dir / "system_state.json").exists()
        assert (memory_dir / "history_summary.json").exists()
        assert (memory_dir / "constraints.json").exists()
        assert (memory_dir / "directive.json").exists()

        # Verify files are valid
        state = validate_memory_file(
            str(memory_dir / "system_state.json"),
            SystemState
        )
        assert state.mode == OperatingMode.SEMI

    def test_load_nonexistent_file_fails(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            validate_memory_file("/nonexistent/file.json", Directive)

    def test_load_invalid_json_fails(self, tmp_path):
        """Test that loading invalid JSON raises error."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            validate_memory_file(str(file_path), Directive)

    def test_load_wrong_schema_fails(self, tmp_path):
        """Test that loading wrong schema raises error."""
        file_path = tmp_path / "wrong.json"

        # Save a Directive
        directive = Directive(
            cycle_id=1,
            mode=DirectiveMode.EXPLORE,
            objective=Objective.IMPROVE_AUC,
            novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1)
        )
        save_memory_file(str(file_path), directive)

        # Try to load as SystemState (should fail)
        with pytest.raises(ValidationError):
            validate_memory_file(str(file_path), SystemState)


# ============================================================================
# Schema Version Test
# ============================================================================

@pytest.mark.unit
def test_schema_version():
    """Test that schema version is defined."""
    assert SCHEMA_VERSION is not None
    assert isinstance(SCHEMA_VERSION, str)
    assert len(SCHEMA_VERSION.split(".")) == 3  # SemVer format
