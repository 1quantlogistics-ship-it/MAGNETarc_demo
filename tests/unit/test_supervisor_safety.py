"""
Unit Tests for Supervisor Safety Layer (CPU-Only, Offline)
============================================================

Tests Supervisor's algorithmic safety rules without LLM or GPU.

Tests:
- Critical violations (auto-veto)
- High-risk warnings
- Medium-risk warnings
- Risk assessment accuracy
- Veto power validation
- Constraint enforcement
- Consensus override logic
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

# Import supervisor
try:
    import sys
    sys.path.insert(0, '/Users/bengibson/Desktop/ARC/arc_clean')
    from agents.supervisor import SupervisorAgent
    from agents.protocol import RiskLevel
    SUPERVISOR_AVAILABLE = True
except ImportError:
    SUPERVISOR_AVAILABLE = False
    pytest.skip("Supervisor not available", allow_module_level=True)


def create_proposal(experiment_id: str, config_changes: Dict[str, Any], novelty="exploit") -> Dict[str, Any]:
    """Create test proposal."""
    return {
        "experiment_id": experiment_id,
        "config_changes": config_changes,
        "changes": config_changes,  # Support both formats
        "novelty_category": novelty,
        "description": f"Test proposal {experiment_id}"
    }


class TestSupervisorSafety:
    """Test suite for Supervisor safety rules."""

    @pytest.fixture
    def supervisor(self, tmp_path):
        """Create supervisor with temp memory."""
        memory_path = tmp_path / "memory"
        memory_path.mkdir()

        # Create empty constraints file
        constraints_file = memory_path / "constraints.json"
        with open(constraints_file, 'w') as f:
            json.dump({"forbidden_ranges": []}, f)

        supervisor = SupervisorAgent(memory_path=str(memory_path))
        return supervisor

    # ========== CRITICAL VIOLATIONS (AUTO-VETO) ==========

    def test_critical_learning_rate_too_high(self, supervisor):
        """Test veto for LR > 0.01."""
        proposal = create_proposal("test_001", {"learning_rate": 0.1})

        risk = supervisor._assess_risk(proposal)

        assert risk == RiskLevel.CRITICAL, f"Expected CRITICAL for LR=0.1, got {risk.value}"

        vote = supervisor.vote_on_proposal(proposal)
        assert vote["decision"] == "reject", "Should reject LR > 0.01"

    def test_critical_batch_size_too_large(self, supervisor):
        """Test veto for BS > 64."""
        proposal = create_proposal("test_002", {"batch_size": 128})

        risk = supervisor._assess_risk(proposal)

        assert risk == RiskLevel.CRITICAL, f"Expected CRITICAL for BS=128, got {risk.value}"

        vote = supervisor.vote_on_proposal(proposal)
        assert vote["decision"] == "reject", "Should reject BS > 64"

    def test_critical_epochs_too_high(self, supervisor):
        """Test veto for epochs > 200."""
        proposal = create_proposal("test_003", {"epochs": 500})

        risk = supervisor._assess_risk(proposal)

        assert risk == RiskLevel.CRITICAL, f"Expected CRITICAL for epochs=500, got {risk.value}"

        vote = supervisor.vote_on_proposal(proposal)
        assert vote["decision"] == "reject", "Should reject epochs > 200"

    def test_critical_invalid_optimizer(self, supervisor):
        """Test veto for unknown optimizer."""
        proposal = create_proposal("test_004", {"optimizer": "invalid_opt"})

        risk = supervisor._assess_risk(proposal)

        assert risk == RiskLevel.CRITICAL, f"Expected CRITICAL for invalid optimizer, got {risk.value}"

    def test_critical_invalid_loss(self, supervisor):
        """Test veto for unknown loss function."""
        proposal = create_proposal("test_005", {"loss": "invalid_loss"})

        risk = supervisor._assess_risk(proposal)

        assert risk == RiskLevel.CRITICAL, f"Expected CRITICAL for invalid loss, got {risk.value}"

    # ========== HIGH VIOLATIONS (STRONG WARNING) ==========

    def test_high_learning_rate_too_low(self, supervisor):
        """Test HIGH risk for LR < 1e-6."""
        proposal = create_proposal("test_006", {"learning_rate": 1e-7})

        risk = supervisor._assess_risk(proposal)

        assert risk == RiskLevel.HIGH, f"Expected HIGH for LR=1e-7, got {risk.value}"

    def test_high_batch_size_too_small(self, supervisor):
        """Test HIGH risk for BS < 2."""
        proposal = create_proposal("test_007", {"batch_size": 1})

        risk = supervisor._assess_risk(proposal)

        assert risk == RiskLevel.HIGH, f"Expected HIGH for BS=1, got {risk.value}"

    def test_high_dropout_too_high(self, supervisor):
        """Test HIGH risk for dropout > 0.7."""
        proposal = create_proposal("test_008", {"dropout": 0.9})

        risk = supervisor._assess_risk(proposal)

        assert risk == RiskLevel.HIGH, f"Expected HIGH for dropout=0.9, got {risk.value}"

    def test_high_epochs_too_low(self, supervisor):
        """Test HIGH risk for epochs < 3."""
        proposal = create_proposal("test_009", {"epochs": 1})

        risk = supervisor._assess_risk(proposal)

        assert risk == RiskLevel.HIGH, f"Expected HIGH for epochs=1, got {risk.value}"

    # ========== SAFE CONFIGURATIONS ==========

    def test_safe_standard_config(self, supervisor):
        """Test approval for standard safe config."""
        proposal = create_proposal("test_010", {
            "learning_rate": 0.0001,
            "batch_size": 8,
            "epochs": 10,
            "dropout": 0.2,
            "optimizer": "adam",
            "loss": "focal"
        })

        risk = supervisor._assess_risk(proposal)

        assert risk in [RiskLevel.LOW, RiskLevel.MEDIUM], f"Expected LOW/MEDIUM for safe config, got {risk.value}"

        vote = supervisor.vote_on_proposal(proposal)
        assert vote["decision"] == "approve", "Should approve safe config"

    def test_safe_efficientnet_config(self, supervisor):
        """Test approval for EfficientNet config."""
        proposal = create_proposal("test_011", {
            "learning_rate": 0.0001,
            "batch_size": 16,
            "epochs": 20,
            "dropout": 0.3,
            "optimizer": "adamw",
            "loss": "cross_entropy",
            "model": "efficientnet_b3"
        })

        risk = supervisor._assess_risk(proposal)

        assert risk in [RiskLevel.LOW, RiskLevel.MEDIUM], "EfficientNet config should be safe"

        vote = supervisor.vote_on_proposal(proposal)
        assert vote["decision"] == "approve", "Should approve EfficientNet config"

    # ========== SAFETY RULE CHECKS ==========

    def test_safety_rules_multiple_violations(self, supervisor):
        """Test handling of multiple violations."""
        proposal = create_proposal("test_012", {
            "learning_rate": 0.1,  # CRITICAL
            "batch_size": 200,      # CRITICAL
            "dropout": 0.95         # HIGH
        })

        violations = supervisor._check_safety_rules(proposal["config_changes"])

        assert len(violations["critical"]) >= 2, "Should detect multiple critical violations"
        assert "Learning rate" in str(violations["critical"])
        assert "Batch size" in str(violations["critical"])

    def test_safety_rules_no_violations(self, supervisor):
        """Test safe config has no violations."""
        config = {
            "learning_rate": 0.0001,
            "batch_size": 8,
            "epochs": 10,
            "optimizer": "adam",
            "loss": "focal"
        }

        violations = supervisor._check_safety_rules(config)

        assert len(violations["critical"]) == 0, "Safe config should have no critical violations"
        assert len(violations["high"]) == 0, "Safe config should have no high violations"

    # ========== CONSTRAINT ENFORCEMENT ==========

    def test_forbidden_range_enforcement(self, tmp_path):
        """Test enforcement of forbidden parameter ranges."""
        memory_path = tmp_path / "memory"
        memory_path.mkdir()

        # Create constraints with forbidden range
        constraints = {
            "forbidden_ranges": [
                {
                    "parameter": "learning_rate",
                    "min": 0.005,
                    "max": 0.01,
                    "reason": "Causes training instability"
                }
            ]
        }

        constraints_file = memory_path / "constraints.json"
        with open(constraints_file, 'w') as f:
            json.dump(constraints, f)

        supervisor = SupervisorAgent(memory_path=str(memory_path))

        # Test config in forbidden range
        proposal = create_proposal("test_013", {"learning_rate": 0.007})

        risk = supervisor._assess_risk(proposal)

        assert risk == RiskLevel.CRITICAL, "Should detect forbidden range violation"

        violations = supervisor._check_constraints(proposal)
        assert len(violations) > 0, "Should report constraint violation"

    # ========== CONSENSUS OVERRIDE ==========

    def test_override_on_critical_risk(self, supervisor):
        """Test Supervisor overrides consensus if risk is critical."""
        proposal = create_proposal("test_014", {"learning_rate": 0.5})  # CRITICAL

        # Simulate consensus approval (66% approve)
        votes = [
            {"decision": "approve", "confidence": 0.8},
            {"decision": "approve", "confidence": 0.7},
            {"decision": "reject", "confidence": 0.6}
        ]

        decision = supervisor.validate_consensus(proposal, votes)

        assert decision.decision == "reject", "Should override consensus and reject"
        assert decision.override_consensus == True, "Should flag as override"
        assert "OVERRIDE" in decision.reasoning.upper(), "Should mention override in reasoning"

    def test_no_override_on_safe_proposal(self, supervisor):
        """Test Supervisor validates consensus for safe proposal."""
        proposal = create_proposal("test_015", {
            "learning_rate": 0.0001,
            "batch_size": 8
        })

        # Simulate consensus approval
        votes = [
            {"decision": "approve", "confidence": 0.8},
            {"decision": "approve", "confidence": 0.9},
            {"decision": "approve", "confidence": 0.7}
        ]

        decision = supervisor.validate_consensus(proposal, votes)

        assert decision.decision == "approve", "Should validate consensus approval"
        assert decision.override_consensus == False, "Should not override safe consensus"

    # ========== EDGE CASES ==========

    def test_empty_config_changes(self, supervisor):
        """Test handling of empty config changes."""
        proposal = create_proposal("test_016", {})

        risk = supervisor._assess_risk(proposal)

        # Empty config should be LOW risk (no violations)
        assert risk == RiskLevel.LOW, "Empty config should be low risk"

    def test_missing_parameters(self, supervisor):
        """Test handling of missing optional parameters."""
        proposal = create_proposal("test_017", {
            "learning_rate": 0.0001
            # Missing: batch_size, epochs, optimizer, loss
        })

        violations = supervisor._check_safety_rules(proposal["config_changes"])

        # Should not crash on missing params
        assert violations is not None
        assert len(violations["critical"]) == 0, "Missing params should not cause critical violations"

    def test_world_model_uncertainty_integration(self, supervisor):
        """Test integration with world-model uncertainty (if available)."""
        # Add world-model prediction to proposal
        proposal = create_proposal("test_018", {
            "learning_rate": 0.0001,
            "batch_size": 8
        })

        # Add low predicted metric
        proposal["world_model_prediction"] = {
            "predicted_auc": 0.55,
            "uncertainty": 0.3,
            "confidence": 0.4
        }

        # For now, risk assessment ignores world-model (can be enhanced later)
        risk = supervisor._assess_risk(proposal)

        # Should still assess based on config rules
        assert risk is not None


def test_supervisor_safety_rules_comprehensive():
    """Comprehensive integration test of all safety rules."""
    print("\n" + "="*60)
    print("SUPERVISOR SAFETY RULES TEST (CPU-ONLY)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        memory_path = Path(tmp_dir) / "memory"
        memory_path.mkdir()

        # Create constraints
        constraints_file = memory_path / "constraints.json"
        with open(constraints_file, 'w') as f:
            json.dump({"forbidden_ranges": []}, f)

        supervisor = SupervisorAgent(memory_path=str(memory_path))

        # Test critical violations
        critical_proposals = [
            ("LR too high", {"learning_rate": 0.1}),
            ("BS too large", {"batch_size": 128}),
            ("Epochs excessive", {"epochs": 500}),
            ("Invalid optimizer", {"optimizer": "invalid"}),
            ("Invalid loss", {"loss": "unknown_loss"})
        ]

        print("\n✓ Testing CRITICAL violations (auto-veto):")
        for name, config in critical_proposals:
            proposal = create_proposal(f"critical_{name}", config)
            risk = supervisor._assess_risk(proposal)
            vote = supervisor.vote_on_proposal(proposal)

            status = "✓ VETOED" if vote["decision"] == "reject" else "✗ FAILED TO VETO"
            print(f"  {status}: {name} (risk={risk.value})")

            assert risk == RiskLevel.CRITICAL, f"{name} should be CRITICAL"
            assert vote["decision"] == "reject", f"{name} should be rejected"

        # Test safe proposals
        safe_proposals = [
            ("Standard config", {
                "learning_rate": 0.0001,
                "batch_size": 8,
                "epochs": 10,
                "optimizer": "adam",
                "loss": "focal"
            }),
            ("EfficientNet config", {
                "learning_rate": 0.0001,
                "batch_size": 16,
                "epochs": 20,
                "optimizer": "adamw",
                "loss": "cross_entropy"
            })
        ]

        print("\n✓ Testing SAFE proposals (should approve):")
        for name, config in safe_proposals:
            proposal = create_proposal(f"safe_{name}", config)
            risk = supervisor._assess_risk(proposal)
            vote = supervisor.vote_on_proposal(proposal)

            status = "✓ APPROVED" if vote["decision"] == "approve" else "✗ FAILED TO APPROVE"
            print(f"  {status}: {name} (risk={risk.value})")

            assert risk in [RiskLevel.LOW, RiskLevel.MEDIUM], f"{name} should be safe"
            assert vote["decision"] == "approve", f"{name} should be approved"

    print("\n" + "="*60)
    print("✓ ALL SUPERVISOR SAFETY TESTS PASSED (CPU-only)")
    print("="*60)


if __name__ == "__main__":
    test_supervisor_safety_rules_comprehensive()
    pytest.main([__file__, "-v"])
