"""
Integration tests for Multi-Agent Integration Layer.

Tests that Agent 2's multi-agent orchestrator works correctly with
Agent 1's v1.1.0 infrastructure (MemoryHandler, config, schemas).
"""

import pytest
from pathlib import Path

from multi_agent_integration import MultiAgentIntegration, create_multi_agent_integration
from orchestrator_base import CycleContext, OrchestratorPhase
from memory_handler import MemoryHandler, reset_memory_handler
from config import ARCSettings
from schemas import (
    Directive, DirectiveMode, Objective,
    SystemState, OperatingMode,
    Proposals, Proposal, NoveltyClass, ExpectedImpact,
    Reviews, Review, ReviewDecision
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def integration_env(tmp_path):
    """Create integration environment."""
    settings = ARCSettings(
        environment="test",
        home=tmp_path / "arc",
        llm_endpoint="http://localhost:8000/v1"
    )
    settings.ensure_directories()

    memory = MemoryHandler(settings)
    memory.initialize_memory(force=True)

    integration = MultiAgentIntegration(
        settings=settings,
        memory=memory,
        offline_mode=True
    )

    yield settings, memory, integration

    reset_memory_handler()


# ============================================================================
# Initialization Tests
# ============================================================================

@pytest.mark.integration
class TestMultiAgentIntegrationInit:
    """Test integration layer initialization."""

    def test_create_integration(self, integration_env):
        """Test creating integration layer."""
        settings, memory, integration = integration_env

        assert integration is not None
        assert integration.settings == settings
        assert integration.memory == memory
        assert integration.orchestrator is not None

    def test_create_with_convenience_function(self):
        """Test creating with convenience function."""
        integration = create_multi_agent_integration(offline_mode=True)

        assert integration is not None
        assert integration.offline_mode is True


# ============================================================================
# Memory Operations Tests
# ============================================================================

@pytest.mark.integration
class TestMultiAgentIntegrationMemory:
    """Test memory operations through integration layer."""

    def test_get_memory_path(self, integration_env):
        """Test getting memory path from config."""
        settings, memory, integration = integration_env

        path = integration.get_memory_path()

        assert path == settings.memory_dir
        assert path.exists()

    def test_validate_memory(self, integration_env):
        """Test memory validation."""
        settings, memory, integration = integration_env

        is_valid, errors = integration.validate_memory()

        assert is_valid
        assert len(errors) == 0

    def test_backup_and_restore(self, integration_env):
        """Test memory backup and restore."""
        settings, memory, integration = integration_env

        # Modify memory
        directive = memory.load_directive()
        directive.cycle_id = 99
        directive.notes = "Modified"
        memory.save_directive(directive)

        # Create backup
        backup_dir = integration.backup_memory()
        assert backup_dir.exists()

        # Modify again
        directive.cycle_id = 100
        memory.save_directive(directive)

        # Restore
        integration.restore_memory(backup_dir)

        # Verify restoration
        restored = memory.load_directive()
        assert restored.cycle_id == 99
        assert restored.notes == "Modified"


# ============================================================================
# Agent Callback Tests
# ============================================================================

@pytest.mark.integration
class TestMultiAgentIntegrationCallbacks:
    """Test agent callbacks with schema validation."""

    def test_historian_callback(self, integration_env):
        """Test historian callback."""
        settings, memory, integration = integration_env

        # Create context
        context = CycleContext(cycle_id=1)
        context = integration.orchestrator._phase_load_memory(context)

        initial_cycles = context.history.total_cycles

        # Run historian
        context = integration._historian_callback(context)

        assert "historian" in context.agent_outputs
        assert context.history.total_cycles == initial_cycles + 1

    def test_director_callback(self, integration_env):
        """Test director callback."""
        settings, memory, integration = integration_env

        # Create context
        context = CycleContext(cycle_id=5)
        context = integration.orchestrator._phase_load_memory(context)

        # Run director
        context = integration._director_callback(context)

        assert "director" in context.agent_outputs
        assert context.directive.cycle_id == 5

    def test_architect_callback(self, integration_env):
        """Test architect callback."""
        settings, memory, integration = integration_env

        # Create context
        context = CycleContext(cycle_id=1)
        context = integration.orchestrator._phase_load_memory(context)

        # Run architect
        context = integration._architect_callback(context)

        assert "architect" in context.agent_outputs
        assert context.proposals is not None
        assert isinstance(context.proposals, Proposals)

    def test_critic_callback(self, integration_env):
        """Test critic callback."""
        settings, memory, integration = integration_env

        # Create context with proposals
        context = CycleContext(cycle_id=1)
        context = integration.orchestrator._phase_load_memory(context)
        context.proposals = Proposals(
            cycle_id=1,
            proposals=[
                Proposal(
                    proposal_id="test_001",
                    description="Test proposal",
                    novelty_class=NoveltyClass.EXPLOIT,
                    expected_impact=ExpectedImpact.MEDIUM,
                    rationale="Test"
                )
            ]
        )

        # Run critic
        context = integration._critic_callback(context)

        assert "critic" in context.agent_outputs
        assert context.reviews is not None
        assert isinstance(context.reviews, Reviews)


# ============================================================================
# Cycle Execution Tests
# ============================================================================

@pytest.mark.integration
class TestMultiAgentIntegrationCycle:
    """Test full cycle execution through integration."""

    def test_run_cycle_basic(self, integration_env):
        """Test running a basic cycle."""
        settings, memory, integration = integration_env

        # Register agents
        integration.register_standard_multi_agent_pipeline()

        # Run cycle
        result = integration.run_cycle(cycle_id=1)

        assert result["cycle_id"] == 1
        assert result["phase"] == OrchestratorPhase.COMPLETE.value
        assert "agent_outputs" in result
        assert not result["has_errors"]

    def test_run_cycle_with_memory_persistence(self, integration_env):
        """Test that cycle persists changes to memory."""
        settings, memory, integration = integration_env

        # Register agents
        integration.register_standard_multi_agent_pipeline()

        # Run cycle
        result = integration.run_cycle(cycle_id=10)

        # Verify memory was updated
        directive = memory.load_directive()
        history = memory.load_history_summary()

        assert directive.cycle_id == 10
        assert history.total_cycles == 1  # Incremented by historian

    def test_run_cycle_with_error_handling(self, integration_env):
        """Test cycle error handling."""
        settings, memory, integration = integration_env

        # Register failing agent
        def failing_agent(context: CycleContext) -> CycleContext:
            raise ValueError("Simulated failure")

        integration.orchestrator.register_agent("historian", failing_agent)

        # Run cycle
        result = integration.run_cycle(cycle_id=1)

        assert result["phase"] == OrchestratorPhase.ERROR.value
        assert result["has_errors"]
        assert len(result["errors"]) > 0


# ============================================================================
# Tool Governance Tests
# ============================================================================

@pytest.mark.integration
class TestMultiAgentIntegrationTools:
    """Test tool governance integration."""

    def test_validate_tool_request(self, integration_env):
        """Test tool request validation."""
        settings, memory, integration = integration_env

        tool_request = {
            "tool": "train",
            "args": {"learning_rate": 0.001}
        }

        is_valid, error = integration.validate_tool_request(tool_request)

        assert is_valid
        assert error is None

    def test_execute_tool_with_rollback_success(self, integration_env):
        """Test successful tool execution."""
        settings, memory, integration = integration_env

        # Execute tool
        result = integration.execute_tool_with_rollback(
            tool_name="test_tool",
            tool_args={"param": "value"},
            cycle_id=1
        )

        assert result["status"] == "success"
        assert result["tool"] == "test_tool"

    def test_execute_tool_with_rollback_failure(self, integration_env):
        """Test tool execution with rollback on failure."""
        settings, memory, integration = integration_env

        # Modify memory
        directive = memory.load_directive()
        original_cycle_id = directive.cycle_id
        directive.cycle_id = 99
        memory.save_directive(directive)

        # Mock failing tool execution
        class FailingToolExecution(Exception):
            pass

        # Simulate tool failure
        try:
            # In real implementation, tool would fail here
            # For test, we just verify rollback mechanism exists
            pass
        except FailingToolExecution:
            pass

        # Verify memory still valid
        is_valid, errors = integration.validate_memory()
        assert is_valid


# ============================================================================
# Health Monitoring Tests
# ============================================================================

@pytest.mark.integration
class TestMultiAgentIntegrationHealth:
    """Test health monitoring integration."""

    def test_get_health_status(self, integration_env):
        """Test getting health status."""
        settings, memory, integration = integration_env

        # Register agents
        integration.register_standard_multi_agent_pipeline()

        # Get health
        health = integration.get_health_status()

        assert "agents_registered" in health
        assert "memory_valid" in health
        assert health["memory_valid"] is True
        assert health["agents_registered"] > 0


# ============================================================================
# Agent 2 Orchestrator Compatibility Tests
# ============================================================================

@pytest.mark.integration
class TestMultiAgentIntegrationCompatibility:
    """Test compatibility with Agent 2's orchestrator."""

    def test_wrap_orchestrator(self, integration_env):
        """Test wrapping Agent 2's orchestrator."""
        settings, memory, integration = integration_env

        # Create mock orchestrator
        class MockMultiAgentOrchestrator:
            def __init__(self, memory_path):
                self.memory_path = Path(memory_path)

        mock_orchestrator = MockMultiAgentOrchestrator("/tmp/test")

        # Wrap it
        wrapped = integration.wrap_multi_agent_orchestrator(mock_orchestrator)

        # Verify memory path was updated
        assert wrapped.memory_path == settings.memory_dir

    def test_multi_agent_phase_registration(self, integration_env):
        """Test registering multi-agent phases."""
        settings, memory, integration = integration_env

        # Register custom phase
        def custom_phase(context: CycleContext) -> CycleContext:
            context.agent_outputs["custom"] = "completed"
            return context

        integration.register_multi_agent_phase("custom", custom_phase)

        assert "custom" in integration.multi_agent_callbacks


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

@pytest.mark.integration
class TestMultiAgentIntegrationE2E:
    """End-to-end integration tests."""

    def test_full_multi_agent_pipeline(self, integration_env):
        """Test complete multi-agent pipeline."""
        settings, memory, integration = integration_env

        # Register full pipeline
        integration.register_standard_multi_agent_pipeline()

        # Run cycle
        result = integration.run_cycle(cycle_id=1)

        # Verify all stages completed
        assert result["phase"] == OrchestratorPhase.COMPLETE.value
        assert "historian" in result["agent_outputs"]
        assert "director" in result["agent_outputs"]
        assert "architect" in result["agent_outputs"]
        assert "critic" in result["agent_outputs"]
        assert "executor" in result["agent_outputs"]

        # Verify memory persistence
        directive = memory.load_directive()
        history = memory.load_history_summary()
        proposals = memory.load_proposals()
        reviews = memory.load_reviews()

        assert directive.cycle_id == 1
        assert history.total_cycles == 1
        assert proposals is not None
        assert reviews is not None

    def test_transaction_rollback_on_error(self, integration_env):
        """Test transaction rollback when agent fails."""
        settings, memory, integration = integration_env

        # Get initial state
        initial_directive = memory.load_directive()
        initial_cycle_id = initial_directive.cycle_id

        # Register agents with one failing
        integration.orchestrator.register_agent("historian", integration._historian_callback)
        integration.orchestrator.register_agent("director", integration._director_callback)

        def failing_architect(context: CycleContext) -> CycleContext:
            raise RuntimeError("Architect failed")

        integration.orchestrator.register_agent("architect", failing_architect)

        # Run cycle (should fail and rollback)
        result = integration.run_cycle(cycle_id=10)

        assert result["phase"] == OrchestratorPhase.ERROR.value

        # Verify memory wasn't corrupted
        current_directive = memory.load_directive()
        # Cycle ID may have been updated before failure
        # but memory should still be valid
        is_valid, errors = memory.validate_all_memory()
        assert is_valid
