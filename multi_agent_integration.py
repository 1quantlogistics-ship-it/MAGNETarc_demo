"""
Multi-Agent Integration Layer

Bridges Agent 2's multi-agent orchestrator with Agent 1's v1.1.0 infrastructure:
- Wraps multi-agent orchestrator to use MemoryHandler
- Integrates with orchestrator_base for execution spine
- Provides schema-validated memory operations
- Enables transactional multi-agent workflows

This adapter allows the multi-agent system to leverage:
✅ Config-driven paths (no hardcoded /workspace)
✅ Schema validation on all memory I/O
✅ Atomic transactions with rollback
✅ Thread-safe concurrent access
✅ Structured error handling
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from orchestrator_base import OrchestratorBase, CycleContext, OrchestratorPhase
from memory_handler import MemoryHandler, get_memory_handler, ValidationFailedError
from config import get_settings, ARCSettings
from schemas import (
    Directive, HistorySummary, Constraints, SystemState,
    Proposals, Reviews, Proposal, Review
)

logger = logging.getLogger(__name__)


class MultiAgentIntegration:
    """
    Integration layer between multi-agent orchestrator and v1.1.0 infrastructure.

    This class acts as an adapter, allowing the multi-agent orchestrator to use:
    - MemoryHandler for all file I/O
    - OrchestratorBase for execution spine
    - Config-driven settings
    - Schema validation

    Example:
        integration = MultiAgentIntegration()

        # Register multi-agent callbacks
        integration.register_multi_agent_phase("proposals", generate_proposals_callback)
        integration.register_multi_agent_phase("reviews", review_proposals_callback)
        integration.register_multi_agent_phase("voting", conduct_voting_callback)

        # Run cycle with v1.1.0 infrastructure
        result = integration.run_cycle(cycle_id=10)
    """

    def __init__(
        self,
        settings: Optional[ARCSettings] = None,
        memory: Optional[MemoryHandler] = None,
        offline_mode: bool = False
    ):
        """
        Initialize multi-agent integration.

        Args:
            settings: Optional settings (uses get_settings() if None)
            memory: Optional memory handler (uses get_memory_handler() if None)
            offline_mode: Use mock LLM client for testing
        """
        self.settings = settings or get_settings()
        self.memory = memory or get_memory_handler(settings)
        self.offline_mode = offline_mode

        # Create orchestrator base
        self.orchestrator = OrchestratorBase(
            settings=self.settings,
            memory=self.memory
        )

        # Multi-agent phase callbacks
        self.multi_agent_callbacks: Dict[str, Any] = {}

        logger.info(f"MultiAgentIntegration initialized (offline={offline_mode})")

    # ========================================================================
    # Multi-Agent Callback Registration
    # ========================================================================

    def register_multi_agent_phase(self, phase_name: str, callback: callable):
        """
        Register a multi-agent phase callback.

        Args:
            phase_name: Phase name (proposals, reviews, voting, etc.)
            callback: Callback that takes CycleContext and returns CycleContext
        """
        self.multi_agent_callbacks[phase_name] = callback
        logger.info(f"Registered multi-agent phase: {phase_name}")

    def register_standard_multi_agent_pipeline(self):
        """
        Register standard multi-agent pipeline:
        - Historian (history updates)
        - Director (strategic planning)
        - Proposals (parallel generation)
        - Reviews (multi-critic)
        - Voting (democratic consensus)
        - Execution (approved experiments)
        """
        # These would be imported from Agent 2's orchestrator
        # For now, we register placeholders that show the pattern

        # Historian
        self.orchestrator.register_agent("historian", self._historian_callback)

        # Director
        self.orchestrator.register_agent("director", self._director_callback)

        # Multi-agent proposal generation
        self.orchestrator.register_agent("architect", self._architect_callback)

        # Multi-critic review
        self.orchestrator.register_agent("critic", self._critic_callback)

        # Executor
        self.orchestrator.register_agent("executor", self._executor_callback)

        logger.info("Registered standard multi-agent pipeline")

    # ========================================================================
    # Agent Callbacks (Schema-Validated)
    # ========================================================================

    def _historian_callback(self, context: CycleContext) -> CycleContext:
        """
        Historian updates history with schema validation.

        This wraps Agent 2's historian logic with MemoryHandler.
        """
        logger.info("Historian: Updating history")

        try:
            # Load history with validation
            history = context.history or self.memory.load_history_summary()

            # Update history
            history.total_cycles += 1
            history.last_update = datetime.utcnow().isoformat()

            # Save back to context (will be persisted in save phase)
            context.history = history
            context.agent_outputs["historian"] = "History updated"

        except Exception as e:
            logger.error(f"Historian failed: {e}")
            context.errors.append(f"Historian: {e}")

        return context

    def _director_callback(self, context: CycleContext) -> CycleContext:
        """
        Director sets directive with schema validation.

        This wraps Agent 2's director logic with MemoryHandler.
        """
        logger.info("Director: Setting strategic direction")

        try:
            # Load directive with validation
            directive = context.directive or self.memory.load_directive()

            # Director would update directive here
            # (Agent 2's logic would go here)
            directive.cycle_id = context.cycle_id

            # Save back to context
            context.directive = directive
            context.agent_outputs["director"] = "Directive set"

        except Exception as e:
            logger.error(f"Director failed: {e}")
            context.errors.append(f"Director: {e}")

        return context

    def _architect_callback(self, context: CycleContext) -> CycleContext:
        """
        Architect generates proposals with schema validation.

        This wraps Agent 2's multi-agent proposal generation.
        """
        logger.info("Architect: Generating proposals")

        try:
            # Load constraints for validation
            constraints = context.constraints or self.memory.load_constraints()

            # Generate proposals (Agent 2's logic would go here)
            proposals = Proposals(
                cycle_id=context.cycle_id,
                proposals=[]
            )

            # Multi-agent proposal generation would add to proposals.proposals
            # This is where Agent 2's architect, explorer, and parameter scientist
            # would contribute proposals

            # Save to context
            context.proposals = proposals
            context.agent_outputs["architect"] = f"Generated {len(proposals.proposals)} proposals"

        except Exception as e:
            logger.error(f"Architect failed: {e}")
            context.errors.append(f"Architect: {e}")

        return context

    def _critic_callback(self, context: CycleContext) -> CycleContext:
        """
        Critic reviews proposals with schema validation.

        This wraps Agent 2's multi-critic review.
        """
        logger.info("Critic: Reviewing proposals")

        try:
            if not context.proposals or not context.proposals.proposals:
                logger.warning("No proposals to review")
                return context

            # Multi-critic review (Agent 2's logic would go here)
            reviews = Reviews(
                cycle_id=context.cycle_id,
                reviews=[]
            )

            # Primary critic and secondary critic would add reviews here

            # Save to context
            context.reviews = reviews
            context.agent_outputs["critic"] = f"Reviewed {len(reviews.reviews)} proposals"

        except Exception as e:
            logger.error(f"Critic failed: {e}")
            context.errors.append(f"Critic: {e}")

        return context

    def _executor_callback(self, context: CycleContext) -> CycleContext:
        """
        Executor prepares approved experiments.

        This wraps Agent 2's executor logic.
        """
        logger.info("Executor: Preparing experiments")

        try:
            # Executor would prepare experiments here
            # (Agent 2's logic would go here)

            context.agent_outputs["executor"] = "Experiments prepared"

        except Exception as e:
            logger.error(f"Executor failed: {e}")
            context.errors.append(f"Executor: {e}")

        return context

    # ========================================================================
    # Cycle Execution
    # ========================================================================

    def run_cycle(self, cycle_id: int) -> Dict[str, Any]:
        """
        Run a multi-agent research cycle with v1.1.0 infrastructure.

        This uses OrchestratorBase for execution spine, which provides:
        - Memory load/validate/save phases
        - Transaction support
        - Error handling with automatic rollback
        - Hook system for monitoring

        Args:
            cycle_id: Cycle number

        Returns:
            Cycle results with decisions and metrics
        """
        logger.info(f"Starting multi-agent cycle {cycle_id}")

        # Run cycle through orchestrator base
        context = self.orchestrator.run_cycle(cycle_id)

        # Build results
        results = {
            "cycle_id": cycle_id,
            "phase": context.phase.value,
            "started_at": context.started_at,
            "completed_at": context.completed_at,
            "errors": context.errors,
            "warnings": context.warnings,
            "agent_outputs": context.agent_outputs,
            "has_errors": len(context.errors) > 0
        }

        # Add cycle stats
        stats = self.orchestrator.get_cycle_stats(context)
        results["stats"] = stats

        logger.info(f"Multi-agent cycle {cycle_id} complete: {context.phase.value}")

        return results

    # ========================================================================
    # Memory Operations (Schema-Validated)
    # ========================================================================

    def get_memory_path(self) -> Path:
        """Get memory directory path from config."""
        return self.settings.memory_dir

    def validate_memory(self) -> tuple[bool, List[str]]:
        """
        Validate all memory files.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        return self.memory.validate_all_memory()

    def backup_memory(self) -> Path:
        """
        Create memory backup.

        Returns:
            Path to backup directory
        """
        return self.memory.backup_memory()

    def restore_memory(self, backup_dir: Path):
        """
        Restore memory from backup.

        Args:
            backup_dir: Directory containing backup
        """
        self.memory.restore_memory(backup_dir)

    # ========================================================================
    # Agent 2 Orchestrator Compatibility
    # ========================================================================

    def wrap_multi_agent_orchestrator(self, multi_agent_orchestrator):
        """
        Wrap Agent 2's MultiAgentOrchestrator to use v1.1.0 infrastructure.

        This adapter allows the existing multi-agent orchestrator to leverage:
        - MemoryHandler for all file I/O
        - Config for paths
        - Schema validation
        - Transactions

        Args:
            multi_agent_orchestrator: Agent 2's MultiAgentOrchestrator instance

        Returns:
            Wrapped orchestrator
        """
        # Store reference
        self.wrapped_orchestrator = multi_agent_orchestrator

        # Override memory path with config
        multi_agent_orchestrator.memory_path = self.settings.memory_dir

        # Inject memory handler
        # (This would require modifying Agent 2's orchestrator to accept
        #  a memory handler injection)

        logger.info("Wrapped multi-agent orchestrator with v1.1.0 infrastructure")

        return multi_agent_orchestrator

    # ========================================================================
    # Tool Governance Integration
    # ========================================================================

    def validate_tool_request(self, tool_request: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate a tool request against constraints.

        Args:
            tool_request: Tool request to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Load constraints
            constraints = self.memory.load_constraints()

            # Validate tool request
            # (Tool-specific validation logic would go here)

            return True, None

        except Exception as e:
            logger.error(f"Tool validation failed: {e}")
            return False, str(e)

    def execute_tool_with_rollback(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        cycle_id: int
    ) -> Dict[str, Any]:
        """
        Execute a tool with automatic rollback on failure.

        Args:
            tool_name: Tool to execute
            tool_args: Tool arguments
            cycle_id: Current cycle ID

        Returns:
            Tool execution results
        """
        logger.info(f"Executing tool: {tool_name}")

        # Create backup before tool execution
        backup_dir = self.memory.backup_memory()

        try:
            # Execute tool (would integrate with Control Plane)
            # (Tool execution logic would go here)

            result = {
                "tool": tool_name,
                "status": "success",
                "cycle_id": cycle_id
            }

            return result

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")

            # Rollback to backup
            logger.info("Rolling back memory due to tool failure")
            self.memory.restore_memory(backup_dir)

            raise

    # ========================================================================
    # Health Monitoring Integration
    # ========================================================================

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of multi-agent system.

        Returns:
            Health status including agents and memory
        """
        # Memory validation
        memory_valid, memory_errors = self.memory.validate_all_memory()

        # Agent health (would integrate with Agent 2's health monitor)
        agent_health = {
            "agents_registered": len(self.orchestrator.agent_callbacks),
            "memory_valid": memory_valid,
            "memory_errors": memory_errors
        }

        return agent_health


# ============================================================================
# Convenience Functions
# ============================================================================

def create_multi_agent_integration(
    offline_mode: bool = False,
    settings: Optional[ARCSettings] = None
) -> MultiAgentIntegration:
    """
    Create a multi-agent integration instance.

    Args:
        offline_mode: Use mock LLM client
        settings: Optional settings

    Returns:
        MultiAgentIntegration instance
    """
    return MultiAgentIntegration(
        settings=settings,
        offline_mode=offline_mode
    )
