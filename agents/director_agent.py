"""
DirectorAgent: Strategic planning and mode control
===================================================

The Director sets research strategy, allocates budgets, and determines
when to explore vs exploit.

Phase E Enhancements:
- Curriculum learning control (Task 2.5)
- Adaptive curriculum pacing based on performance
- Curriculum stage progression decisions
"""

from typing import Dict, Any, List, Optional
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter

# Optional Historian integration for adaptive strategy
try:
    from agents.historian_agent import HistorianAgent
    HISTORIAN_AVAILABLE = True
except ImportError:
    HISTORIAN_AVAILABLE = False

# Phase E: Curriculum strategy support (Task 2.5)
try:
    from schemas.curriculum_strategy import (
        CurriculumStrategy, PacingStrategy, DifficultyMetric
    )
    CURRICULUM_STRATEGY_AVAILABLE = True
except ImportError:
    CURRICULUM_STRATEGY_AVAILABLE = False


class DirectorAgent(BaseAgent):
    """
    Strategic planning agent.

    Responsibilities:
    - Set research mode (explore/exploit/recover)
    - Allocate novelty budgets
    - Define strategic objectives
    - Detect stagnation and regression
    """

    def __init__(
        self,
        agent_id: str = "director_001",
        model: str = "claude-sonnet-4.5",
        llm_router: LLMRouter = None,
        voting_weight: float = 2.0,
        memory_path: str = "/workspace/arc/memory"
    ):
        """Initialize Director agent."""
        super().__init__(
            agent_id=agent_id,
            role="director",
            model=model,
            capabilities=[AgentCapability.STRATEGY, AgentCapability.PROPOSAL_GENERATION],
            voting_weight=voting_weight,
            priority="high",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate strategic directive.

        Args:
            input_data: Contains history_summary and constraints

        Returns:
            Strategic directive (mode, budget, objectives)
        """
        import time
        start_time = time.time()

        try:
            # Read memory
            history = self.read_memory("history_summary.json")
            constraints = self.read_memory("constraints.json")

            # Build prompt
            prompt = self._build_directive_prompt(history, constraints)

            # Get LLM client
            client = self.llm_router.get_client_for_role(self.role)

            # Generate directive
            response = client.generate_json(prompt, max_tokens=1500, temperature=0.7)

            # Write to memory
            self.write_memory("directive.json", response)

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("generate_directive", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("generate_directive", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Directors can override proposals if they conflict with strategy.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision
        """
        # Read current directive
        directive = self.read_memory("directive.json")

        # Check alignment with strategy
        mode = directive.get("mode", "explore")
        novelty_category = proposal.get("novelty_category", "explore")

        # Simple alignment check
        if mode == "exploit" and novelty_category == "wildcat":
            return {
                "decision": "reject",
                "confidence": 0.9,
                "reasoning": "Wildcat experiments conflict with exploit mode strategy",
                "suggested_changes": {"novelty_category": "exploit"}
            }

        return {
            "decision": "approve",
            "confidence": 0.8,
            "reasoning": "Proposal aligns with strategic directive"
        }

    def _build_directive_prompt(
        self,
        history: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Build prompt for directive generation."""
        return f"""You are the Director agent in ARC (Autonomous Research Collective).
Your role is to set strategic direction for the research process.

# Current Research History
{history}

# Safety Constraints
{constraints}

# Your Task
Generate a strategic directive that includes:
1. Research mode (explore/exploit/recover)
2. Novelty budget allocation (exploit/explore/wildcat counts)
3. Strategic objective
4. Focus areas for experiments
5. Forbidden/encouraged parameter axes

Return ONLY a valid JSON object with this structure:
{{
  "mode": "explore" | "exploit" | "recover",
  "objective": "brief description",
  "novelty_budget": {{"exploit": N, "explore": M, "wildcat": K}},
  "focus_areas": ["area1", "area2"],
  "forbidden_axes": ["param1", "param2"],
  "encouraged_axes": ["param3", "param4"],
  "notes": "strategic reasoning"
}}"""

    def compute_adaptive_strategy(
        self,
        historian: Optional['HistorianAgent'] = None,
        stagnation_threshold: float = 0.01,
        regression_threshold: float = -0.05,
        window: int = 5
    ) -> Dict[str, Any]:
        """
        Compute strategy based on algorithmic analysis of performance trends.

        This replaces pure LLM-based strategy with data-driven decision-making.

        Args:
            historian: Historian agent with training history access
            stagnation_threshold: Min improvement required to avoid stagnation mode
            regression_threshold: Performance drop that triggers recovery
            window: Number of recent experiments to analyze

        Returns:
            Strategic directive with mode and reasoning
        """
        # Load history
        history = self.read_memory("history_summary.json") or {}
        training_history = self.read_memory("training_history.json") or {}

        # Extract metrics
        total_cycles = history.get("total_cycles", 0)
        best_metrics = history.get("best_metrics", {})
        recent_experiments = history.get("recent_experiments", [])

        # Default strategy (no history)
        if total_cycles < 3 or not recent_experiments:
            return {
                "mode": "explore",
                "objective": "Initial exploration - building knowledge base",
                "novelty_budget": {"exploit": 1, "explore": 2, "wildcat": 0},
                "reasoning": "Insufficient history for adaptive strategy",
                "strategy_type": "default"
            }

        # Analyze performance trend
        if historian and HISTORIAN_AVAILABLE:
            trend = historian.get_performance_trend(metric="auc", window=window)
            stagnated = historian.detect_stagnation(
                metric="auc",
                threshold=stagnation_threshold,
                window=window
            )
        else:
            # Manual trend analysis
            trend = []
            for exp in recent_experiments[-window:]:
                metrics = exp.get("metrics", {})
                if "auc" in metrics:
                    trend.append(metrics["auc"])

            stagnated = self._detect_stagnation_simple(trend, stagnation_threshold)

        # Compute improvement
        improvement = 0.0
        if len(trend) >= 2:
            improvement = trend[-1] - trend[0]

        # Detect regression
        regressed = improvement < regression_threshold

        # Choose strategy based on analysis
        if regressed:
            # Performance dropped significantly → RECOVER
            return {
                "mode": "recover",
                "objective": "Recover from performance regression",
                "novelty_budget": {"exploit": 3, "explore": 0, "wildcat": 0},
                "focus_areas": ["proven_configs", "baseline_restoration"],
                "reasoning": f"Performance regressed by {improvement:.3f} (threshold: {regression_threshold})",
                "strategy_type": "algorithmic_recovery",
                "metrics": {
                    "improvement": improvement,
                    "trend": trend,
                    "regressed": True
                }
            }

        elif stagnated:
            # No improvement for several cycles → EXPLORE
            return {
                "mode": "explore",
                "objective": "Break stagnation with novel approaches",
                "novelty_budget": {"exploit": 0, "explore": 2, "wildcat": 1},
                "focus_areas": ["unexplored_regions", "alternative_architectures"],
                "reasoning": f"Stagnation detected: improvement {improvement:.3f} < {stagnation_threshold}",
                "strategy_type": "algorithmic_exploration",
                "metrics": {
                    "improvement": improvement,
                    "trend": trend,
                    "stagnated": True
                }
            }

        elif improvement > 0.05:
            # Strong recent improvement → EXPLOIT
            return {
                "mode": "exploit",
                "objective": "Exploit recent breakthroughs",
                "novelty_budget": {"exploit": 3, "explore": 0, "wildcat": 0},
                "focus_areas": ["successful_patterns", "fine_tuning"],
                "reasoning": f"Strong improvement {improvement:.3f} - double down on success",
                "strategy_type": "algorithmic_exploitation",
                "metrics": {
                    "improvement": improvement,
                    "trend": trend,
                    "strong_progress": True
                }
            }

        else:
            # Moderate progress → BALANCED EXPLORE
            return {
                "mode": "explore",
                "objective": "Balanced exploration with moderate progress",
                "novelty_budget": {"exploit": 1, "explore": 2, "wildcat": 0},
                "focus_areas": ["incremental_improvements", "nearby_configs"],
                "reasoning": f"Moderate progress {improvement:.3f} - continue exploring",
                "strategy_type": "algorithmic_balanced",
                "metrics": {
                    "improvement": improvement,
                    "trend": trend
                }
            }

    def _detect_stagnation_simple(
        self,
        trend: List[float],
        threshold: float
    ) -> bool:
        """
        Simple stagnation detection without Historian.

        Args:
            trend: List of recent metric values
            threshold: Min improvement required

        Returns:
            True if stagnated
        """
        if len(trend) < 2:
            return False

        # Check if improvement from first to last is below threshold
        improvement = trend[-1] - trend[0]
        return improvement < threshold

    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get current strategy summary."""
        directive = self.read_memory("directive.json") or {}

        return {
            "mode": directive.get("mode", "unknown"),
            "objective": directive.get("objective", ""),
            "novelty_budget": directive.get("novelty_budget", {}),
            "strategy_type": directive.get("strategy_type", "llm_based"),
            "metrics": directive.get("metrics", {})
        }

    def decide_curriculum_progression(
        self,
        curriculum: CurriculumStrategy,
        current_stage_id: int,
        current_epoch: int,
        validation_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Decide whether to progress to next curriculum stage.

        Phase E: Task 2.5 - Director controls curriculum learning progression
        based on performance and pacing strategy.

        Args:
            curriculum: Active curriculum strategy
            current_stage_id: Current curriculum stage (0-indexed)
            current_epoch: Current epoch within stage
            validation_metrics: Recent validation metrics (auc, sensitivity, specificity)

        Returns:
            Decision dict with progression action and reasoning
        """
        if not CURRICULUM_STRATEGY_AVAILABLE:
            raise RuntimeError("Curriculum strategy schema not available")

        # Get current stage
        if current_stage_id >= len(curriculum.stages):
            return {
                "action": "complete",
                "reasoning": "Curriculum complete - all stages finished",
                "next_stage_id": None
            }

        current_stage = curriculum.stages[current_stage_id]

        # Check 1: Minimum epochs requirement
        if current_epoch < current_stage.min_epochs:
            return {
                "action": "continue",
                "reasoning": f"Stage {current_stage_id} requires min {current_stage.min_epochs} epochs (currently {current_epoch})",
                "next_stage_id": current_stage_id
            }

        # Check 2: Maximum epochs limit (if set)
        if current_stage.max_epochs and current_epoch >= current_stage.max_epochs:
            return {
                "action": "progress",
                "reasoning": f"Stage {current_stage_id} max epochs ({current_stage.max_epochs}) reached",
                "next_stage_id": current_stage_id + 1
            }

        # Check 3: Progression criterion (if set)
        if current_stage.progression_criterion:
            criterion_met = True
            for metric_name, threshold in current_stage.progression_criterion.items():
                actual_value = validation_metrics.get(metric_name, 0.0)

                if actual_value < threshold:
                    criterion_met = False
                    break

            if criterion_met:
                return {
                    "action": "progress",
                    "reasoning": f"Stage {current_stage_id} progression criteria met: {current_stage.progression_criterion}",
                    "next_stage_id": current_stage_id + 1,
                    "metrics": validation_metrics
                }
            else:
                return {
                    "action": "continue",
                    "reasoning": f"Stage {current_stage_id} progression criteria not met yet",
                    "next_stage_id": current_stage_id,
                    "metrics": validation_metrics
                }

        # Check 4: Clinical safety constraint (sensitivity threshold)
        sensitivity = validation_metrics.get("sensitivity", 0.0)
        if sensitivity < curriculum.min_sensitivity_threshold:
            return {
                "action": "halt",
                "reasoning": f"Sensitivity ({sensitivity:.3f}) below minimum threshold ({curriculum.min_sensitivity_threshold:.3f})",
                "next_stage_id": current_stage_id,
                "safety_violation": True
            }

        # Default: Continue current stage (no criteria specified)
        return {
            "action": "continue",
            "reasoning": f"Stage {current_stage_id} in progress (no progression criteria)",
            "next_stage_id": current_stage_id
        }

    def compute_curriculum_pacing(
        self,
        curriculum: CurriculumStrategy,
        total_epochs_planned: int
    ) -> List[int]:
        """
        Compute epoch boundaries for each curriculum stage based on pacing strategy.

        Phase E: Task 2.5 - Translate pacing strategy into concrete epoch schedule.

        Args:
            curriculum: Curriculum strategy with pacing config
            total_epochs_planned: Total training epochs planned

        Returns:
            List of epoch boundaries for each stage (e.g., [0, 5, 15, 30] for 3 stages)
        """
        if not CURRICULUM_STRATEGY_AVAILABLE:
            raise RuntimeError("Curriculum strategy schema not available")

        num_stages = len(curriculum.stages)
        pacing_strategy = curriculum.pacing_strategy
        pacing_params = curriculum.pacing_params or {}

        boundaries = [0]  # Start at epoch 0

        if pacing_strategy == PacingStrategy.LINEAR:
            # Equal epochs per stage
            epochs_per_stage = total_epochs_planned // num_stages

            for i in range(1, num_stages):
                boundaries.append(i * epochs_per_stage)

            boundaries.append(total_epochs_planned)

        elif pacing_strategy == PacingStrategy.EXPONENTIAL:
            # Accelerating progression (more epochs in later stages)
            import math

            for i in range(1, num_stages):
                # Exponential growth
                fraction = (math.exp(i / num_stages) - 1) / (math.e - 1)
                boundary = int(fraction * total_epochs_planned)
                boundaries.append(boundary)

            boundaries.append(total_epochs_planned)

        elif pacing_strategy == PacingStrategy.ROOT:
            # Decelerating progression (more epochs in early stages)
            import math

            for i in range(1, num_stages):
                # Square root growth
                fraction = math.sqrt(i / num_stages)
                boundary = int(fraction * total_epochs_planned)
                boundaries.append(boundary)

            boundaries.append(total_epochs_planned)

        elif pacing_strategy == PacingStrategy.STEP:
            # Discrete jumps at predefined epochs
            step_epochs = pacing_params.get("step_epochs", [])

            if len(step_epochs) != num_stages - 1:
                raise ValueError(
                    f"Step pacing requires {num_stages - 1} step_epochs, got {len(step_epochs)}"
                )

            boundaries.extend(step_epochs)
            boundaries.append(total_epochs_planned)

        elif pacing_strategy == PacingStrategy.ADAPTIVE:
            # Adaptive pacing (boundaries computed at runtime based on performance)
            # Return minimum epochs per stage as placeholder
            for stage in curriculum.stages:
                boundaries.append(boundaries[-1] + stage.min_epochs)

        else:
            # Default to linear
            epochs_per_stage = total_epochs_planned // num_stages
            for i in range(1, num_stages + 1):
                boundaries.append(i * epochs_per_stage)

        return boundaries[:num_stages + 1]  # Ensure correct length
