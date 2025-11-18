"""
DirectorAgent: Strategic planning and mode control
===================================================

The Director sets research strategy, allocates budgets, and determines
when to explore vs exploit.
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
