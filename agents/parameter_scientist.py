"""
ParameterScientistAgent: Hyperparameter optimization specialist
================================================================

The Parameter Scientist uses advanced optimization strategies to
efficiently search hyperparameter space and recommend optimal configs.
"""

from typing import Dict, Any, List
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter


class ParameterScientistAgent(BaseAgent):
    """
    Hyperparameter optimization agent.

    Responsibilities:
    - Apply advanced optimization strategies (Bayesian, TPE, CMA-ES)
    - Model parameter-performance relationships
    - Recommend optimal hyperparameter configurations
    - Prune unpromising parameter regions
    - Guide efficient search strategies
    """

    def __init__(
        self,
        agent_id: str = "parameter_scientist_001",
        model: str = "deepseek-r1",
        llm_router: LLMRouter = None,
        voting_weight: float = 1.5,
        memory_path: str = "/workspace/arc/memory"
    ):
        """Initialize Parameter Scientist agent."""
        super().__init__(
            agent_id=agent_id,
            role="parameter_scientist",
            model=model,
            capabilities=[AgentCapability.PROPOSAL_GENERATION, AgentCapability.EXPLORATION],
            voting_weight=voting_weight,
            priority="medium",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate hyperparameter optimization proposals.

        Args:
            input_data: Contains optimization strategy and history

        Returns:
            Optimized hyperparameter proposals
        """
        import time
        start_time = time.time()

        try:
            # Read memory
            directive = self.read_memory("directive.json")
            history = self.read_memory("history_summary.json")
            constraints = self.read_memory("constraints.json")

            # Build prompt
            prompt = self._build_optimization_prompt(directive, history, constraints)

            # Get LLM client
            client = self.llm_router.get_client_for_role(self.role)

            # Generate optimization proposals
            response = client.generate_json(prompt, max_tokens=2500, temperature=0.7)

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("optimize_parameters", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("optimize_parameters", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate proposal based on hyperparameter optimization theory.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision based on optimization principles
        """
        # Read history to check parameter performance relationships
        history = self.read_memory("history_summary.json")

        config_changes = proposal.get("config_changes", {})
        successful_patterns = history.get("successful_patterns", [])

        # Check if proposal aligns with successful patterns
        alignment_score = 0.0
        for pattern in successful_patterns:
            matches = sum(
                1 for k, v in config_changes.items()
                if k in pattern and self._in_range(v, pattern.get(k))
            )
            if matches > 0:
                alignment_score += matches / len(config_changes)

        if alignment_score > 0.5:
            return {
                "decision": "approve",
                "confidence": 0.85,
                "reasoning": f"Proposal aligns with successful parameter patterns (score: {alignment_score:.2f})"
            }
        else:
            return {
                "decision": "approve",
                "confidence": 0.6,
                "reasoning": "Proposal explores new parameter regions (uncertain outcome)"
            }

    def _in_range(self, value, pattern_value) -> bool:
        """Check if value is in range of pattern value (simple heuristic)."""
        if isinstance(pattern_value, list) and len(pattern_value) == 2:
            return pattern_value[0] <= value <= pattern_value[1]
        elif isinstance(pattern_value, (int, float)):
            return abs(value - pattern_value) / max(abs(pattern_value), 1) < 0.2
        return False

    def _build_optimization_prompt(
        self,
        directive: Dict[str, Any],
        history: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Build prompt for hyperparameter optimization."""
        return f"""You are the Parameter Scientist agent in ARC (Autonomous Research Collective).
Your role is to apply advanced hyperparameter optimization strategies.

# Strategic Directive
{directive}

# Research History (with parameter-performance data)
{history}

# Safety Constraints
{constraints}

# Your Task
Generate hyperparameter optimization proposals using:
1. Bayesian optimization (model parameter-performance relationship)
2. Surrogate modeling (predict promising regions)
3. Acquisition functions (balance exploration-exploitation)
4. Smart parameter space pruning

Analyze past experiments to identify:
- High-performing parameter regions
- Parameter interactions and dependencies
- Diminishing returns regions (to prune)

Return ONLY a valid JSON object:
{{
  "proposals": [
    {{
      "experiment_id": "exp_param_XXX",
      "name": "descriptive_name",
      "hypothesis": "parameter optimization hypothesis",
      "novelty_category": "exploit" | "explore",
      "optimization_strategy": "bayesian" | "tpe" | "surrogate" | "gradient",
      "predicted_metrics": {{"auc": 0.XX, "sensitivity": 0.XX, "specificity": 0.XX}},
      "config_changes": {{"param1": value1, "param2": value2}},
      "justification": "optimization rationale",
      "expected_improvement": 0.XX,
      "confidence_interval": [lower, upper]
    }}
  ],
  "pruned_regions": [
    {{"parameter": "X", "range": [min, max], "reason": "diminishing returns"}}
  ]
}}"""
