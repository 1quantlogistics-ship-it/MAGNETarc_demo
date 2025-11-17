"""
ExplorerAgent: Parameter space exploration specialist
======================================================

The Explorer systematically explores the parameter space to discover
high-potential regions and identify boundaries.
"""

from typing import Dict, Any, List
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter


class ExplorerAgent(BaseAgent):
    """
    Parameter space exploration agent.

    Responsibilities:
    - Systematic parameter space exploration
    - Identify high-potential regions
    - Map parameter boundaries
    - Generate exploratory proposals
    - Use exploration strategies (grid, random, bayesian)
    """

    def __init__(
        self,
        agent_id: str = "explorer_001",
        model: str = "qwen2.5-32b",
        llm_router: LLMRouter = None,
        voting_weight: float = 1.2,
        memory_path: str = "/workspace/arc/memory"
    ):
        """Initialize Explorer agent."""
        super().__init__(
            agent_id=agent_id,
            role="explorer",
            model=model,
            capabilities=[AgentCapability.EXPLORATION, AgentCapability.PROPOSAL_GENERATION],
            voting_weight=voting_weight,
            priority="medium",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate exploratory experiment proposals.

        Args:
            input_data: Contains exploration strategy and bounds

        Returns:
            List of exploratory proposals
        """
        import time
        start_time = time.time()

        try:
            # Read memory
            directive = self.read_memory("directive.json")
            history = self.read_memory("history_summary.json")
            constraints = self.read_memory("constraints.json")

            # Build prompt
            prompt = self._build_exploration_prompt(directive, history, constraints)

            # Get LLM client
            client = self.llm_router.get_client_for_role(self.role)

            # Generate exploration proposals
            response = client.generate_json(prompt, max_tokens=2500, temperature=0.9)

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("generate_exploration", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("generate_exploration", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explorers favor novel parameter combinations.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision favoring novelty
        """
        novelty_category = proposal.get("novelty_category", "exploit")

        # Explorers prefer explore/wildcat over exploit
        if novelty_category in ["explore", "wildcat"]:
            return {
                "decision": "approve",
                "confidence": 0.9,
                "reasoning": "Proposal supports parameter space exploration"
            }
        else:
            return {
                "decision": "approve",
                "confidence": 0.6,
                "reasoning": "Exploit experiments have limited exploration value"
            }

    def _build_exploration_prompt(
        self,
        directive: Dict[str, Any],
        history: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Build prompt for exploration."""
        return f"""You are the Explorer agent in ARC (Autonomous Research Collective).
Your role is to systematically explore the parameter space to discover optimal regions.

# Strategic Directive
{directive}

# Research History
{history}

# Safety Constraints
{constraints}

# Your Task
Generate exploratory experiment proposals that:
1. Systematically sample unexplored parameter regions
2. Test parameter boundaries (within safety constraints)
3. Use diverse exploration strategies (grid, random, bayesian)
4. Maximize information gain about parameter space

Focus on NOVELTY and COVERAGE, not incremental improvements.

Return ONLY a valid JSON object:
{{
  "proposals": [
    {{
      "experiment_id": "exp_explore_XXX",
      "name": "descriptive_name",
      "hypothesis": "exploration hypothesis",
      "novelty_category": "explore" | "wildcat",
      "exploration_strategy": "grid" | "random" | "bayesian" | "boundary",
      "predicted_metrics": {{"auc": 0.XX, "sensitivity": 0.XX, "specificity": 0.XX}},
      "config_changes": {{"param1": value1, "param2": value2}},
      "justification": "exploration rationale",
      "information_gain_estimate": 0.XX
    }}
  ]
}}"""
