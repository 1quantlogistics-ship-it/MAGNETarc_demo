"""
DirectorAgent: Strategic planning and mode control
===================================================

The Director sets research strategy, allocates budgets, and determines
when to explore vs exploit.
"""

from typing import Dict, Any
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter


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
