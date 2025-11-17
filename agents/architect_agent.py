"""
ArchitectAgent: Experiment design and proposal generation
==========================================================

The Architect generates hypothesis-driven experiment proposals
based on Director's strategy and Historian's insights.
"""

from typing import Dict, Any, List
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter


class ArchitectAgent(BaseAgent):
    """
    Experiment design agent.

    Responsibilities:
    - Generate experiment proposals
    - Design configuration changes
    - Predict metric impacts
    - Assign novelty categories
    """

    def __init__(
        self,
        agent_id: str = "architect_001",
        model: str = "deepseek-r1",
        llm_router: LLMRouter = None,
        voting_weight: float = 1.5,
        memory_path: str = "/workspace/arc/memory"
    ):
        """Initialize Architect agent."""
        super().__init__(
            agent_id=agent_id,
            role="architect",
            model=model,
            capabilities=[AgentCapability.PROPOSAL_GENERATION],
            voting_weight=voting_weight,
            priority="medium",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate experiment proposals.

        Args:
            input_data: Contains directive, history, constraints

        Returns:
            List of experiment proposals
        """
        import time
        start_time = time.time()

        try:
            # Read memory
            directive = self.read_memory("directive.json")
            history = self.read_memory("history_summary.json")
            constraints = self.read_memory("constraints.json")

            # Build prompt
            prompt = self._build_proposal_prompt(directive, history, constraints)

            # Get LLM client
            client = self.llm_router.get_client_for_role(self.role)

            # Generate proposals
            response = client.generate_json(prompt, max_tokens=3000, temperature=0.8)

            # Write to memory
            self.write_memory("proposals.json", response)

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("generate_proposals", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("generate_proposals", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Architects generally approve their own proposals.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision
        """
        # Architects approve by default (Critics will scrutinize)
        return {
            "decision": "approve",
            "confidence": 0.85,
            "reasoning": "Proposal designed to meet strategic objectives"
        }

    def _build_proposal_prompt(
        self,
        directive: Dict[str, Any],
        history: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Build prompt for proposal generation."""
        return f"""You are the Architect agent in ARC (Autonomous Research Collective).
Your role is to design novel experiments based on strategic direction.

# Strategic Directive
{directive}

# Research History
{history}

# Safety Constraints
{constraints}

# Your Task
Generate {directive.get('novelty_budget', {}).get('exploit', 2) + directive.get('novelty_budget', {}).get('explore', 1)} experiment proposals.

Each proposal must include:
1. Unique experiment ID
2. Descriptive name
3. Scientific hypothesis
4. Novelty category (exploit/explore/wildcat)
5. Predicted metrics
6. Configuration changes
7. Justification

Return ONLY a valid JSON object:
{{
  "proposals": [
    {{
      "experiment_id": "exp_XXX",
      "name": "descriptive_name",
      "hypothesis": "scientific hypothesis",
      "novelty_category": "exploit" | "explore" | "wildcat",
      "predicted_metrics": {{"auc": 0.XX, "sensitivity": 0.XX, "specificity": 0.XX}},
      "config_changes": {{"param1": value1, "param2": value2}},
      "justification": "why this experiment is valuable"
    }}
  ]
}}"""
