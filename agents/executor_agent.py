"""
ExecutorAgent: Training execution and results collection
=========================================================

The Executor translates approved proposals into training jobs
and collects experimental results.
"""

from typing import Dict, Any
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter


class ExecutorAgent(BaseAgent):
    """
    Training execution agent.

    Responsibilities:
    - Generate safe config diffs
    - Execute training via Control Plane
    - Monitor training progress
    - Collect and report metrics
    """

    def __init__(
        self,
        agent_id: str = "executor_001",
        model: str = "deepseek-r1",
        llm_router: LLMRouter = None,
        voting_weight: float = 1.0,
        memory_path: str = "/workspace/arc/memory"
    ):
        """Initialize Executor agent."""
        super().__init__(
            agent_id=agent_id,
            role="executor",
            model=model,
            capabilities=[AgentCapability.EXECUTION],
            voting_weight=voting_weight,
            priority="medium",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute approved experiments.

        Args:
            input_data: Contains approved proposals

        Returns:
            Execution status and commands
        """
        import time
        start_time = time.time()

        try:
            # Read approved proposals
            reviews = self.read_memory("reviews.json")
            proposals = self.read_memory("proposals.json")

            # Filter approved proposals
            approved = self._get_approved_proposals(proposals, reviews)

            # Build execution plan
            prompt = self._build_execution_prompt(approved)

            # Get LLM client
            client = self.llm_router.get_client_for_role(self.role)

            # Generate execution plan
            response = client.generate_json(prompt, max_tokens=2000, temperature=0.5)

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("plan_execution", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("plan_execution", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executors check if proposal is technically feasible.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision based on feasibility
        """
        # Check if config changes are valid
        config_changes = proposal.get("config_changes", {})

        # Simple validation: ensure numeric params are reasonable
        for param, value in config_changes.items():
            if isinstance(value, (int, float)):
                if value < 0 or value > 1e6:
                    return {
                        "decision": "reject",
                        "confidence": 0.9,
                        "reasoning": f"Parameter {param}={value} is outside reasonable range",
                        "suggested_changes": None
                    }

        # Default: approve
        return {
            "decision": "approve",
            "confidence": 0.8,
            "reasoning": "Configuration is technically feasible"
        }

    def _get_approved_proposals(
        self,
        proposals: Dict[str, Any],
        reviews: Dict[str, Any]
    ) -> list:
        """Filter proposals to only approved ones."""
        approved = []
        proposal_list = proposals.get("proposals", [])
        review_list = reviews.get("reviews", [])

        # Create review lookup
        review_map = {r["experiment_id"]: r for r in review_list}

        for proposal in proposal_list:
            exp_id = proposal["experiment_id"]
            review = review_map.get(exp_id, {})

            if review.get("decision") == "approve":
                approved.append(proposal)

        return approved

    def _build_execution_prompt(self, approved_proposals: list) -> str:
        """Build prompt for execution planning."""
        return f"""You are the Executor agent in ARC (Autonomous Research Collective).
Your role is to translate approved proposals into executable training commands.

# Approved Proposals
{approved_proposals}

# Your Task
For each approved proposal, generate:
1. Training command (safe, validated)
2. Estimated duration
3. Resource requirements

Return ONLY a valid JSON object:
{{
  "executions": [
    {{
      "experiment_id": "exp_XXX",
      "command": "python train.py --param1 value1 --param2 value2",
      "estimated_duration_minutes": NN,
      "status": "queued",
      "notes": "execution notes"
    }}
  ]
}}"""
