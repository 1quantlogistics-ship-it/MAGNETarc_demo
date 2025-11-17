"""
HistorianAgent: Memory management and learning
===============================================

The Historian compresses experiment history, tracks patterns,
and infers constraints from past failures.
"""

from typing import Dict, Any, List
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter


class HistorianAgent(BaseAgent):
    """
    Memory and learning agent.

    Responsibilities:
    - Compress experiment history
    - Track winning/failing configurations
    - Infer forbidden parameter ranges
    - Analyze performance trends
    """

    def __init__(
        self,
        agent_id: str = "historian_001",
        model: str = "deepseek-r1",
        llm_router: LLMRouter = None,
        voting_weight: float = 1.0,
        memory_path: str = "/workspace/arc/memory"
    ):
        """Initialize Historian agent."""
        super().__init__(
            agent_id=agent_id,
            role="historian",
            model=model,
            capabilities=[AgentCapability.MEMORY_MANAGEMENT],
            voting_weight=voting_weight,
            priority="medium",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update history summary with new experiment results.

        Args:
            input_data: Contains new experiment results

        Returns:
            Updated history summary
        """
        import time
        start_time = time.time()

        try:
            # Read current history
            history = self.read_memory("history_summary.json") or {}
            constraints = self.read_memory("constraints.json") or {}

            # Get experiment results from input
            new_results = input_data.get("experiment_results", [])

            # Build prompt
            prompt = self._build_update_prompt(history, constraints, new_results)

            # Get LLM client
            client = self.llm_router.get_client_for_role(self.role)

            # Generate updated history
            response = client.generate_json(prompt, max_tokens=2500, temperature=0.5)

            # Write to memory
            self.write_memory("history_summary.json", response.get("history", {}))
            self.write_memory("constraints.json", response.get("constraints", {}))

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("update_history", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("update_history", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Historians vote based on historical precedent.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision based on history
        """
        # Read history
        history = self.read_memory("history_summary.json")

        # Check if similar config failed before
        failed_configs = history.get("failed_configs", [])
        config_changes = proposal.get("config_changes", {})

        for failed in failed_configs:
            # Simple similarity check (exact match)
            if all(failed.get(k) == v for k, v in config_changes.items() if k in failed):
                return {
                    "decision": "reject",
                    "confidence": 0.85,
                    "reasoning": "Similar configuration failed previously",
                    "suggested_changes": None
                }

        # Default: approve
        return {
            "decision": "approve",
            "confidence": 0.7,
            "reasoning": "No historical evidence of failure"
        }

    def _build_update_prompt(
        self,
        history: Dict[str, Any],
        constraints: Dict[str, Any],
        new_results: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for history update."""
        return f"""You are the Historian agent in ARC (Autonomous Research Collective).
Your role is to maintain a compressed history of all experiments and learn from results.

# Current History Summary
{history}

# Current Constraints
{constraints}

# New Experiment Results
{new_results}

# Your Task
Update the history summary and constraints based on new results:
1. Update total cycles and experiments
2. Update best metrics if improved
3. Add recent experiments to history
4. Identify failed configs and update forbidden ranges
5. Detect successful patterns
6. Infer new constraints from failures

Return ONLY a valid JSON object:
{{
  "history": {{
    "total_cycles": N,
    "total_experiments": M,
    "best_metrics": {{"auc": 0.XX, "sensitivity": 0.XX, "specificity": 0.XX}},
    "recent_experiments": [...],
    "failed_configs": [...],
    "successful_patterns": [...]
  }},
  "constraints": {{
    "forbidden_ranges": [{{"parameter": "X", "min": A, "max": B, "reason": "..."}}],
    "unstable_configs": [...],
    "safe_baselines": [...]
  }}
}}"""
