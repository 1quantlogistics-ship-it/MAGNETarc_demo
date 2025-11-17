"""
CriticSecondaryAgent: Secondary safety reviewer
================================================

The Secondary Critic provides a second opinion on proposals,
focusing on scientific rigor and experimental design quality.
"""

from typing import Dict, Any
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter


class CriticSecondaryAgent(BaseAgent):
    """
    Secondary safety review agent.

    Responsibilities:
    - Provide second opinion on proposals
    - Challenge scientific assumptions
    - Evaluate experimental design quality
    - Detect logical flaws in hypotheses
    - Prevent groupthink bias
    """

    def __init__(
        self,
        agent_id: str = "critic_secondary_001",
        model: str = "deepseek-r1",
        llm_router: LLMRouter = None,
        voting_weight: float = 1.8,
        memory_path: str = "/workspace/arc/memory"
    ):
        """Initialize Secondary Critic agent."""
        super().__init__(
            agent_id=agent_id,
            role="critic_secondary",
            model=model,
            capabilities=[AgentCapability.SAFETY_REVIEW, AgentCapability.VALIDATION],
            voting_weight=voting_weight,
            priority="high",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide secondary review of proposals.

        Args:
            input_data: Contains proposals and primary reviews

        Returns:
            Secondary review opinions
        """
        import time
        start_time = time.time()

        try:
            # Read memory
            proposals = self.read_memory("proposals.json")
            primary_reviews = self.read_memory("reviews.json")
            history = self.read_memory("history_summary.json")

            # Build prompt
            prompt = self._build_secondary_review_prompt(proposals, primary_reviews, history)

            # Get LLM client
            client = self.llm_router.get_client_for_role(self.role)

            # Generate secondary reviews
            response = client.generate_json(prompt, max_tokens=2500, temperature=0.6)

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("secondary_review", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("secondary_review", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide critical second opinion.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision with scientific critique
        """
        # Check hypothesis quality
        hypothesis = proposal.get("hypothesis", "")

        # Simple heuristic: reject if hypothesis is too vague
        if len(hypothesis.split()) < 5:
            return {
                "decision": "revise",
                "confidence": 0.8,
                "reasoning": "Hypothesis is too vague and lacks scientific rigor",
                "suggested_changes": {"hypothesis": "Provide more specific, testable hypothesis"}
            }

        # Check if justification is provided
        justification = proposal.get("justification", "")
        if len(justification.split()) < 5:
            return {
                "decision": "revise",
                "confidence": 0.75,
                "reasoning": "Insufficient justification for proposed experiment",
                "suggested_changes": {"justification": "Provide detailed scientific rationale"}
            }

        # Default: approve with skepticism
        return {
            "decision": "approve",
            "confidence": 0.7,
            "reasoning": "Proposal meets minimum scientific standards"
        }

    def _build_secondary_review_prompt(
        self,
        proposals: Dict[str, Any],
        primary_reviews: Dict[str, Any],
        history: Dict[str, Any]
    ) -> str:
        """Build prompt for secondary review."""
        return f"""You are the Secondary Critic agent in ARC (Autonomous Research Collective).
Your role is to provide a SECOND OPINION on proposals, challenging the primary Critic.

# Proposals
{proposals}

# Primary Critic Reviews
{primary_reviews}

# Research History
{history}

# Your Task
For each proposal, provide an INDEPENDENT review that:
1. Challenges scientific assumptions
2. Evaluates experimental design quality
3. Detects logical flaws in hypotheses
4. Questions the primary Critic's reasoning (if needed)
5. Prevents groupthink and confirmation bias

You may DISAGREE with the primary Critic. Your job is to catch what they missed.

Return ONLY a valid JSON object:
{{
  "secondary_reviews": [
    {{
      "experiment_id": "exp_XXX",
      "decision": "approve" | "reject" | "revise",
      "confidence": 0.XX,
      "reasoning": "independent critical analysis",
      "disagrees_with_primary": true | false,
      "disagreement_reasoning": "why you disagree with primary Critic" or null,
      "suggested_changes": {{...}} or null
    }}
  ]
}}"""
