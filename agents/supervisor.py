"""
SupervisorAgent: Meta-level oversight and validation
=====================================================

The Supervisor validates proposals, checks risk levels, and has veto power
over decisions. Operates as final gatekeeper for all actions.
"""

from typing import Dict, Any, List
from agents.base import BaseAgent, AgentCapability
from agents.protocol import RiskLevel, SupervisorDecision
from llm.router import LLMRouter


class SupervisorAgent(BaseAgent):
    """
    Meta-level oversight agent.

    Responsibilities:
    - Validate all proposals before execution
    - Assess risk levels (low/medium/high/critical)
    - Approve/reject/override consensus decisions
    - Enforce system-level safety constraints
    - Monitor agent behavior and detect anomalies
    - Has VETO POWER over all actions
    """

    def __init__(
        self,
        agent_id: str = "supervisor_001",
        model: str = "llama-3-8b-local",
        llm_router: LLMRouter = None,
        voting_weight: float = 3.0,  # Highest weight (veto power)
        memory_path: str = "/workspace/arc/memory"
    ):
        """Initialize Supervisor agent."""
        super().__init__(
            agent_id=agent_id,
            role="supervisor",
            model=model,
            capabilities=[AgentCapability.SUPERVISION, AgentCapability.VALIDATION, AgentCapability.SAFETY_REVIEW],
            voting_weight=voting_weight,
            priority="critical",
            offline=True,  # Supervisor can work offline
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate proposals and consensus votes.

        Args:
            input_data: Contains proposals, reviews, and votes

        Returns:
            Supervisor decision (approve/reject/override)
        """
        import time
        start_time = time.time()

        try:
            # Read memory
            proposals = self.read_memory("proposals.json")
            reviews = self.read_memory("reviews.json")
            constraints = self.read_memory("constraints.json")

            # Build prompt
            prompt = self._build_validation_prompt(proposals, reviews, constraints)

            # Get LLM client (can use mock for offline)
            client = self.llm_router.get_client_for_role(self.role)

            # Generate supervisor decision
            response = client.generate_json(prompt, max_tokens=2000, temperature=0.3)

            # Log decision
            self.log_decision("supervisor_validation", response)

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("validate_proposals", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("validate_proposals", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Supervisor vote has VETO POWER.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision (can override consensus)
        """
        # Perform strict validation
        risk_level = self._assess_risk(proposal)

        # VETO if risk is CRITICAL
        if risk_level == RiskLevel.CRITICAL:
            return {
                "decision": "reject",
                "confidence": 1.0,
                "reasoning": "SUPERVISOR VETO: Critical risk detected",
                "suggested_changes": None
            }

        # Warn if risk is HIGH
        if risk_level == RiskLevel.HIGH:
            return {
                "decision": "revise",
                "confidence": 0.95,
                "reasoning": "High risk detected - requires modifications",
                "suggested_changes": {"risk_mitigation": "Add safety checks"}
            }

        # Approve if risk is acceptable
        return {
            "decision": "approve",
            "confidence": 0.9,
            "reasoning": f"Supervisor approval - risk level: {risk_level.value}"
        }

    def validate_consensus(
        self,
        proposal: Dict[str, Any],
        votes: List[Dict[str, Any]]
    ) -> SupervisorDecision:
        """
        Validate consensus decision and optionally override.

        Args:
            proposal: Experiment proposal
            votes: List of agent votes

        Returns:
            Supervisor decision (can override consensus)
        """
        # Calculate consensus
        approve_count = sum(1 for v in votes if v.get("decision") == "approve")
        total_votes = len(votes)
        consensus_pct = approve_count / total_votes if total_votes > 0 else 0.0

        # Assess risk
        risk_level = self._assess_risk(proposal)

        # Check for constraint violations
        constraints_violated = self._check_constraints(proposal)

        # Decide whether to override
        override_consensus = False
        decision = "approve" if consensus_pct > 0.66 else "reject"

        # OVERRIDE if critical risk detected (even if consensus approves)
        if risk_level == RiskLevel.CRITICAL:
            decision = "reject"
            override_consensus = consensus_pct > 0.5
            reasoning = "SUPERVISOR OVERRIDE: Critical safety risk detected"

        # OVERRIDE if consensus rejects a safe experiment (prevent excessive caution)
        elif risk_level == RiskLevel.LOW and consensus_pct < 0.33:
            decision = "approve"
            override_consensus = True
            reasoning = "SUPERVISOR OVERRIDE: Low-risk experiment rejected by excessive caution"

        else:
            reasoning = f"Supervisor validates consensus decision (risk: {risk_level.value})"

        return SupervisorDecision(
            proposal_id=proposal.get("experiment_id", "unknown"),
            decision=decision,
            risk_assessment=risk_level,
            reasoning=reasoning,
            constraints_violated=constraints_violated,
            override_consensus=override_consensus
        )

    def _assess_risk(self, proposal: Dict[str, Any]) -> RiskLevel:
        """
        Assess risk level of a proposal.

        Args:
            proposal: Experiment proposal

        Returns:
            RiskLevel enum
        """
        # Read constraints
        constraints = self.read_memory("constraints.json")

        # Check for forbidden parameters
        config_changes = proposal.get("config_changes", {})
        forbidden_ranges = constraints.get("forbidden_ranges", [])

        for forbidden in forbidden_ranges:
            param = forbidden.get("parameter")
            min_val = forbidden.get("min")
            max_val = forbidden.get("max")

            if param in config_changes:
                value = config_changes[param]
                if min_val <= value <= max_val:
                    return RiskLevel.CRITICAL  # Forbidden range

        # Check novelty category
        novelty = proposal.get("novelty_category", "exploit")
        if novelty == "wildcat":
            return RiskLevel.HIGH
        elif novelty == "explore":
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _check_constraints(self, proposal: Dict[str, Any]) -> List[str]:
        """
        Check which constraints are violated.

        Args:
            proposal: Experiment proposal

        Returns:
            List of constraint violation descriptions
        """
        violations = []
        constraints = self.read_memory("constraints.json")
        config_changes = proposal.get("config_changes", {})
        forbidden_ranges = constraints.get("forbidden_ranges", [])

        for forbidden in forbidden_ranges:
            param = forbidden.get("parameter")
            min_val = forbidden.get("min")
            max_val = forbidden.get("max")
            reason = forbidden.get("reason", "unknown")

            if param in config_changes:
                value = config_changes[param]
                if min_val <= value <= max_val:
                    violations.append(f"{param}={value} in forbidden range [{min_val}, {max_val}] ({reason})")

        return violations

    def _build_validation_prompt(
        self,
        proposals: Dict[str, Any],
        reviews: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Build prompt for supervisor validation."""
        return f"""You are the Supervisor agent in ARC (Autonomous Research Collective).
Your role is to provide FINAL VALIDATION before any experiments are executed.

You have VETO POWER. You can override consensus if safety is at risk.

# Proposals
{proposals}

# Critic Reviews
{reviews}

# Safety Constraints (MUST ENFORCE)
{constraints}

# Your Task
For each proposal:
1. Assess risk level (low/medium/high/critical)
2. Validate safety constraints are satisfied
3. Decide: approve/reject/revise/override
4. Determine if consensus override is needed

Use your VETO POWER if:
- Critical safety risk detected
- Constraints violated
- Consensus is dangerously wrong

Return ONLY a valid JSON object:
{{
  "decisions": [
    {{
      "experiment_id": "exp_XXX",
      "decision": "approve" | "reject" | "revise" | "override",
      "risk_assessment": "low" | "medium" | "high" | "critical",
      "reasoning": "detailed supervisor reasoning",
      "constraints_violated": ["violation1", "violation2"] or [],
      "override_consensus": true | false
    }}
  ],
  "summary": "Overall supervisor assessment"
}}"""
