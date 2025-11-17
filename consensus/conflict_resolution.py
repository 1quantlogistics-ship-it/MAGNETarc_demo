"""
ConflictResolver: Resolution strategies for voting conflicts
=============================================================

Handles cases where consensus is not reached or agents strongly disagree.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from consensus.voting import VoteResult, VoteDecision


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies"""
    MAJORITY_RULE = "majority_rule"          # Simple majority wins
    WEIGHTED_MAJORITY = "weighted_majority"  # Weighted majority wins
    SUPERVISOR_OVERRIDE = "supervisor_override"  # Supervisor makes final call
    HIGHEST_CONFIDENCE = "highest_confidence"  # Trust most confident voter
    DIRECTOR_OVERRIDE = "director_override"  # Director has final say
    CONSERVATIVE = "conservative"            # Default to reject if unclear
    PROGRESSIVE = "progressive"              # Default to approve if unclear
    MEDIATION = "mediation"                  # Attempt to find compromise


class ConflictResolver:
    """
    Resolves voting conflicts and disagreements.

    Handles:
    - Tie votes
    - Controversial decisions (low consensus)
    - Agent disagreements
    - Supervisor overrides
    """

    def __init__(
        self,
        default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.CONSERVATIVE,
        supervisor_override_threshold: float = 0.9  # Supervisor confidence threshold
    ):
        """
        Initialize conflict resolver.

        Args:
            default_strategy: Default resolution strategy
            supervisor_override_threshold: Min confidence for supervisor override
        """
        self.default_strategy = default_strategy
        self.supervisor_override_threshold = supervisor_override_threshold

    def resolve_conflict(
        self,
        vote_result: VoteResult,
        strategy: Optional[ConflictResolutionStrategy] = None,
        supervisor_vote: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve a voting conflict.

        Args:
            vote_result: VoteResult with conflict
            strategy: Resolution strategy to use (None = use default)
            supervisor_vote: Supervisor's vote (if available)

        Returns:
            Resolution result with final decision
        """
        strategy = strategy or self.default_strategy

        # Check if supervisor override applies
        if supervisor_vote and self._should_supervisor_override(supervisor_vote):
            return self._apply_supervisor_override(vote_result, supervisor_vote)

        # Apply selected strategy
        if strategy == ConflictResolutionStrategy.MAJORITY_RULE:
            return self._resolve_by_majority(vote_result)

        elif strategy == ConflictResolutionStrategy.WEIGHTED_MAJORITY:
            return self._resolve_by_weighted_majority(vote_result)

        elif strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            return self._resolve_by_highest_confidence(vote_result)

        elif strategy == ConflictResolutionStrategy.CONSERVATIVE:
            return self._resolve_conservative(vote_result)

        elif strategy == ConflictResolutionStrategy.PROGRESSIVE:
            return self._resolve_progressive(vote_result)

        elif strategy == ConflictResolutionStrategy.SUPERVISOR_OVERRIDE:
            if supervisor_vote:
                return self._apply_supervisor_override(vote_result, supervisor_vote)
            else:
                # Fallback to conservative if no supervisor
                return self._resolve_conservative(vote_result)

        elif strategy == ConflictResolutionStrategy.MEDIATION:
            return self._resolve_by_mediation(vote_result)

        else:
            # Default: conservative
            return self._resolve_conservative(vote_result)

    def _should_supervisor_override(self, supervisor_vote: Dict[str, Any]) -> bool:
        """Check if supervisor should override consensus."""
        confidence = supervisor_vote.get("confidence", 0.0)
        return confidence >= self.supervisor_override_threshold

    def _apply_supervisor_override(
        self,
        vote_result: VoteResult,
        supervisor_vote: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply supervisor override."""
        decision_map = {
            "approve": VoteDecision.APPROVE,
            "reject": VoteDecision.REJECT,
            "revise": VoteDecision.REVISE,
            "abstain": VoteDecision.ABSTAIN
        }

        supervisor_decision = supervisor_vote.get("decision", "reject")

        return {
            "proposal_id": vote_result.proposal_id,
            "final_decision": decision_map.get(supervisor_decision, VoteDecision.REJECT).value,
            "resolution_strategy": "supervisor_override",
            "reasoning": supervisor_vote.get("reasoning", "Supervisor override applied"),
            "original_consensus": vote_result.decision.value,
            "override_applied": True,
            "confidence": supervisor_vote.get("confidence", 1.0)
        }

    def _resolve_by_majority(self, vote_result: VoteResult) -> Dict[str, Any]:
        """Resolve by simple majority (count votes)."""
        votes = vote_result.votes

        decision_counts = {
            "approve": 0,
            "reject": 0,
            "revise": 0
        }

        for vote in votes:
            decision = vote.get("decision", "abstain")
            if decision in decision_counts:
                decision_counts[decision] += 1

        # Find majority
        max_count = max(decision_counts.values())
        majority_decision = [d for d, c in decision_counts.items() if c == max_count][0]

        decision_map = {
            "approve": VoteDecision.APPROVE,
            "reject": VoteDecision.REJECT,
            "revise": VoteDecision.REVISE
        }

        return {
            "proposal_id": vote_result.proposal_id,
            "final_decision": majority_decision,
            "resolution_strategy": "majority_rule",
            "reasoning": f"Simple majority: {max_count}/{len(votes)} votes for {majority_decision}",
            "original_consensus": vote_result.decision.value,
            "override_applied": False,
            "confidence": max_count / len(votes) if len(votes) > 0 else 0.0
        }

    def _resolve_by_weighted_majority(self, vote_result: VoteResult) -> Dict[str, Any]:
        """Resolve by weighted majority (use weighted score)."""
        # Already calculated in vote_result
        return {
            "proposal_id": vote_result.proposal_id,
            "final_decision": vote_result.decision.value,
            "resolution_strategy": "weighted_majority",
            "reasoning": f"Weighted consensus score: {vote_result.weighted_score:.2f}",
            "original_consensus": vote_result.decision.value,
            "override_applied": False,
            "confidence": vote_result.confidence
        }

    def _resolve_by_highest_confidence(self, vote_result: VoteResult) -> Dict[str, Any]:
        """Resolve by trusting most confident voter."""
        votes = vote_result.votes

        # Find vote with highest confidence
        max_confidence = 0.0
        most_confident_vote = None

        for vote in votes:
            confidence = vote.get("confidence", 0.0)
            if confidence > max_confidence:
                max_confidence = confidence
                most_confident_vote = vote

        if most_confident_vote is None:
            # Fallback to conservative
            return self._resolve_conservative(vote_result)

        decision = most_confident_vote.get("decision", "reject")

        return {
            "proposal_id": vote_result.proposal_id,
            "final_decision": decision,
            "resolution_strategy": "highest_confidence",
            "reasoning": f"Trusting most confident voter: {most_confident_vote.get('agent_id', 'unknown')} (confidence: {max_confidence:.2f})",
            "original_consensus": vote_result.decision.value,
            "override_applied": vote_result.decision.value != decision,
            "confidence": max_confidence
        }

    def _resolve_conservative(self, vote_result: VoteResult) -> Dict[str, Any]:
        """Conservative resolution: reject if unclear."""
        return {
            "proposal_id": vote_result.proposal_id,
            "final_decision": "reject",
            "resolution_strategy": "conservative",
            "reasoning": "Conservative policy: reject when consensus unclear",
            "original_consensus": vote_result.decision.value,
            "override_applied": vote_result.decision.value != "reject",
            "confidence": 0.5
        }

    def _resolve_progressive(self, vote_result: VoteResult) -> Dict[str, Any]:
        """Progressive resolution: approve if no strong objections."""
        votes = vote_result.votes

        # Count strong rejections (high confidence reject votes)
        strong_rejections = sum(
            1 for v in votes
            if v.get("decision") == "reject" and v.get("confidence", 0.0) > 0.8
        )

        # If no strong rejections, approve
        if strong_rejections == 0:
            final_decision = "approve"
            reasoning = "Progressive policy: approve when no strong objections"
        else:
            final_decision = "reject"
            reasoning = f"Progressive policy: reject due to {strong_rejections} strong objections"

        return {
            "proposal_id": vote_result.proposal_id,
            "final_decision": final_decision,
            "resolution_strategy": "progressive",
            "reasoning": reasoning,
            "original_consensus": vote_result.decision.value,
            "override_applied": vote_result.decision.value != final_decision,
            "confidence": 0.6
        }

    def _resolve_by_mediation(self, vote_result: VoteResult) -> Dict[str, Any]:
        """Attempt to find compromise (suggest revisions)."""
        # If any votes suggest revise, go with revise
        revise_votes = [v for v in vote_result.votes if v.get("decision") == "revise"]

        if len(revise_votes) > 0:
            final_decision = "revise"
            reasoning = f"Mediation: {len(revise_votes)} agents suggest revisions"
        else:
            # Otherwise, use weighted majority
            final_decision = vote_result.decision.value
            reasoning = "Mediation: using weighted consensus as no revisions suggested"

        return {
            "proposal_id": vote_result.proposal_id,
            "final_decision": final_decision,
            "resolution_strategy": "mediation",
            "reasoning": reasoning,
            "original_consensus": vote_result.decision.value,
            "override_applied": vote_result.decision.value != final_decision,
            "confidence": vote_result.confidence
        }

    def detect_controversy(self, vote_result: VoteResult) -> Dict[str, Any]:
        """
        Detect if a vote is controversial.

        Args:
            vote_result: VoteResult to analyze

        Returns:
            Controversy analysis
        """
        votes = vote_result.votes

        # Calculate variance in decisions
        decision_counts = {
            "approve": 0,
            "reject": 0,
            "revise": 0
        }

        for vote in votes:
            decision = vote.get("decision", "abstain")
            if decision in decision_counts:
                decision_counts[decision] += 1

        total_active = sum(decision_counts.values())
        if total_active == 0:
            return {"controversial": False, "reason": "no_active_votes"}

        # Calculate entropy (higher = more controversial)
        import math
        entropy = 0.0
        for count in decision_counts.values():
            if count > 0:
                p = count / total_active
                entropy -= p * math.log2(p)

        # Controversial if entropy > 1.0 (indicates significant disagreement)
        controversial = entropy > 1.0

        return {
            "controversial": controversial,
            "entropy": entropy,
            "decision_distribution": decision_counts,
            "reason": "high_disagreement" if controversial else "consensus_reached"
        }
