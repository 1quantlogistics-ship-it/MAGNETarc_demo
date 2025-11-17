"""
VotingSystem: Democratic voting mechanism for multi-agent consensus
====================================================================

Supports weighted voting, confidence scoring, and consensus analysis.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class VoteDecision(Enum):
    """Vote decision types"""
    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"
    ABSTAIN = "abstain"


@dataclass
class VoteResult:
    """
    Result of a voting process.

    Attributes:
        proposal_id: ID of proposal voted on
        total_votes: Total number of votes cast
        weighted_score: Weighted consensus score (-1.0 to 1.0)
        consensus_reached: Whether consensus threshold met
        decision: Final decision (approve/reject/revise)
        votes: Individual vote details
        confidence: Average confidence across votes
    """
    proposal_id: str
    total_votes: int
    weighted_score: float
    consensus_reached: bool
    decision: VoteDecision
    votes: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "total_votes": self.total_votes,
            "weighted_score": self.weighted_score,
            "consensus_reached": self.consensus_reached,
            "decision": self.decision.value,
            "votes": self.votes,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class ConsensusMetrics:
    """
    Metrics about consensus process.

    Attributes:
        agreement_rate: Percentage of votes in agreement
        disagreement_rate: Percentage of votes in disagreement
        abstention_rate: Percentage of abstentions
        avg_confidence: Average confidence across all votes
        controversial: Whether vote was controversial (low agreement)
    """
    agreement_rate: float
    disagreement_rate: float
    abstention_rate: float
    avg_confidence: float
    controversial: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agreement_rate": self.agreement_rate,
            "disagreement_rate": self.disagreement_rate,
            "abstention_rate": self.abstention_rate,
            "avg_confidence": self.avg_confidence,
            "controversial": self.controversial
        }


class VotingSystem:
    """
    Democratic voting system for multi-agent consensus.

    Features:
    - Weighted voting (agents have different voting weights)
    - Confidence scoring
    - Consensus threshold enforcement
    - Tie-breaking rules
    - Vote analytics
    """

    def __init__(
        self,
        consensus_threshold: float = 0.66,
        min_votes_required: int = 2,
        enable_confidence_weighting: bool = True
    ):
        """
        Initialize voting system.

        Args:
            consensus_threshold: Minimum weighted score for consensus (0.0-1.0)
            min_votes_required: Minimum number of votes required
            enable_confidence_weighting: Weight votes by confidence
        """
        self.consensus_threshold = consensus_threshold
        self.min_votes_required = min_votes_required
        self.enable_confidence_weighting = enable_confidence_weighting

    def conduct_vote(
        self,
        proposal: Dict[str, Any],
        votes: List[Dict[str, Any]]
    ) -> VoteResult:
        """
        Conduct a vote on a proposal.

        Args:
            proposal: Proposal to vote on
            votes: List of votes from agents
                Each vote: {
                    "agent_id": str,
                    "decision": "approve" | "reject" | "revise" | "abstain",
                    "confidence": float (0.0-1.0),
                    "voting_weight": float,
                    "reasoning": str
                }

        Returns:
            VoteResult with consensus decision
        """
        proposal_id = proposal.get("experiment_id", "unknown")

        # Filter out abstentions for score calculation
        active_votes = [v for v in votes if v.get("decision") != "abstain"]

        if len(active_votes) < self.min_votes_required:
            # Not enough votes - default to reject
            return VoteResult(
                proposal_id=proposal_id,
                total_votes=len(votes),
                weighted_score=0.0,
                consensus_reached=False,
                decision=VoteDecision.REJECT,
                votes=votes,
                confidence=0.0,
                metadata={"reason": "insufficient_votes"}
            )

        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(active_votes)

        # Determine decision
        decision = self._determine_decision(weighted_score, active_votes)

        # Check consensus
        consensus_reached = abs(weighted_score) >= self.consensus_threshold

        # Calculate average confidence
        avg_confidence = sum(v.get("confidence", 0.5) for v in active_votes) / len(active_votes)

        return VoteResult(
            proposal_id=proposal_id,
            total_votes=len(votes),
            weighted_score=weighted_score,
            consensus_reached=consensus_reached,
            decision=decision,
            votes=votes,
            confidence=avg_confidence,
            metadata={
                "active_votes": len(active_votes),
                "abstentions": len(votes) - len(active_votes)
            }
        )

    def _calculate_weighted_score(self, votes: List[Dict[str, Any]]) -> float:
        """
        Calculate weighted consensus score.

        Returns value between -1.0 (unanimous reject) and 1.0 (unanimous approve).

        Args:
            votes: List of active votes

        Returns:
            Weighted score
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for vote in votes:
            decision = vote.get("decision", "abstain")
            weight = vote.get("voting_weight", 1.0)
            confidence = vote.get("confidence", 0.5)

            # Apply confidence weighting if enabled
            if self.enable_confidence_weighting:
                weight = weight * confidence

            # Map decision to score
            if decision == "approve":
                score = 1.0
            elif decision == "reject":
                score = -1.0
            elif decision == "revise":
                score = 0.0  # Neutral
            else:
                continue  # Skip abstain

            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _determine_decision(
        self,
        weighted_score: float,
        votes: List[Dict[str, Any]]
    ) -> VoteDecision:
        """
        Determine final decision based on weighted score.

        Args:
            weighted_score: Calculated weighted score
            votes: List of votes

        Returns:
            Final decision
        """
        # Count decision types
        decision_counts = {
            "approve": 0,
            "reject": 0,
            "revise": 0
        }

        for vote in votes:
            decision = vote.get("decision", "abstain")
            if decision in decision_counts:
                decision_counts[decision] += 1

        # Strong consensus to approve
        if weighted_score >= self.consensus_threshold:
            return VoteDecision.APPROVE

        # Strong consensus to reject
        if weighted_score <= -self.consensus_threshold:
            return VoteDecision.REJECT

        # If majority wants revision
        if decision_counts["revise"] > decision_counts["approve"] and \
           decision_counts["revise"] > decision_counts["reject"]:
            return VoteDecision.REVISE

        # Tie or unclear - default to reject (conservative)
        return VoteDecision.REJECT

    def analyze_consensus(self, vote_result: VoteResult) -> ConsensusMetrics:
        """
        Analyze consensus quality.

        Args:
            vote_result: Result to analyze

        Returns:
            ConsensusMetrics
        """
        votes = vote_result.votes
        total = len(votes)

        if total == 0:
            return ConsensusMetrics(
                agreement_rate=0.0,
                disagreement_rate=0.0,
                abstention_rate=0.0,
                avg_confidence=0.0,
                controversial=True
            )

        # Count decisions
        approve_count = sum(1 for v in votes if v.get("decision") == "approve")
        reject_count = sum(1 for v in votes if v.get("decision") == "reject")
        revise_count = sum(1 for v in votes if v.get("decision") == "revise")
        abstain_count = sum(1 for v in votes if v.get("decision") == "abstain")

        # Calculate rates
        majority_decision = max(approve_count, reject_count, revise_count)
        agreement_rate = majority_decision / total
        disagreement_rate = (total - majority_decision - abstain_count) / total
        abstention_rate = abstain_count / total

        # Controversial if agreement < 60%
        controversial = agreement_rate < 0.6

        return ConsensusMetrics(
            agreement_rate=agreement_rate,
            disagreement_rate=disagreement_rate,
            abstention_rate=abstention_rate,
            avg_confidence=vote_result.confidence,
            controversial=controversial
        )

    def get_vote_summary(self, vote_results: List[VoteResult]) -> Dict[str, Any]:
        """
        Get summary statistics across multiple votes.

        Args:
            vote_results: List of vote results

        Returns:
            Summary dictionary
        """
        if not vote_results:
            return {
                "total_votes": 0,
                "consensus_rate": 0.0,
                "avg_confidence": 0.0,
                "controversial_rate": 0.0
            }

        total = len(vote_results)
        consensus_count = sum(1 for v in vote_results if v.consensus_reached)
        avg_confidence = sum(v.confidence for v in vote_results) / total

        # Analyze consensus for each
        metrics = [self.analyze_consensus(v) for v in vote_results]
        controversial_count = sum(1 for m in metrics if m.controversial)

        return {
            "total_votes": total,
            "consensus_rate": consensus_count / total,
            "avg_confidence": avg_confidence,
            "controversial_rate": controversial_count / total,
            "decision_breakdown": self._get_decision_breakdown(vote_results)
        }

    def _get_decision_breakdown(self, vote_results: List[VoteResult]) -> Dict[str, int]:
        """Get breakdown of final decisions."""
        breakdown = {
            "approve": 0,
            "reject": 0,
            "revise": 0,
            "abstain": 0
        }

        for result in vote_results:
            decision = result.decision.value
            if decision in breakdown:
                breakdown[decision] += 1

        return breakdown
