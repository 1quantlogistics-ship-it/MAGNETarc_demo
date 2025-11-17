"""
ARC Consensus Mechanisms
=========================

Multi-agent voting and conflict resolution for democratic decision-making.

Key Components:
- VotingSystem: Democratic voting on proposals
- ConflictResolver: Resolution of voting conflicts
- Consensus metrics and analysis
"""

from consensus.voting import VotingSystem, VoteResult, ConsensusMetrics
from consensus.conflict_resolution import ConflictResolver, ConflictResolutionStrategy

__all__ = [
    "VotingSystem",
    "VoteResult",
    "ConsensusMetrics",
    "ConflictResolver",
    "ConflictResolutionStrategy"
]
