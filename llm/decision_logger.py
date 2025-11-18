"""
Decision Logger: Structured audit trail for multi-agent decisions
=================================================================

Provides JSONL-based logging for complete transparency:
- Individual agent votes with reasoning
- Consensus calculations and weighted scores
- Conflict resolution events
- Supervisor override justifications
- Voting pattern analysis
- Confidence score tracking

All logs are append-only JSONL format for easy analysis and replay.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


logger = logging.getLogger(__name__)


class LogEventType(Enum):
    """Types of decision events to log"""
    VOTE_CAST = "vote_cast"
    CONSENSUS_REACHED = "consensus_reached"
    CONSENSUS_FAILED = "consensus_failed"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    SUPERVISOR_OVERRIDE = "supervisor_override"
    SUPERVISOR_APPROVAL = "supervisor_approval"
    PROPOSAL_GENERATED = "proposal_generated"
    PROPOSAL_APPROVED = "proposal_approved"
    PROPOSAL_REJECTED = "proposal_rejected"
    CYCLE_STARTED = "cycle_started"
    CYCLE_COMPLETED = "cycle_completed"


@dataclass
class VoteLogEntry:
    """Individual agent vote record"""
    timestamp: str
    event_type: str
    cycle_id: int
    proposal_id: str
    agent_id: str
    agent_role: str
    voting_weight: float
    decision: str  # approve, reject, revise, abstain
    confidence: float
    reasoning: str
    constraints_checked: List[str]
    metadata: Dict[str, Any]


@dataclass
class ConsensusLogEntry:
    """Consensus calculation record"""
    timestamp: str
    event_type: str
    cycle_id: int
    proposal_id: str
    total_votes: int
    weighted_score: float
    consensus_reached: bool
    final_decision: str
    confidence: float
    vote_distribution: Dict[str, int]  # approve: 4, reject: 1, revise: 1
    participating_agents: List[str]
    metadata: Dict[str, Any]


@dataclass
class ConflictLogEntry:
    """Conflict detection and resolution record"""
    timestamp: str
    event_type: str
    cycle_id: int
    proposal_id: str
    conflict_type: str  # tie, controversial, low_confidence
    entropy: float  # measure of disagreement
    resolution_strategy: str
    original_decision: str
    final_decision: str
    override_applied: bool
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class SupervisorLogEntry:
    """Supervisor decision record"""
    timestamp: str
    event_type: str
    cycle_id: int
    proposal_id: str
    supervisor_decision: str
    risk_assessment: str  # low, medium, high, critical
    consensus_decision: str
    override_consensus: bool
    confidence: float
    reasoning: str
    constraints_violated: List[str]
    safety_concerns: List[str]
    metadata: Dict[str, Any]


class DecisionLogger:
    """
    Structured decision logger with JSONL output.

    Features:
    - Append-only JSONL format
    - Automatic log rotation by cycle
    - Type-safe log entries
    - Query and analysis utilities
    """

    def __init__(self, log_dir: str = "/workspace/arc/memory/logs"):
        """
        Initialize decision logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Separate log files for different event types
        self.vote_log = self.log_dir / "votes.jsonl"
        self.consensus_log = self.log_dir / "consensus.jsonl"
        self.conflict_log = self.log_dir / "conflicts.jsonl"
        self.supervisor_log = self.log_dir / "supervisor.jsonl"
        self.cycle_log = self.log_dir / "cycles.jsonl"

        logger.info(f"DecisionLogger initialized (log_dir={log_dir})")

    def log_vote(
        self,
        cycle_id: int,
        proposal_id: str,
        agent_id: str,
        agent_role: str,
        voting_weight: float,
        decision: str,
        confidence: float,
        reasoning: str,
        constraints_checked: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an individual agent vote.

        Args:
            cycle_id: Research cycle ID
            proposal_id: Proposal being voted on
            agent_id: Agent identifier
            agent_role: Agent role (director, critic, etc.)
            voting_weight: Agent's voting weight
            decision: Vote decision (approve/reject/revise/abstain)
            confidence: Vote confidence (0.0-1.0)
            reasoning: Agent's reasoning
            constraints_checked: Constraints validated
            metadata: Additional metadata
        """
        entry = VoteLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            event_type=LogEventType.VOTE_CAST.value,
            cycle_id=cycle_id,
            proposal_id=proposal_id,
            agent_id=agent_id,
            agent_role=agent_role,
            voting_weight=voting_weight,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            constraints_checked=constraints_checked or [],
            metadata=metadata or {}
        )

        self._append_log(self.vote_log, asdict(entry))
        logger.debug(f"Logged vote: {agent_id} -> {decision} (confidence: {confidence:.2f})")

    def log_consensus(
        self,
        cycle_id: int,
        proposal_id: str,
        total_votes: int,
        weighted_score: float,
        consensus_reached: bool,
        final_decision: str,
        confidence: float,
        vote_distribution: Dict[str, int],
        participating_agents: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log consensus calculation result.

        Args:
            cycle_id: Research cycle ID
            proposal_id: Proposal being voted on
            total_votes: Total number of votes cast
            weighted_score: Weighted consensus score
            consensus_reached: Whether consensus threshold was met
            final_decision: Final decision (approve/reject/revise)
            confidence: Consensus confidence
            vote_distribution: Vote counts by decision type
            participating_agents: List of agent IDs that voted
            metadata: Additional metadata
        """
        event_type = (
            LogEventType.CONSENSUS_REACHED.value if consensus_reached
            else LogEventType.CONSENSUS_FAILED.value
        )

        entry = ConsensusLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            cycle_id=cycle_id,
            proposal_id=proposal_id,
            total_votes=total_votes,
            weighted_score=weighted_score,
            consensus_reached=consensus_reached,
            final_decision=final_decision,
            confidence=confidence,
            vote_distribution=vote_distribution,
            participating_agents=participating_agents,
            metadata=metadata or {}
        )

        self._append_log(self.consensus_log, asdict(entry))
        logger.info(f"Logged consensus: {final_decision} (score: {weighted_score:.2f}, consensus: {consensus_reached})")

    def log_conflict(
        self,
        cycle_id: int,
        proposal_id: str,
        conflict_type: str,
        entropy: float,
        resolution_strategy: str,
        original_decision: str,
        final_decision: str,
        override_applied: bool,
        reasoning: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log conflict detection and resolution.

        Args:
            cycle_id: Research cycle ID
            proposal_id: Proposal with conflict
            conflict_type: Type of conflict (tie, controversial, etc.)
            entropy: Disagreement entropy measure
            resolution_strategy: Strategy used to resolve
            original_decision: Original consensus decision
            final_decision: Decision after resolution
            override_applied: Whether decision was overridden
            reasoning: Explanation of resolution
            metadata: Additional metadata
        """
        entry = ConflictLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            event_type=LogEventType.CONFLICT_RESOLVED.value,
            cycle_id=cycle_id,
            proposal_id=proposal_id,
            conflict_type=conflict_type,
            entropy=entropy,
            resolution_strategy=resolution_strategy,
            original_decision=original_decision,
            final_decision=final_decision,
            override_applied=override_applied,
            reasoning=reasoning,
            metadata=metadata or {}
        )

        self._append_log(self.conflict_log, asdict(entry))
        logger.warning(f"Logged conflict: {conflict_type} resolved via {resolution_strategy}")

    def log_supervisor_decision(
        self,
        cycle_id: int,
        proposal_id: str,
        supervisor_decision: str,
        risk_assessment: str,
        consensus_decision: str,
        override_consensus: bool,
        confidence: float,
        reasoning: str,
        constraints_violated: Optional[List[str]] = None,
        safety_concerns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log supervisor oversight decision.

        Args:
            cycle_id: Research cycle ID
            proposal_id: Proposal reviewed
            supervisor_decision: Supervisor's decision
            risk_assessment: Risk level (low/medium/high/critical)
            consensus_decision: What consensus decided
            override_consensus: Whether supervisor overrode consensus
            confidence: Supervisor confidence
            reasoning: Supervisor reasoning
            constraints_violated: Constraints that were violated
            safety_concerns: Safety issues identified
            metadata: Additional metadata
        """
        event_type = (
            LogEventType.SUPERVISOR_OVERRIDE.value if override_consensus
            else LogEventType.SUPERVISOR_APPROVAL.value
        )

        entry = SupervisorLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            cycle_id=cycle_id,
            proposal_id=proposal_id,
            supervisor_decision=supervisor_decision,
            risk_assessment=risk_assessment,
            consensus_decision=consensus_decision,
            override_consensus=override_consensus,
            confidence=confidence,
            reasoning=reasoning,
            constraints_violated=constraints_violated or [],
            safety_concerns=safety_concerns or [],
            metadata=metadata or {}
        )

        self._append_log(self.supervisor_log, asdict(entry))

        if override_consensus:
            logger.warning(f"Logged supervisor OVERRIDE: {consensus_decision} -> {supervisor_decision} (risk: {risk_assessment})")
        else:
            logger.info(f"Logged supervisor approval: {supervisor_decision}")

    def log_cycle_event(
        self,
        cycle_id: int,
        event_type: LogEventType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log cycle lifecycle event.

        Args:
            cycle_id: Research cycle ID
            event_type: Event type (CYCLE_STARTED, CYCLE_COMPLETED)
            metadata: Additional metadata
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "cycle_id": cycle_id,
            "metadata": metadata or {}
        }

        self._append_log(self.cycle_log, entry)
        logger.info(f"Logged cycle event: {event_type.value} (cycle {cycle_id})")

    def _append_log(self, log_file: Path, entry: Dict[str, Any]) -> None:
        """
        Append entry to JSONL log file.

        Args:
            log_file: Log file path
            entry: Log entry dictionary
        """
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write log entry: {e}")

    def query_votes(
        self,
        cycle_id: Optional[int] = None,
        agent_id: Optional[str] = None,
        proposal_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query vote logs.

        Args:
            cycle_id: Filter by cycle ID
            agent_id: Filter by agent ID
            proposal_id: Filter by proposal ID
            limit: Maximum entries to return

        Returns:
            List of matching vote entries
        """
        return self._query_log(
            self.vote_log,
            cycle_id=cycle_id,
            agent_id=agent_id,
            proposal_id=proposal_id,
            limit=limit
        )

    def query_consensus(
        self,
        cycle_id: Optional[int] = None,
        proposal_id: Optional[str] = None,
        consensus_reached: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query consensus logs.

        Args:
            cycle_id: Filter by cycle ID
            proposal_id: Filter by proposal ID
            consensus_reached: Filter by consensus outcome
            limit: Maximum entries to return

        Returns:
            List of matching consensus entries
        """
        entries = self._query_log(
            self.consensus_log,
            cycle_id=cycle_id,
            proposal_id=proposal_id,
            limit=limit
        )

        if consensus_reached is not None:
            entries = [e for e in entries if e.get('consensus_reached') == consensus_reached]

        return entries

    def query_supervisor_overrides(
        self,
        cycle_id: Optional[int] = None,
        risk_level: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query supervisor override logs.

        Args:
            cycle_id: Filter by cycle ID
            risk_level: Filter by risk assessment
            limit: Maximum entries to return

        Returns:
            List of supervisor override entries
        """
        entries = self._query_log(
            self.supervisor_log,
            cycle_id=cycle_id,
            limit=limit
        )

        # Filter for overrides only
        entries = [e for e in entries if e.get('override_consensus', False)]

        if risk_level:
            entries = [e for e in entries if e.get('risk_assessment') == risk_level]

        return entries

    def _query_log(
        self,
        log_file: Path,
        cycle_id: Optional[int] = None,
        agent_id: Optional[str] = None,
        proposal_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query JSONL log file.

        Args:
            log_file: Log file to query
            cycle_id: Filter by cycle ID
            agent_id: Filter by agent ID
            proposal_id: Filter by proposal ID
            limit: Maximum entries to return

        Returns:
            List of matching entries
        """
        if not log_file.exists():
            return []

        results = []

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())

                        # Apply filters
                        if cycle_id is not None and entry.get('cycle_id') != cycle_id:
                            continue
                        if agent_id and entry.get('agent_id') != agent_id:
                            continue
                        if proposal_id and entry.get('proposal_id') != proposal_id:
                            continue

                        results.append(entry)

                        if len(results) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error querying log file {log_file}: {e}")

        return results

    def get_voting_stats(self, cycle_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get voting statistics.

        Args:
            cycle_id: Filter by cycle ID (None = all cycles)

        Returns:
            Voting statistics dictionary
        """
        votes = self.query_votes(cycle_id=cycle_id, limit=10000)

        if not votes:
            return {
                "total_votes": 0,
                "by_agent": {},
                "by_decision": {},
                "avg_confidence": 0.0
            }

        by_agent = {}
        by_decision = {"approve": 0, "reject": 0, "revise": 0, "abstain": 0}
        total_confidence = 0.0

        for vote in votes:
            agent_id = vote.get('agent_id', 'unknown')
            decision = vote.get('decision', 'abstain')
            confidence = vote.get('confidence', 0.0)

            if agent_id not in by_agent:
                by_agent[agent_id] = {"total": 0, "decisions": {}}

            by_agent[agent_id]["total"] += 1
            by_agent[agent_id]["decisions"][decision] = by_agent[agent_id]["decisions"].get(decision, 0) + 1

            if decision in by_decision:
                by_decision[decision] += 1

            total_confidence += confidence

        return {
            "total_votes": len(votes),
            "by_agent": by_agent,
            "by_decision": by_decision,
            "avg_confidence": total_confidence / len(votes) if votes else 0.0
        }


# Global decision logger instance
_global_logger: Optional[DecisionLogger] = None


def get_decision_logger(log_dir: Optional[str] = None) -> DecisionLogger:
    """
    Get global decision logger instance.

    Args:
        log_dir: Log directory (None = use default)

    Returns:
        DecisionLogger instance
    """
    global _global_logger

    if _global_logger is None:
        if log_dir is None:
            log_dir = "/workspace/arc/memory/logs"
        _global_logger = DecisionLogger(log_dir)

    return _global_logger
