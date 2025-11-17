"""
Inter-Agent Communication Protocol
===================================

Defines message schemas and communication patterns for multi-agent coordination.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
import json


class MessageType(Enum):
    """Types of inter-agent messages"""
    PROPOSAL = "proposal"           # Experiment proposal
    VOTE = "vote"                   # Vote on proposal
    APPROVAL = "approval"           # Supervisor approval
    REJECTION = "rejection"         # Supervisor rejection
    REVISION_REQUEST = "revision_request"  # Request changes
    NOTIFICATION = "notification"   # General notification
    QUERY = "query"                 # Information request
    RESPONSE = "response"           # Response to query


class RiskLevel(Enum):
    """Risk levels for proposals"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentMessage:
    """
    Standard message format for inter-agent communication.

    All agents use this format to exchange information.
    """
    message_id: str                  # Unique message identifier
    message_type: MessageType        # Type of message
    sender_id: str                   # Sending agent ID
    recipient_id: Optional[str]      # Target agent ID (None = broadcast)
    timestamp: str                   # ISO-8601 timestamp
    cycle_id: int                    # Current research cycle
    payload: Dict[str, Any]          # Message-specific data
    metadata: Dict[str, Any]         # Optional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["message_type"] = self.message_type.value
        return d

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create from dictionary."""
        data["message_type"] = MessageType(data["message_type"])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentMessage":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ProposalMessage:
    """
    Proposal message payload schema.

    Used by Architect agents to propose experiments.
    """
    proposal_id: str
    experiment_name: str
    hypothesis: str
    novelty_category: str  # exploit, explore, wildcat
    predicted_metrics: Dict[str, float]
    config_changes: Dict[str, Any]
    risk_level: RiskLevel
    justification: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        return d


@dataclass
class VoteMessage:
    """
    Vote message payload schema.

    Used by Critic agents to vote on proposals.
    """
    proposal_id: str
    decision: str  # approve, reject, revise
    confidence: float  # 0.0-1.0
    reasoning: str
    suggested_changes: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SupervisorDecision:
    """
    Supervisor decision payload schema.

    Used by Supervisor to approve/reject proposals.
    """
    proposal_id: str
    decision: str  # approve, reject, revise, override
    risk_assessment: RiskLevel
    reasoning: str
    constraints_violated: List[str]
    override_consensus: bool  # True if overriding democratic vote

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["risk_assessment"] = self.risk_assessment.value
        return d


class MessageQueue:
    """
    Simple file-based message queue for inter-agent communication.

    Uses JSONL format for append-only message log.
    """

    def __init__(self, queue_path: str = "/workspace/arc/memory/messages.jsonl"):
        """
        Initialize message queue.

        Args:
            queue_path: Path to message queue file
        """
        self.queue_path = queue_path

    def send(self, message: AgentMessage) -> bool:
        """
        Send a message (append to queue).

        Args:
            message: Message to send

        Returns:
            True if send successful
        """
        try:
            import os
            os.makedirs(os.path.dirname(self.queue_path), exist_ok=True)

            with open(self.queue_path, 'a') as f:
                f.write(message.to_json() + '\n')
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            return False

    def receive(
        self,
        recipient_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        since_timestamp: Optional[str] = None
    ) -> List[AgentMessage]:
        """
        Receive messages from queue (read with filters).

        Args:
            recipient_id: Filter by recipient (None = all broadcasts)
            message_type: Filter by message type
            since_timestamp: Only messages after this timestamp

        Returns:
            List of matching messages
        """
        try:
            messages = []

            with open(self.queue_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    msg = AgentMessage.from_json(line)

                    # Apply filters
                    if recipient_id and msg.recipient_id != recipient_id:
                        continue
                    if message_type and msg.message_type != message_type:
                        continue
                    if since_timestamp and msg.timestamp < since_timestamp:
                        continue

                    messages.append(msg)

            return messages
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error receiving messages: {e}")
            return []

    def clear(self) -> bool:
        """
        Clear message queue (delete file).

        Returns:
            True if cleared successfully
        """
        try:
            import os
            if os.path.exists(self.queue_path):
                os.remove(self.queue_path)
            return True
        except Exception as e:
            print(f"Error clearing queue: {e}")
            return False


def create_proposal_message(
    sender_id: str,
    cycle_id: int,
    proposal: ProposalMessage
) -> AgentMessage:
    """
    Helper to create a proposal message.

    Args:
        sender_id: ID of architect agent
        cycle_id: Current cycle
        proposal: Proposal payload

    Returns:
        Formatted AgentMessage
    """
    import uuid
    return AgentMessage(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.PROPOSAL,
        sender_id=sender_id,
        recipient_id=None,  # Broadcast
        timestamp=datetime.now().isoformat(),
        cycle_id=cycle_id,
        payload=proposal.to_dict(),
        metadata={}
    )


def create_vote_message(
    sender_id: str,
    recipient_id: str,
    cycle_id: int,
    vote: VoteMessage
) -> AgentMessage:
    """
    Helper to create a vote message.

    Args:
        sender_id: ID of critic agent
        recipient_id: ID of supervisor or orchestrator
        cycle_id: Current cycle
        vote: Vote payload

    Returns:
        Formatted AgentMessage
    """
    import uuid
    return AgentMessage(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.VOTE,
        sender_id=sender_id,
        recipient_id=recipient_id,
        timestamp=datetime.now().isoformat(),
        cycle_id=cycle_id,
        payload=vote.to_dict(),
        metadata={}
    )


def create_supervisor_decision_message(
    sender_id: str,
    cycle_id: int,
    decision: SupervisorDecision
) -> AgentMessage:
    """
    Helper to create a supervisor decision message.

    Args:
        sender_id: Supervisor agent ID
        cycle_id: Current cycle
        decision: Decision payload

    Returns:
        Formatted AgentMessage
    """
    import uuid
    msg_type = (
        MessageType.APPROVAL if decision.decision == "approve"
        else MessageType.REJECTION if decision.decision == "reject"
        else MessageType.REVISION_REQUEST
    )

    return AgentMessage(
        message_id=str(uuid.uuid4()),
        message_type=msg_type,
        sender_id=sender_id,
        recipient_id=None,  # Broadcast
        timestamp=datetime.now().isoformat(),
        cycle_id=cycle_id,
        payload=decision.to_dict(),
        metadata={}
    )
