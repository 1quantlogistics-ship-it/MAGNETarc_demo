"""
BaseAgent: Abstract base class for all ARC agents
==================================================

Defines the standard interface for agent lifecycle, communication, and voting.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import time


class AgentState(Enum):
    """Agent operational states"""
    INACTIVE = "inactive"      # Not started
    ACTIVE = "active"          # Ready and idle
    BUSY = "busy"              # Processing task
    FAILED = "failed"          # Error state
    OFFLINE = "offline"        # Intentionally offline (for local models)


class AgentCapability(Enum):
    """Agent capability tags for role-based routing"""
    STRATEGY = "strategy"                    # Strategic planning
    PROPOSAL_GENERATION = "proposal_generation"  # Experiment design
    SAFETY_REVIEW = "safety_review"         # Risk assessment
    CONSTRAINT_CHECKING = "constraint_checking"  # Validation
    MEMORY_MANAGEMENT = "memory_management" # History tracking
    EXECUTION = "execution"                 # Training execution
    EXPLORATION = "exploration"             # Parameter space exploration
    VALIDATION = "validation"               # Schema/sanity checking
    SUPERVISION = "supervision"             # Meta-level oversight


class BaseAgent(ABC):
    """
    Abstract base class for all ARC agents.

    All agents must implement:
    - process(): Core agent logic
    - vote_on_proposal(): Consensus participation
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        model: str,
        capabilities: List[AgentCapability],
        voting_weight: float = 1.0,
        priority: str = "medium",
        offline: bool = False,
        memory_path: str = "/workspace/arc/memory"
    ):
        """
        Initialize an ARC agent.

        Args:
            agent_id: Unique identifier (e.g., "director_001")
            role: Agent role (director, architect, critic, etc.)
            model: LLM model identifier (e.g., "claude-sonnet-4.5")
            capabilities: List of AgentCapability tags
            voting_weight: Weight in consensus votes (higher = more influence)
            priority: Priority level (low, medium, high, critical)
            offline: Whether agent operates offline (no network)
            memory_path: Path to shared memory directory
        """
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.capabilities = capabilities
        self.voting_weight = voting_weight
        self.priority = priority
        self.offline = offline
        self.memory_path = memory_path

        # State tracking
        self.state = AgentState.INACTIVE
        self.last_activity = None
        self.current_task = None

        # Performance metrics
        self.metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_response_time_ms": 0.0,
            "total_votes": 0,
            "vote_agreement_rate": 0.0
        }

    def activate(self) -> bool:
        """
        Activate the agent (transition to ACTIVE state).

        Returns:
            True if activation successful
        """
        if self.state == AgentState.FAILED:
            return False
        self.state = AgentState.ACTIVE
        self.last_activity = datetime.now().isoformat()
        return True

    def deactivate(self) -> None:
        """Deactivate the agent (transition to INACTIVE state)."""
        self.state = AgentState.INACTIVE
        self.current_task = None

    def health_check(self) -> Dict[str, Any]:
        """
        Check agent health status.

        Returns:
            Health report with state, metrics, and status
        """
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "state": self.state.value,
            "last_activity": self.last_activity,
            "current_task": self.current_task,
            "metrics": self.metrics,
            "healthy": self.state in [AgentState.ACTIVE, AgentState.BUSY]
        }

    def read_memory(self, filename: str) -> Dict[str, Any]:
        """
        Read from shared memory file.

        Args:
            filename: Memory file to read (e.g., "directive.json")

        Returns:
            Parsed JSON content
        """
        import os
        filepath = os.path.join(self.memory_path, filename)

        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}

    def write_memory(self, filename: str, data: Dict[str, Any]) -> bool:
        """
        Write to shared memory file.

        Args:
            filename: Memory file to write
            data: Data to write (will be JSON-serialized)

        Returns:
            True if write successful
        """
        import os
        filepath = os.path.join(self.memory_path, filename)

        try:
            os.makedirs(self.memory_path, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error writing memory {filename}: {e}")
            return False

    def log_decision(self, decision_type: str, data: Dict[str, Any]) -> None:
        """
        Log a decision to the decision history.

        Args:
            decision_type: Type of decision (vote, approval, override, etc.)
            data: Decision data
        """
        import os
        decisions_path = os.path.join(self.memory_path, "decisions")
        os.makedirs(decisions_path, exist_ok=True)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "role": self.role,
            "decision_type": decision_type,
            **data
        }

        # Append to JSONL file
        log_file = os.path.join(decisions_path, f"{decision_type}.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core agent processing logic.

        Args:
            input_data: Input data for processing

        Returns:
            Processed output
        """
        pass

    @abstractmethod
    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vote on a proposal (for consensus mechanism).

        Args:
            proposal: Proposal to evaluate

        Returns:
            Vote decision with reasoning
            {
                "decision": "approve" | "reject" | "revise",
                "confidence": 0.0-1.0,
                "reasoning": "...",
                "suggested_changes": {...} (optional)
            }
        """
        pass

    def _track_task(self, task_name: str, success: bool, duration_ms: float) -> None:
        """
        Internal method to track task execution metrics.

        Args:
            task_name: Name of task executed
            success: Whether task succeeded
            duration_ms: Execution time in milliseconds
        """
        self.metrics["total_tasks"] += 1
        if success:
            self.metrics["successful_tasks"] += 1
        else:
            self.metrics["failed_tasks"] += 1

        # Update average response time
        total = self.metrics["total_tasks"]
        prev_avg = self.metrics["avg_response_time_ms"]
        self.metrics["avg_response_time_ms"] = (
            (prev_avg * (total - 1) + duration_ms) / total
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"id={self.agent_id} "
            f"role={self.role} "
            f"state={self.state.value}>"
        )
