"""
BaseNavalAgent: Abstract base class for MAGNET naval design agents
===================================================================

Adapted from ARC's BaseAgent for naval autonomous research.
Defines the standard interface for naval agent lifecycle and LLM interaction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import time
import re


class AgentState(Enum):
    """Agent operational states"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    BUSY = "busy"
    FAILED = "failed"
    OFFLINE = "offline"


class AgentCapability(Enum):
    """Agent capability tags for naval domain"""
    HYPOTHESIS_GENERATION = "hypothesis_generation"  # Explorer
    EXPERIMENTAL_DESIGN = "experimental_design"      # Architect
    ANALYSIS = "analysis"                            # Critic
    SUPERVISION = "supervision"                      # Supervisor
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"      # Director


@dataclass
class NavalAgentConfig:
    """Configuration for a naval agent"""
    agent_id: str
    role: str  # explorer, architect, critic, supervisor, director
    model: str  # Model identifier (e.g., "local-deepseek")
    voting_weight: float = 1.0
    priority: str = "medium"
    offline: bool = False
    memory_path: str = "/workspace/magnet/memory"
    gpu_id: Optional[int] = None  # For GPU assignment


@dataclass
class NavalAgentResponse:
    """Standard response format for naval agents"""
    agent_id: str
    action: str  # e.g., "submit_hypothesis", "submit_experiments", "submit_critique"
    reasoning: str  # LLM-generated reasoning
    confidence: float  # 0.0-1.0
    data: Dict[str, Any]  # Action-specific data
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class BaseNavalAgent(ABC):
    """
    Abstract base class for all MAGNET naval agents.

    All agents must implement:
    - autonomous_cycle(): Core agent logic for one research cycle
    """

    def __init__(self, config: NavalAgentConfig, llm_client):
        """
        Initialize a naval agent.

        Args:
            config: Agent configuration
            llm_client: LLM client instance for generation
        """
        self.config = config
        self.llm = llm_client

        # Extract config fields
        self.agent_id = config.agent_id
        self.role = config.role
        self.model = config.model
        self.voting_weight = config.voting_weight
        self.memory_path = config.memory_path

        # State tracking
        self.state = AgentState.INACTIVE
        self.last_activity: Optional[str] = None
        self.current_task: Optional[str] = None

        # Performance metrics
        self.metrics = {
            "total_cycles": 0,
            "successful_cycles": 0,
            "failed_cycles": 0,
            "avg_response_time_ms": 0.0,
            "total_llm_calls": 0,
            "avg_tokens_per_call": 0.0
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
            filename: Memory file to read (e.g., "knowledge_base.json")

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
            decision_type: Type of decision
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
    def autonomous_cycle(self, context: Dict[str, Any]) -> NavalAgentResponse:
        """
        Core agent logic for one autonomous research cycle.

        Args:
            context: Context data for the cycle (e.g., knowledge_base, current_best, etc.)

        Returns:
            NavalAgentResponse with action and data
        """
        pass

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text from LLM.

        Args:
            prompt: Prompt to send to LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        start_time = time.time()
        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            duration_ms = (time.time() - start_time) * 1000
            self._track_llm_call(duration_ms, len(response.split()))
            return response
        except Exception as e:
            print(f"Error generating from LLM: {e}")
            raise

    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON.

        Handles various formats:
        - Pure JSON
        - JSON in code blocks (```json ... ```)
        - JSON with <think> tags (DeepSeek-R1 style)

        Args:
            response: Raw LLM response text

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If no valid JSON found
        """
        # Remove <think> tags if present (DeepSeek-R1 thinking)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        # Try to find JSON in code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in LLM response")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}")

    def _track_llm_call(self, duration_ms: float, token_count: int) -> None:
        """
        Track LLM call metrics.

        Args:
            duration_ms: Call duration in milliseconds
            token_count: Approximate token count
        """
        self.metrics["total_llm_calls"] += 1
        total = self.metrics["total_llm_calls"]
        prev_avg = self.metrics["avg_tokens_per_call"]
        self.metrics["avg_tokens_per_call"] = (
            (prev_avg * (total - 1) + token_count) / total
        )

    def _track_cycle(self, success: bool, duration_ms: float) -> None:
        """
        Track cycle execution metrics.

        Args:
            success: Whether cycle succeeded
            duration_ms: Execution time in milliseconds
        """
        self.metrics["total_cycles"] += 1
        if success:
            self.metrics["successful_cycles"] += 1
        else:
            self.metrics["failed_cycles"] += 1

        # Update average response time
        total = self.metrics["total_cycles"]
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
