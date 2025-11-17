"""
Mock LLM Client for Testing

Provides mock implementations of LLM clients that simulate responses
without requiring actual LLM backend. Supports all ARC roles with
configurable responses and failure modes.

Usage:
    from tests.mocks.llm_client import MockLLMClient

    client = MockLLMClient()
    response = client.chat_completion("What is ARC?")

    # With role-specific responses
    client = MockLLMClient(role="historian")
    response = client.chat_completion("Analyze history")
"""

import json
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from enum import Enum


class MockMode(str, Enum):
    """Mock LLM behavior modes."""
    SUCCESS = "success"          # Always return valid responses
    FAILURE = "failure"          # Always fail with errors
    TIMEOUT = "timeout"          # Simulate timeout errors
    INVALID_JSON = "invalid_json"  # Return malformed JSON
    PARTIAL = "partial"          # Return incomplete responses
    RANDOM = "random"            # Random behavior for stress testing


class MockLLMClient:
    """
    Mock LLM client that simulates ARC role responses.

    Supports all five ARC roles (Historian, Director, Architect, Critic, Executor)
    with realistic JSON responses. Can simulate various failure modes for testing.

    Attributes:
        role: Optional ARC role to specialize responses
        mode: Behavior mode (success, failure, etc.)
        call_count: Number of times chat_completion was called
        call_history: List of all messages received
    """

    def __init__(
        self,
        role: Optional[Literal["historian", "director", "architect", "critic", "executor"]] = None,
        mode: MockMode = MockMode.SUCCESS,
        delay: float = 0.0
    ):
        """
        Initialize mock LLM client.

        Args:
            role: Specialize responses for specific ARC role
            mode: Behavior mode for testing different scenarios
            delay: Simulated response delay in seconds
        """
        self.role = role
        self.mode = mode
        self.delay = delay
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
        self._responses = self._load_role_responses()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simulate chat completion API call.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (ignored in mock)
            max_tokens: Maximum tokens (ignored in mock)
            **kwargs: Additional parameters (ignored in mock)

        Returns:
            Mock API response in OpenAI format

        Raises:
            TimeoutError: If mode is TIMEOUT
            ValueError: If mode is FAILURE
        """
        import time

        self.call_count += 1
        self.call_history.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Simulate delay
        if self.delay > 0:
            time.sleep(self.delay)

        # Handle failure modes
        if self.mode == MockMode.FAILURE:
            raise ValueError("Mock LLM failure")
        elif self.mode == MockMode.TIMEOUT:
            raise TimeoutError("Mock LLM timeout")

        # Extract user message
        user_message = next(
            (msg["content"] for msg in messages if msg["role"] == "user"),
            ""
        )

        # Generate response based on role and mode
        response_content = self._generate_response(user_message)

        # Return OpenAI-style response
        return {
            "id": f"mock-{self.call_count}",
            "object": "chat.completion",
            "created": int(datetime.utcnow().timestamp()),
            "model": "mock-llm",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }
        }

    def _generate_response(self, user_message: str) -> str:
        """
        Generate mock response based on role and mode.

        Args:
            user_message: User's message content

        Returns:
            Generated response string
        """
        if self.mode == MockMode.INVALID_JSON:
            return '{"incomplete": "json", broken'

        if self.mode == MockMode.PARTIAL:
            return self._get_partial_response()

        # Return role-specific response
        if self.role in self._responses:
            return json.dumps(self._responses[self.role], indent=2)

        # Default generic response
        return json.dumps({
            "status": "success",
            "message": "Mock LLM response",
            "role": self.role or "generic",
            "received_message": user_message[:100]
        }, indent=2)

    def _load_role_responses(self) -> Dict[str, Dict[str, Any]]:
        """Load pre-defined responses for each ARC role."""
        return {
            "historian": self._get_historian_response(),
            "director": self._get_director_response(),
            "architect": self._get_architect_response(),
            "critic": self._get_critic_response(),
            "executor": self._get_executor_response()
        }

    def _get_historian_response(self) -> Dict[str, Any]:
        """Generate historian role response."""
        return {
            "history_summary": {
                "total_cycles": 10,
                "total_experiments": 30,
                "best_metrics": {
                    "auc": 0.89,
                    "sensitivity": 0.87,
                    "specificity": 0.91
                },
                "recent_experiments": [
                    {
                        "experiment_id": "exp_10_1",
                        "auc": 0.89,
                        "training_time": 145.2,
                        "timestamp": datetime.utcnow().isoformat(),
                        "success": True
                    }
                ],
                "failed_configs": [],
                "successful_patterns": [
                    {"learning_rate": 0.001, "batch_size": 32}
                ],
                "performance_trends": {
                    "auc_trend": "improving",
                    "sensitivity_trend": "stable",
                    "specificity_trend": "improving",
                    "cycles_without_improvement": 0,
                    "consecutive_regressions": 0
                },
                "last_updated": datetime.utcnow().isoformat()
            },
            "constraints": {
                "forbidden_ranges": [
                    {
                        "param": "learning_rate",
                        "min": 0.1,
                        "max": 1.0,
                        "reason": "Causes training instability"
                    }
                ],
                "unstable_configs": [],
                "safe_baselines": [
                    {"learning_rate": 0.001, "batch_size": 32}
                ],
                "max_learning_rate": 0.1,
                "min_batch_size": 8,
                "max_batch_size": 256,
                "last_updated": datetime.utcnow().isoformat()
            },
            "performance_trends": {
                "auc_trend": "improving",
                "sensitivity_trend": "stable",
                "specificity_trend": "improving"
            }
        }

    def _get_director_response(self) -> Dict[str, Any]:
        """Generate director role response."""
        return {
            "cycle_id": 11,
            "mode": "explore",
            "objective": "improve_auc",
            "novelty_budget": {
                "exploit": 3,
                "explore": 2,
                "wildcat": 1
            },
            "focus_areas": ["learning_rate", "architecture", "optimizer"],
            "forbidden_axes": ["dataset", "num_epochs"],
            "encouraged_axes": ["dropout", "weight_decay"],
            "notes": "Continue exploration. AUC trending upward, try architectural variations.",
            "timestamp": datetime.utcnow().isoformat()
        }

    def _get_architect_response(self) -> Dict[str, Any]:
        """Generate architect role response."""
        return {
            "cycle_id": 11,
            "proposals": [
                {
                    "experiment_id": "exp_11_1",
                    "novelty_class": "exploit",
                    "hypothesis": "Fine-tuning learning rate will improve convergence",
                    "changes": {
                        "learning_rate": 0.0008
                    },
                    "expected_impact": {
                        "auc": "up",
                        "sensitivity": "same",
                        "specificity": "same"
                    },
                    "resource_cost": "low",
                    "rationale": "Previous experiments show 0.001 works well, slightly lower may be better"
                },
                {
                    "experiment_id": "exp_11_2",
                    "novelty_class": "explore",
                    "hypothesis": "Adding batch normalization will stabilize training",
                    "changes": {
                        "use_batch_norm": True,
                        "learning_rate": 0.001
                    },
                    "expected_impact": {
                        "auc": "up",
                        "sensitivity": "up",
                        "specificity": "up"
                    },
                    "resource_cost": "medium",
                    "rationale": "Batch norm improves generalization in similar architectures"
                },
                {
                    "experiment_id": "exp_11_3",
                    "novelty_class": "wildcat",
                    "hypothesis": "Switching to EfficientNet backbone may boost performance",
                    "changes": {
                        "architecture": "efficientnet_b0",
                        "learning_rate": 0.001,
                        "batch_size": 64
                    },
                    "expected_impact": {
                        "auc": "up",
                        "sensitivity": "unknown",
                        "specificity": "unknown"
                    },
                    "resource_cost": "high",
                    "rationale": "EfficientNet shows strong results in medical imaging tasks"
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

    def _get_critic_response(self) -> Dict[str, Any]:
        """Generate critic role response."""
        return {
            "cycle_id": 11,
            "reviews": [
                {
                    "proposal_id": "exp_11_1",
                    "decision": "approve",
                    "issues": [],
                    "reasoning": "Low-risk refinement with clear hypothesis. Historical data supports this direction.",
                    "risk_level": "low"
                },
                {
                    "proposal_id": "exp_11_2",
                    "decision": "approve",
                    "issues": ["May increase training time by 15-20%"],
                    "reasoning": "Solid architectural improvement. Resource cost acceptable for potential gain.",
                    "risk_level": "medium"
                },
                {
                    "proposal_id": "exp_11_3",
                    "decision": "revise",
                    "issues": [
                        "High resource cost",
                        "Untested architecture in this domain",
                        "May require hyperparameter tuning"
                    ],
                    "reasoning": "Interesting idea but needs more justification. Suggest pilot study first.",
                    "risk_level": "high"
                }
            ],
            "approved": ["exp_11_1", "exp_11_2"],
            "rejected": [],
            "revise": ["exp_11_3"],
            "timestamp": datetime.utcnow().isoformat()
        }

    def _get_executor_response(self) -> Dict[str, Any]:
        """Generate executor role response."""
        return {
            "experiments": [
                {
                    "experiment_id": "exp_11_1",
                    "config_patch": {
                        "learning_rate": 0.0008
                    },
                    "commands": [
                        "python train.py --config experiments/exp_11_1/config.yaml",
                        "python evaluate.py --experiment exp_11_1"
                    ],
                    "status": "ready"
                },
                {
                    "experiment_id": "exp_11_2",
                    "config_patch": {
                        "use_batch_norm": True,
                        "learning_rate": 0.001
                    },
                    "commands": [
                        "python train.py --config experiments/exp_11_2/config.yaml",
                        "python evaluate.py --experiment exp_11_2"
                    ],
                    "status": "ready"
                }
            ],
            "validation": "passed",
            "timestamp": datetime.utcnow().isoformat()
        }

    def _get_partial_response(self) -> str:
        """Generate partial/incomplete response for testing error handling."""
        return '{"cycle_id": 11, "mode": "explore", "objective":'

    def reset(self) -> None:
        """Reset call history and counters."""
        self.call_count = 0
        self.call_history = []

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get the last call made to this client."""
        return self.call_history[-1] if self.call_history else None

    def set_mode(self, mode: MockMode) -> None:
        """Change the mock behavior mode."""
        self.mode = mode

    def set_role(self, role: Optional[str]) -> None:
        """Change the role specialization."""
        self.role = role
        self._responses = self._load_role_responses()


# ============================================================================
# Mock HTTP Requests (for orchestrators using requests.post)
# ============================================================================

class MockLLMResponse:
    """Mock HTTP response for requests library."""

    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        """
        Initialize mock response.

        Args:
            json_data: Response JSON data
            status_code: HTTP status code
        """
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
        self.ok = 200 <= status_code < 300

    def json(self) -> Dict[str, Any]:
        """Return JSON data."""
        return self._json_data

    def raise_for_status(self) -> None:
        """Raise HTTPError if status indicates error."""
        if not self.ok:
            raise Exception(f"HTTP {self.status_code}")


def mock_llm_request(
    url: str,
    json_data: Optional[Dict[str, Any]] = None,
    role: Optional[str] = None,
    mode: MockMode = MockMode.SUCCESS,
    **kwargs
) -> MockLLMResponse:
    """
    Mock requests.post() for LLM API calls.

    Use with pytest monkeypatch:
        monkeypatch.setattr("requests.post", mock_llm_request)

    Args:
        url: Request URL
        json_data: Request JSON payload
        role: ARC role to mock
        mode: Mock behavior mode
        **kwargs: Additional request parameters

    Returns:
        MockLLMResponse object
    """
    client = MockLLMClient(role=role, mode=mode)

    # Extract messages from request
    messages = json_data.get("messages", []) if json_data else []

    try:
        response = client.chat_completion(messages)
        return MockLLMResponse(response, status_code=200)
    except TimeoutError:
        return MockLLMResponse({"error": "timeout"}, status_code=504)
    except Exception as e:
        return MockLLMResponse({"error": str(e)}, status_code=500)


# ============================================================================
# Pytest Fixtures (imported by conftest.py)
# ============================================================================

def get_mock_llm_client(role: Optional[str] = None, mode: MockMode = MockMode.SUCCESS):
    """Factory function for creating mock LLM clients."""
    return MockLLMClient(role=role, mode=mode)
