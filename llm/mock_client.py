"""
MockLLMClient: Offline mock LLM for development and testing
============================================================

Returns deterministic, structured responses without network calls.
Useful for:
- Offline development
- Testing multi-agent infrastructure
- Dashboard development
- CI/CD pipelines
"""

import json
from typing import Dict, Any, Optional, List
import random


class MockLLMClient:
    """
    Mock LLM client that returns deterministic responses offline.

    No network calls, no model dependencies, fully deterministic.
    """

    def __init__(
        self,
        model_name: str = "mock-llm",
        seed: int = 42
    ):
        """
        Initialize mock LLM client.

        Args:
            model_name: Mock model identifier
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.endpoint = "mock://offline"
        random.seed(seed)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate mock text response.

        Returns structured JSON based on prompt keywords.

        Args:
            prompt: Input prompt
            max_tokens: Ignored (mock)
            temperature: Ignored (mock)
            stop_sequences: Ignored (mock)
            **kwargs: Ignored (mock)

        Returns:
            Mock response string
        """
        prompt_lower = prompt.lower()

        # Detect role from prompt and return appropriate mock response
        if "director" in prompt_lower or "strategic" in prompt_lower:
            return self._mock_director_response()
        elif "architect" in prompt_lower or "proposal" in prompt_lower:
            return self._mock_architect_response()
        elif "critic" in prompt_lower or "review" in prompt_lower or "safety" in prompt_lower:
            return self._mock_critic_response()
        elif "historian" in prompt_lower or "history" in prompt_lower:
            return self._mock_historian_response()
        elif "executor" in prompt_lower or "training" in prompt_lower:
            return self._mock_executor_response()
        elif "supervisor" in prompt_lower or "validate" in prompt_lower:
            return self._mock_supervisor_response()
        else:
            return self._mock_generic_response()

    def generate_json(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate mock JSON response.

        Args:
            prompt: Input prompt
            max_tokens: Ignored (mock)
            temperature: Ignored (mock)

        Returns:
            Mock JSON dictionary
        """
        response = self.generate(prompt, max_tokens, temperature)
        # Response is already valid JSON
        return json.loads(response)

    def health_check(self) -> bool:
        """Mock health check (always healthy)."""
        return True

    def _mock_director_response(self) -> str:
        """Mock Director response."""
        response = {
            "mode": "explore",
            "objective": "establish_baseline_and_explore_parameter_space",
            "novelty_budget": {
                "exploit": 3,
                "explore": 2,
                "wildcat": 1
            },
            "focus_areas": ["learning_rate_tuning", "batch_size_optimization"],
            "forbidden_axes": [],
            "encouraged_axes": ["stable_hyperparameters"],
            "notes": "Mock strategic directive - focus on foundational experiments"
        }
        return json.dumps(response, indent=2)

    def _mock_architect_response(self) -> str:
        """Mock Architect response."""
        response = {
            "proposals": [
                {
                    "experiment_id": "exp_mock_001",
                    "name": "baseline_lr_0.001",
                    "hypothesis": "Standard learning rate provides stable baseline",
                    "novelty_category": "exploit",
                    "predicted_metrics": {"auc": 0.82, "sensitivity": 0.75, "specificity": 0.80},
                    "config_changes": {"learning_rate": 0.001, "batch_size": 32},
                    "justification": "Mock baseline experiment"
                },
                {
                    "experiment_id": "exp_mock_002",
                    "name": "higher_lr_0.01",
                    "hypothesis": "Higher learning rate accelerates convergence",
                    "novelty_category": "explore",
                    "predicted_metrics": {"auc": 0.78, "sensitivity": 0.72, "specificity": 0.76},
                    "config_changes": {"learning_rate": 0.01, "batch_size": 32},
                    "justification": "Mock exploration experiment"
                }
            ]
        }
        return json.dumps(response, indent=2)

    def _mock_critic_response(self) -> str:
        """Mock Critic response."""
        response = {
            "reviews": [
                {
                    "experiment_id": "exp_mock_001",
                    "decision": "approve",
                    "confidence": 0.95,
                    "reasoning": "Mock approval - baseline experiment is low risk",
                    "risk_level": "low"
                },
                {
                    "experiment_id": "exp_mock_002",
                    "decision": "approve",
                    "confidence": 0.85,
                    "reasoning": "Mock approval - exploration within safe bounds",
                    "risk_level": "medium"
                }
            ]
        }
        return json.dumps(response, indent=2)

    def _mock_historian_response(self) -> str:
        """Mock Historian response."""
        response = {
            "total_cycles": 5,
            "total_experiments": 12,
            "best_metrics": {
                "auc": 0.87,
                "sensitivity": 0.82,
                "specificity": 0.85
            },
            "recent_experiments": [
                {"name": "exp_004_lr_tuning", "auc": 0.85, "status": "success"},
                {"name": "exp_005_batch_opt", "auc": 0.87, "status": "success"}
            ],
            "failed_configs": [
                {"learning_rate": 0.1, "reason": "unstable_training"}
            ],
            "successful_patterns": [
                {"learning_rate_range": [0.001, 0.01], "batch_size": 32}
            ],
            "notes": "Mock history summary - system learning effectively"
        }
        return json.dumps(response, indent=2)

    def _mock_executor_response(self) -> str:
        """Mock Executor response."""
        response = {
            "status": "queued",
            "experiment_id": "exp_mock_001",
            "training_command": "python train.py --lr 0.001 --batch_size 32",
            "estimated_duration_minutes": 45,
            "notes": "Mock executor - training job queued"
        }
        return json.dumps(response, indent=2)

    def _mock_supervisor_response(self) -> str:
        """Mock Supervisor response."""
        response = {
            "decision": "approve",
            "risk_assessment": "low",
            "reasoning": "Mock supervisor approval - proposals within safety bounds",
            "constraints_violated": [],
            "override_consensus": False,
            "approved_experiments": ["exp_mock_001", "exp_mock_002"],
            "notes": "Mock validation passed"
        }
        return json.dumps(response, indent=2)

    def _mock_generic_response(self) -> str:
        """Mock generic response."""
        response = {
            "status": "success",
            "message": "Mock LLM response - operation completed",
            "data": {"mock": True}
        }
        return json.dumps(response, indent=2)

    def __repr__(self) -> str:
        return f"<MockLLMClient model={self.model_name} offline=True>"
