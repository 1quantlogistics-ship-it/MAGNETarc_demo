"""
LocalLLMClient: vLLM client for local DeepSeek-R1 inference
============================================================

Client for interacting with locally-hosted DeepSeek-R1 via vLLM.
Provides OpenAI-compatible API interface with connection pooling and retry logic.
"""

import json
import time
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class LocalLLMConfig:
    """Configuration for local LLM client"""
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    endpoint: str = "http://localhost:8000/v1/completions"
    api_key: Optional[str] = None  # Not required for local vLLM
    timeout: int = 120  # seconds
    max_retries: int = 3
    retry_delay: float = 2.0  # seconds


class LocalLLMClient:
    """
    Client for local vLLM server (DeepSeek-R1).

    Provides:
    - Text generation with configurable parameters
    - JSON extraction from responses
    - Health checking
    - Retry logic with exponential backoff
    """

    def __init__(self, config: Optional[LocalLLMConfig] = None):
        """
        Initialize local LLM client.

        Args:
            config: Client configuration (uses defaults if None)
        """
        self.config = config or LocalLLMConfig()
        self.session = requests.Session()
        self.total_calls = 0
        self.failed_calls = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            Generated text

        Raises:
            RuntimeError: If generation fails after retries
        """
        self.total_calls += 1

        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop or []
        }

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    self.config.endpoint,
                    json=payload,
                    timeout=self.config.timeout,
                    headers=self._get_headers()
                )

                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["text"]
                    return text

                elif response.status_code == 503:
                    # Service unavailable - retry
                    print(f"LLM service unavailable (503), retrying in {self.config.retry_delay}s...")
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue

                else:
                    error_msg = f"LLM API error: {response.status_code} - {response.text}"
                    print(error_msg)
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    else:
                        raise RuntimeError(error_msg)

            except requests.exceptions.Timeout:
                print(f"LLM request timeout (attempt {attempt + 1}/{self.config.max_retries})")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    self.failed_calls += 1
                    raise RuntimeError("LLM request timeout after retries")

            except requests.exceptions.ConnectionError:
                print(f"LLM connection error (attempt {attempt + 1}/{self.config.max_retries})")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    self.failed_calls += 1
                    raise RuntimeError("LLM connection error after retries")

            except Exception as e:
                print(f"Unexpected error during LLM call: {e}")
                self.failed_calls += 1
                raise

        self.failed_calls += 1
        raise RuntimeError("Failed to generate response after all retries")

    def generate_json(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate and parse JSON response.

        Args:
            prompt: Input prompt (should request JSON output)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If no valid JSON found in response
        """
        response_text = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return self.extract_json(response_text)

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response text.

        Handles various formats:
        - Pure JSON objects
        - JSON in markdown code blocks (```json ... ```)
        - JSON with <think> tags (DeepSeek-R1 reasoning)
        - JSON in backticks

        Args:
            text: Raw LLM response text

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If no valid JSON found
        """
        # Remove <think> tags if present (DeepSeek-R1 thinking process)
        text_cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Try to find JSON in code blocks
        code_block_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'`(\{.*?\})`'
        ]

        for pattern in code_block_patterns:
            match = re.search(pattern, text_cleaned, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        # Try to find raw JSON object
        json_match = re.search(r'\{.*\}', text_cleaned, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Try to find JSON array
        array_match = re.search(r'\[.*\]', text_cleaned, re.DOTALL)
        if array_match:
            json_str = array_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"No valid JSON found in LLM response: {text[:200]}...")

    def health_check(self) -> bool:
        """
        Check if LLM service is healthy.

        Returns:
            True if service is responding
        """
        try:
            # Send minimal request
            response = self.session.post(
                self.config.endpoint,
                json={
                    "model": self.config.model_name,
                    "prompt": "Hello",
                    "max_tokens": 5
                },
                timeout=10,
                headers=self._get_headers()
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.

        Returns:
            Dictionary with call counts and success rate
        """
        success_rate = (
            (self.total_calls - self.failed_calls) / self.total_calls
            if self.total_calls > 0
            else 0.0
        )

        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "success_rate": success_rate,
            "endpoint": self.config.endpoint,
            "model": self.config.model_name
        }

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json"
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def __repr__(self) -> str:
        return (
            f"<LocalLLMClient "
            f"model={self.config.model_name} "
            f"endpoint={self.config.endpoint} "
            f"calls={self.total_calls}>"
        )


class MockLLMClient:
    """
    Mock LLM client for testing without actual LLM service.

    Returns predefined responses for testing agent logic.
    """

    def __init__(self):
        """Initialize mock client."""
        self.total_calls = 0
        self.call_history = []

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate mock response.

        Args:
            prompt: Input prompt
            max_tokens: Ignored
            temperature: Ignored

        Returns:
            Mock response string
        """
        self.total_calls += 1
        self.call_history.append({"prompt": prompt[:100], "max_tokens": max_tokens})

        # Return mock JSON response
        if "hypothesis" in prompt.lower():
            return self._mock_hypothesis_response()
        elif "validate" in prompt.lower():
            return self._mock_validation_response()
        elif "insight" in prompt.lower():
            return self._mock_insights_response()
        else:
            return self._mock_generic_response()

    def generate_json(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate and parse mock JSON response."""
        response = self.generate(prompt, max_tokens, temperature)
        return json.loads(response)

    def health_check(self) -> bool:
        """Always return healthy."""
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get mock stats."""
        return {
            "total_calls": self.total_calls,
            "failed_calls": 0,
            "success_rate": 1.0,
            "endpoint": "mock",
            "model": "mock-llm"
        }

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON from text."""
        return LocalLLMClient.extract_json(text)

    def _mock_hypothesis_response(self) -> str:
        """Mock hypothesis generation response."""
        return json.dumps({
            "statement": "Increasing hull spacing from 4.5m to 5.5m improves stability without significant speed penalty",
            "type": "exploration",
            "test_protocol": {
                "parameters_to_vary": ["hull_spacing"],
                "ranges": [[4.0, 6.0]],
                "num_samples": 8,
                "fixed_parameters": {}
            },
            "expected_outcome": "Stability score increases by 5-10%",
            "success_criteria": "stability_score > 78 AND speed_score > 68",
            "reasoning": "Hull spacing affects lateral stability and wave interference",
            "confidence": 0.75
        })

    def _mock_validation_response(self) -> str:
        """Mock validation response."""
        return json.dumps({
            "validation": "valid",
            "notes": "Designs appear physically realistic and appropriate for testing",
            "concerns": []
        })

    def _mock_insights_response(self) -> str:
        """Mock insights extraction response."""
        return json.dumps({
            "insights": [
                "Longer hulls (>19m) show better speed performance",
                "Hull spacing around 4.5m provides good stability",
                "Deadrise angle has minimal impact in tested range"
            ]
        })

    def _mock_generic_response(self) -> str:
        """Mock generic response."""
        return json.dumps({
            "response": "Mock LLM response",
            "success": True
        })
