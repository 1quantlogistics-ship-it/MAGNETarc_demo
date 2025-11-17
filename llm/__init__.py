"""
ARC LLM Integration Layer
==========================

Multi-model LLM client with role-based routing and offline support.

Key Components:
- LLMClient: Base LLM client (OpenAI-compatible API)
- MockLLMClient: Offline mock client for development/testing
- LLMRouter: Routes agent roles to specific models
- Model registry: Available model definitions
"""

from llm.client import LLMClient
from llm.mock_client import MockLLMClient
from llm.router import LLMRouter
from llm.models import ModelConfig, get_model_config

__all__ = ["LLMClient", "MockLLMClient", "LLMRouter", "ModelConfig", "get_model_config"]
