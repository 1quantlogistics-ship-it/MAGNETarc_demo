"""
ARC Phase D: Multi-Agent Infrastructure
========================================

This package contains the multi-agent architecture for ARC v1.1.0+.
Supports heterogeneous models, democratic voting, and supervisor validation.

Key Components:
- BaseAgent: Abstract base class for all agents
- AgentRegistry: Registration and discovery system
- Communication protocol for inter-agent messaging
"""

from agents.base import BaseAgent, AgentState, AgentCapability
from agents.registry import AgentRegistry

__version__ = "1.1.0-alpha"
__all__ = ["BaseAgent", "AgentState", "AgentCapability", "AgentRegistry"]
