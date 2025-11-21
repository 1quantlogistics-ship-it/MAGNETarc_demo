"""
MAGNET Naval Agent Infrastructure
==================================

This package contains the naval-specialized agents for MAGNET autonomous design.

Naval Agents:
- BaseNavalAgent: Abstract base class for all naval agents
- ExplorerAgent: Design space exploration
- ExperimentalArchitectAgent: Novel hull generation
- CriticNavalAgent: Safety validation
- HistorianNavalAgent: Knowledge base curation
- SupervisorNavalAgent: Meta-learning and coordination
"""

from agents.base_naval_agent import BaseNavalAgent, NavalAgentConfig, NavalAgentResponse

__version__ = "0.1.0"
__all__ = ["BaseNavalAgent", "NavalAgentConfig", "NavalAgentResponse"]
