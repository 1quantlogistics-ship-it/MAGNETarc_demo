"""
Memory module for autonomous research knowledge management.

Exports:
- KnowledgeBase: Persistent storage for experiments, principles, and best designs
- MetricsTracker: Performance monitoring and metrics collection
"""

from .knowledge_base import KnowledgeBase
from .metrics_tracker import MetricsTracker

__all__ = ['KnowledgeBase', 'MetricsTracker']
