"""
AgentRegistry: Registration and discovery system for ARC agents
================================================================

Manages all active agents, provides discovery by role/capability,
and monitors agent health.
"""

from typing import Dict, List, Optional
from agents.base import BaseAgent, AgentState, AgentCapability


class AgentRegistry:
    """
    Central registry for all ARC agents.

    Provides:
    - Agent registration/deregistration
    - Discovery by role, capability, or ID
    - Health monitoring
    - Priority-based agent selection
    """

    def __init__(self):
        """Initialize empty registry."""
        self._agents: Dict[str, BaseAgent] = {}
        self._by_role: Dict[str, List[str]] = {}
        self._by_capability: Dict[AgentCapability, List[str]] = {}

    def register(self, agent: BaseAgent) -> bool:
        """
        Register an agent.

        Args:
            agent: Agent instance to register

        Returns:
            True if registration successful
        """
        if agent.agent_id in self._agents:
            print(f"Warning: Agent {agent.agent_id} already registered")
            return False

        # Store agent
        self._agents[agent.agent_id] = agent

        # Index by role
        if agent.role not in self._by_role:
            self._by_role[agent.role] = []
        self._by_role[agent.role].append(agent.agent_id)

        # Index by capabilities
        for capability in agent.capabilities:
            if capability not in self._by_capability:
                self._by_capability[capability] = []
            self._by_capability[capability].append(agent.agent_id)

        return True

    def deregister(self, agent_id: str) -> bool:
        """
        Deregister an agent.

        Args:
            agent_id: ID of agent to remove

        Returns:
            True if deregistration successful
        """
        if agent_id not in self._agents:
            return False

        agent = self._agents[agent_id]

        # Remove from role index
        if agent.role in self._by_role:
            self._by_role[agent.role].remove(agent_id)
            if not self._by_role[agent.role]:
                del self._by_role[agent.role]

        # Remove from capability index
        for capability in agent.capabilities:
            if capability in self._by_capability:
                self._by_capability[capability].remove(agent_id)
                if not self._by_capability[capability]:
                    del self._by_capability[capability]

        # Remove agent
        del self._agents[agent_id]
        return True

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent instance or None
        """
        return self._agents.get(agent_id)

    def get_agents_by_role(self, role: str) -> List[BaseAgent]:
        """
        Get all agents with a specific role.

        Args:
            role: Role to filter by (director, architect, critic, etc.)

        Returns:
            List of agents with that role
        """
        agent_ids = self._by_role.get(role, [])
        return [self._agents[aid] for aid in agent_ids]

    def get_agents_by_capability(self, capability: AgentCapability) -> List[BaseAgent]:
        """
        Get all agents with a specific capability.

        Args:
            capability: Capability to filter by

        Returns:
            List of agents with that capability
        """
        agent_ids = self._by_capability.get(capability, [])
        return [self._agents[aid] for aid in agent_ids]

    def get_all_agents(self) -> List[BaseAgent]:
        """
        Get all registered agents.

        Returns:
            List of all agents
        """
        return list(self._agents.values())

    def get_active_agents(self) -> List[BaseAgent]:
        """
        Get all agents in ACTIVE or BUSY state.

        Returns:
            List of active agents
        """
        return [
            agent for agent in self._agents.values()
            if agent.state in [AgentState.ACTIVE, AgentState.BUSY]
        ]

    def get_health_report(self) -> Dict:
        """
        Get health report for all agents.

        Returns:
            Comprehensive health report
        """
        agents_status = [agent.health_check() for agent in self._agents.values()]

        healthy = sum(1 for s in agents_status if s["healthy"])
        total = len(agents_status)

        return {
            "total_agents": total,
            "healthy_agents": healthy,
            "unhealthy_agents": total - healthy,
            "agents": agents_status,
            "by_role": {
                role: len(agent_ids)
                for role, agent_ids in self._by_role.items()
            },
            "by_state": self._get_state_distribution()
        }

    def _get_state_distribution(self) -> Dict[str, int]:
        """Get distribution of agents by state."""
        distribution = {}
        for agent in self._agents.values():
            state_name = agent.state.value
            distribution[state_name] = distribution.get(state_name, 0) + 1
        return distribution

    def select_best_agent(
        self,
        role: Optional[str] = None,
        capability: Optional[AgentCapability] = None,
        prefer_online: bool = True
    ) -> Optional[BaseAgent]:
        """
        Select the best agent for a task based on criteria.

        Args:
            role: Required role (optional)
            capability: Required capability (optional)
            prefer_online: Prefer online agents over offline

        Returns:
            Best agent or None
        """
        candidates = self.get_all_agents()

        # Filter by role
        if role:
            candidates = [a for a in candidates if a.role == role]

        # Filter by capability
        if capability:
            candidates = [a for a in candidates if capability in a.capabilities]

        # Filter by state (must be active)
        candidates = [a for a in candidates if a.state == AgentState.ACTIVE]

        if not candidates:
            return None

        # Prefer online agents if requested
        if prefer_online:
            online_candidates = [a for a in candidates if not a.offline]
            if online_candidates:
                candidates = online_candidates

        # Sort by priority (critical > high > medium > low) and voting weight
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        candidates.sort(
            key=lambda a: (priority_order.get(a.priority, 0), a.voting_weight),
            reverse=True
        )

        return candidates[0]

    def __len__(self) -> int:
        """Return number of registered agents."""
        return len(self._agents)

    def __repr__(self) -> str:
        return f"<AgentRegistry agents={len(self._agents)}>"
