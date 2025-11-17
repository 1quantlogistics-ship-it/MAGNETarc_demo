"""
Mock Data Generator: Generate synthetic data for dashboard development
=======================================================================

Provides mock agent activity, voting records, and supervisor decisions
for testing dashboard without live agents.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json


class MockDataGenerator:
    """
    Generate synthetic data for ARC dashboard.

    Useful for:
    - Dashboard development without live agents
    - Testing visualization components
    - Demo mode
    """

    def __init__(self, seed: int = 42):
        """
        Initialize mock data generator.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.agent_ids = [
            "director_001",
            "architect_001",
            "critic_001",
            "critic_secondary_001",
            "historian_001",
            "executor_001",
            "explorer_001",
            "parameter_scientist_001",
            "supervisor_001"
        ]

        self.agent_roles = {
            "director_001": "director",
            "architect_001": "architect",
            "critic_001": "critic",
            "critic_secondary_001": "critic_secondary",
            "historian_001": "historian",
            "executor_001": "executor",
            "explorer_001": "explorer",
            "parameter_scientist_001": "parameter_scientist",
            "supervisor_001": "supervisor"
        }

    def generate_agent_status(self) -> List[Dict[str, Any]]:
        """
        Generate mock agent status data.

        Returns:
            List of agent status dictionaries
        """
        statuses = []

        states = ["active", "busy", "inactive"]
        models = {
            "director_001": "claude-sonnet-4.5",
            "architect_001": "deepseek-r1",
            "critic_001": "qwen2.5-32b",
            "critic_secondary_001": "deepseek-r1",
            "historian_001": "deepseek-r1",
            "executor_001": "deepseek-r1",
            "explorer_001": "qwen2.5-32b",
            "parameter_scientist_001": "deepseek-r1",
            "supervisor_001": "llama-3-8b-local"
        }

        for agent_id in self.agent_ids:
            state = random.choice(states)

            statuses.append({
                "agent_id": agent_id,
                "role": self.agent_roles[agent_id],
                "model": models[agent_id],
                "state": state,
                "last_activity": (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat(),
                "current_task": f"Processing cycle {random.randint(10, 25)}" if state == "busy" else None,
                "metrics": {
                    "total_tasks": random.randint(50, 200),
                    "successful_tasks": random.randint(45, 195),
                    "failed_tasks": random.randint(0, 10),
                    "avg_response_time_ms": random.uniform(500, 3000),
                    "total_votes": random.randint(30, 150),
                    "vote_agreement_rate": random.uniform(0.65, 0.95)
                },
                "healthy": state in ["active", "busy"],
                "voting_weight": self._get_voting_weight(self.agent_roles[agent_id])
            })

        return statuses

    def generate_voting_records(self, num_records: int = 20) -> List[Dict[str, Any]]:
        """
        Generate mock voting records.

        Args:
            num_records: Number of voting records to generate

        Returns:
            List of voting record dictionaries
        """
        records = []
        decisions = ["approve", "reject", "revise"]

        for i in range(num_records):
            proposal_id = f"exp_mock_{i:03d}"

            # Generate votes from multiple agents
            votes = []
            for agent_id in random.sample(self.agent_ids, k=random.randint(5, 9)):
                decision = random.choice(decisions)
                votes.append({
                    "agent_id": agent_id,
                    "role": self.agent_roles[agent_id],
                    "decision": decision,
                    "confidence": random.uniform(0.6, 0.99),
                    "voting_weight": self._get_voting_weight(self.agent_roles[agent_id]),
                    "reasoning": f"Mock reasoning from {self.agent_roles[agent_id]}"
                })

            # Calculate consensus
            weighted_score = sum(
                (1 if v["decision"] == "approve" else -1 if v["decision"] == "reject" else 0) * v["voting_weight"]
                for v in votes
            ) / sum(v["voting_weight"] for v in votes)

            consensus_reached = abs(weighted_score) >= 0.66
            final_decision = "approve" if weighted_score > 0.66 else "reject" if weighted_score < -0.66 else "revise"

            records.append({
                "proposal_id": proposal_id,
                "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
                "votes": votes,
                "weighted_score": weighted_score,
                "consensus_reached": consensus_reached,
                "final_decision": final_decision,
                "total_votes": len(votes)
            })

        return sorted(records, key=lambda x: x["timestamp"], reverse=True)

    def generate_supervisor_decisions(self, num_decisions: int = 15) -> List[Dict[str, Any]]:
        """
        Generate mock supervisor decisions.

        Args:
            num_decisions: Number of decisions to generate

        Returns:
            List of supervisor decision dictionaries
        """
        decisions = []
        decision_types = ["approve", "reject", "revise", "override"]
        risk_levels = ["low", "medium", "high", "critical"]

        for i in range(num_decisions):
            decision_type = random.choice(decision_types)
            risk_level = random.choice(risk_levels)

            # Override more likely with high/critical risk
            override_consensus = decision_type == "override" or (risk_level in ["high", "critical"] and random.random() > 0.7)

            decisions.append({
                "proposal_id": f"exp_mock_{i:03d}",
                "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
                "decision": decision_type,
                "risk_assessment": risk_level,
                "reasoning": f"Mock supervisor reasoning for {decision_type} decision",
                "constraints_violated": [] if decision_type == "approve" else [f"constraint_{random.randint(1, 3)}"],
                "override_consensus": override_consensus,
                "confidence": random.uniform(0.85, 0.99)
            })

        return sorted(decisions, key=lambda x: x["timestamp"], reverse=True)

    def generate_consensus_metrics(self) -> Dict[str, Any]:
        """
        Generate mock consensus quality metrics.

        Returns:
            Consensus metrics dictionary
        """
        return {
            "total_votes_conducted": random.randint(50, 150),
            "consensus_rate": random.uniform(0.70, 0.90),
            "avg_confidence": random.uniform(0.75, 0.88),
            "controversial_rate": random.uniform(0.10, 0.25),
            "decision_breakdown": {
                "approve": random.randint(30, 70),
                "reject": random.randint(10, 30),
                "revise": random.randint(5, 15)
            },
            "avg_voting_time_seconds": random.uniform(5, 20),
            "supervisor_override_rate": random.uniform(0.05, 0.15)
        }

    def generate_agent_performance_comparison(self) -> List[Dict[str, Any]]:
        """
        Generate mock agent performance comparison data.

        Returns:
            List of agent performance data
        """
        performance = []

        for agent_id in self.agent_ids:
            performance.append({
                "agent_id": agent_id,
                "role": self.agent_roles[agent_id],
                "success_rate": random.uniform(0.85, 0.98),
                "avg_response_time_ms": random.uniform(500, 3000),
                "total_tasks": random.randint(50, 200),
                "vote_agreement_with_consensus": random.uniform(0.70, 0.95)
            })

        return performance

    def generate_risk_distribution(self) -> Dict[str, int]:
        """
        Generate mock risk level distribution.

        Returns:
            Risk distribution dictionary
        """
        total = 100
        critical = random.randint(2, 8)
        high = random.randint(10, 20)
        medium = random.randint(25, 40)
        low = total - critical - high - medium

        return {
            "low": low,
            "medium": medium,
            "high": high,
            "critical": critical
        }

    def generate_agent_voting_patterns(self) -> Dict[str, Dict[str, float]]:
        """
        Generate mock agent voting pattern matrix (agreement rates).

        Returns:
            Matrix of agent-agent agreement rates
        """
        patterns = {}

        for agent1 in self.agent_ids:
            patterns[agent1] = {}
            for agent2 in self.agent_ids:
                if agent1 == agent2:
                    patterns[agent1][agent2] = 1.0
                else:
                    # Same role type tend to agree more
                    role1 = self.agent_roles[agent1]
                    role2 = self.agent_roles[agent2]

                    if "critic" in role1 and "critic" in role2:
                        agreement = random.uniform(0.80, 0.95)
                    elif role1 == role2:
                        agreement = random.uniform(0.75, 0.90)
                    else:
                        agreement = random.uniform(0.60, 0.80)

                    patterns[agent1][agent2] = agreement

        return patterns

    def generate_proposal_quality_trends(self, num_cycles: int = 20) -> List[Dict[str, Any]]:
        """
        Generate mock proposal quality trends over cycles.

        Args:
            num_cycles: Number of cycles to generate

        Returns:
            List of cycle quality metrics
        """
        trends = []

        for cycle in range(1, num_cycles + 1):
            # Simulate improvement over time
            base_quality = 0.60 + (cycle / num_cycles) * 0.25
            noise = random.uniform(-0.05, 0.05)

            trends.append({
                "cycle_id": cycle,
                "timestamp": (datetime.now() - timedelta(days=num_cycles - cycle)).isoformat(),
                "avg_proposal_quality": max(0.5, min(0.95, base_quality + noise)),
                "consensus_score": random.uniform(0.65, 0.85),
                "proposals_approved": random.randint(3, 8),
                "proposals_rejected": random.randint(0, 3)
            })

        return trends

    def _get_voting_weight(self, role: str) -> float:
        """Get voting weight for a role."""
        weights = {
            "supervisor": 3.0,
            "director": 2.0,
            "critic": 2.0,
            "critic_secondary": 1.8,
            "architect": 1.5,
            "parameter_scientist": 1.5,
            "explorer": 1.2,
            "historian": 1.0,
            "executor": 1.0
        }
        return weights.get(role, 1.0)

    def save_mock_data(self, output_path: str = "/workspace/arc/memory/mock_dashboard_data.json") -> bool:
        """
        Generate and save all mock data to file.

        Args:
            output_path: Path to save JSON file

        Returns:
            True if save successful
        """
        try:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            mock_data = {
                "generated_at": datetime.now().isoformat(),
                "agent_status": self.generate_agent_status(),
                "voting_records": self.generate_voting_records(20),
                "supervisor_decisions": self.generate_supervisor_decisions(15),
                "consensus_metrics": self.generate_consensus_metrics(),
                "agent_performance": self.generate_agent_performance_comparison(),
                "risk_distribution": self.generate_risk_distribution(),
                "voting_patterns": self.generate_agent_voting_patterns(),
                "proposal_quality_trends": self.generate_proposal_quality_trends(20)
            }

            with open(output_path, 'w') as f:
                json.dump(mock_data, f, indent=2)

            return True
        except Exception as e:
            print(f"Error saving mock data: {e}")
            return False


# Convenience function for dashboard use
def get_mock_data() -> Dict[str, Any]:
    """
    Get generated mock data for dashboard.

    Returns:
        Dictionary with all mock data
    """
    generator = MockDataGenerator()

    return {
        "generated_at": datetime.now().isoformat(),
        "agent_status": generator.generate_agent_status(),
        "voting_records": generator.generate_voting_records(20),
        "supervisor_decisions": generator.generate_supervisor_decisions(15),
        "consensus_metrics": generator.generate_consensus_metrics(),
        "agent_performance": generator.generate_agent_performance_comparison(),
        "risk_distribution": generator.generate_risk_distribution(),
        "voting_patterns": generator.generate_agent_voting_patterns(),
        "proposal_quality_trends": generator.generate_proposal_quality_trends(20)
    }
