"""Supervisor Naval Agent - Strategy Adjustment"""
from agents.base_naval_agent import BaseNavalAgent, NavalAgentConfig, NavalAgentResponse
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

class SupervisorNavalAgent(BaseNavalAgent):
    """Meta-learning agent that adjusts exploration strategy"""
    
    STAGNATION_THRESHOLD = 5
    HIGH_IMPROVEMENT_THRESHOLD = 5.0
    LOW_IMPROVEMENT_THRESHOLD = 0.5
    EXPLORATION_TEMP_MIN = 0.5
    EXPLORATION_TEMP_MAX = 2.0
    EXPLORATION_TEMP_DEFAULT = 1.0

    def __init__(self, config: NavalAgentConfig, llm_client):
        super().__init__(config, llm_client)
        self.current_temperature = self.EXPLORATION_TEMP_DEFAULT
        self.strategy_history: List[Dict[str, Any]] = []

    def autonomous_cycle(self, context: Dict[str, Any]) -> NavalAgentResponse:
        """Analyze research progress and adjust strategy"""
        experiment_history = context.get("experiment_history", [])
        cycle_number = context.get("cycle_number", 0)
        current_best_score = context.get("current_best_score", 0.0)

        trajectory = self._analyze_trajectory(experiment_history, cycle_number)
        stagnation = self._detect_stagnation(experiment_history, current_best_score)
        improvement_rate = trajectory.get("improvement_rate", 0.0)

        strategy = self._determine_strategy(
            improvement_rate=improvement_rate,
            stagnation_detected=stagnation["detected"],
            cycles_without_improvement=stagnation["cycles_without_improvement"],
            cycle_number=cycle_number
        )

        reasoning = f"Trend: {trajectory.get('trend', 'unknown')} | Strategy: {strategy['mode']}"

        strategy_record = {
            "cycle": cycle_number,
            "timestamp": datetime.now().isoformat(),
            "improvement_rate": improvement_rate,
            "stagnation_detected": stagnation["detected"],
            "cycles_without_improvement": stagnation["cycles_without_improvement"],
            "strategy": strategy["mode"],
            "exploration_temperature": strategy["exploration_temperature"],
        }
        self.strategy_history.append(strategy_record)

        response_data = {
            "strategy_adjustment": strategy,
            "trajectory_analysis": trajectory,
            "stagnation_analysis": stagnation,
            "reasoning_summary": reasoning,
            "strategy_history": self.strategy_history[-10:],
        }

        return NavalAgentResponse(
            agent_id=self.config.agent_id,
            action="adjust_strategy",
            reasoning=reasoning,
            confidence=strategy["confidence"],
            data=response_data
        )

    def _analyze_trajectory(self, experiment_history: List[Dict[str, Any]], cycle_number: int) -> Dict[str, Any]:
        """Analyze research trajectory"""
        if not experiment_history:
            return {"improvement_rate": 0.0, "total_experiments": 0, "best_score_history": [], "trend": "no_data"}

        valid_experiments = [exp for exp in experiment_history if exp.get("results", {}).get("is_valid", False)]
        if not valid_experiments:
            return {"improvement_rate": 0.0, "total_experiments": len(experiment_history), "best_score_history": [], "trend": "no_valid_results"}

        cycles_seen = {}
        for exp in valid_experiments:
            cycle = exp.get("cycle_number", 0)
            score = exp.get("results", {}).get("overall_score", 0.0)
            if cycle not in cycles_seen:
                cycles_seen[cycle] = score
            else:
                cycles_seen[cycle] = max(cycles_seen[cycle], score)

        best_score_history = [{"cycle": c, "best_score": s} for c, s in sorted(cycles_seen.items())]

        if len(best_score_history) >= 2:
            cycles = np.array([entry["cycle"] for entry in best_score_history])
            scores = np.array([entry["best_score"] for entry in best_score_history])
            slope, intercept = np.polyfit(cycles, scores, 1)
            improvement_rate = float(slope)

            if improvement_rate > self.HIGH_IMPROVEMENT_THRESHOLD:
                trend = "rapid_improvement"
            elif improvement_rate > self.LOW_IMPROVEMENT_THRESHOLD:
                trend = "steady_improvement"
            elif improvement_rate > -self.LOW_IMPROVEMENT_THRESHOLD:
                trend = "stagnant"
            else:
                trend = "degrading"
        else:
            improvement_rate = 0.0
            trend = "insufficient_data"

        return {
            "improvement_rate": improvement_rate,
            "total_experiments": len(experiment_history),
            "total_valid": len(valid_experiments),
            "best_score_history": best_score_history,
            "trend": trend,
            "current_best": best_score_history[-1]["best_score"] if best_score_history else 0.0
        }

    def _detect_stagnation(self, experiment_history: List[Dict[str, Any]], current_best_score: float) -> Dict[str, Any]:
        """Detect stagnation"""
        if not experiment_history:
            return {"detected": False, "cycles_without_improvement": 0, "last_improvement_cycle": 0, "recommendation": "continue"}

        cycle_best = {}
        for exp in experiment_history:
            if not exp.get("results", {}).get("is_valid", False):
                continue
            cycle = exp.get("cycle_number", 0)
            score = exp.get("results", {}).get("overall_score", 0.0)
            if cycle not in cycle_best:
                cycle_best[cycle] = score
            else:
                cycle_best[cycle] = max(cycle_best[cycle], score)

        if not cycle_best:
            return {"detected": False, "cycles_without_improvement": 0, "last_improvement_cycle": 0, "recommendation": "continue"}

        sorted_cycles = sorted(cycle_best.keys())
        best_score_overall = max(cycle_best.values())

        last_improvement_cycle = 0
        for cycle in sorted_cycles:
            if cycle_best[cycle] >= best_score_overall - 0.1:
                if last_improvement_cycle == 0:
                    last_improvement_cycle = cycle
                break

        current_cycle = sorted_cycles[-1] if sorted_cycles else 0
        cycles_without_improvement = current_cycle - last_improvement_cycle

        stagnation_detected = cycles_without_improvement >= self.STAGNATION_THRESHOLD

        recommendation = "increase_exploration" if stagnation_detected else "continue"

        return {
            "detected": stagnation_detected,
            "cycles_without_improvement": cycles_without_improvement,
            "last_improvement_cycle": last_improvement_cycle,
            "best_score_overall": best_score_overall,
            "recommendation": recommendation
        }

    def _determine_strategy(self, improvement_rate: float, stagnation_detected: bool, cycles_without_improvement: int, cycle_number: int) -> Dict[str, Any]:
        """Determine strategy"""
        strategy = {
            "mode": "balanced",
            "exploration_temperature": self.EXPLORATION_TEMP_DEFAULT,
            "sampling_weights": {"latin_hypercube": 0.4, "gaussian": 0.4, "edge_corner": 0.2},
            "confidence": 0.7
        }

        if cycle_number <= 3:
            strategy["mode"] = "exploration"
            strategy["exploration_temperature"] = 1.5
            strategy["sampling_weights"] = {"latin_hypercube": 0.6, "gaussian": 0.2, "edge_corner": 0.2}
            strategy["confidence"] = 0.8
            self.current_temperature = 1.5

        elif stagnation_detected:
            strategy["mode"] = "exploration_boost"
            strategy["exploration_temperature"] = min(self.current_temperature * 1.5, self.EXPLORATION_TEMP_MAX)
            strategy["sampling_weights"] = {"latin_hypercube": 0.5, "gaussian": 0.2, "edge_corner": 0.3}
            strategy["confidence"] = 0.9
            self.current_temperature = strategy["exploration_temperature"]

        elif improvement_rate > self.HIGH_IMPROVEMENT_THRESHOLD:
            strategy["mode"] = "exploitation"
            strategy["exploration_temperature"] = max(self.current_temperature * 0.8, self.EXPLORATION_TEMP_MIN)
            strategy["sampling_weights"] = {"latin_hypercube": 0.2, "gaussian": 0.7, "edge_corner": 0.1}
            strategy["confidence"] = 0.85
            self.current_temperature = strategy["exploration_temperature"]

        elif improvement_rate > self.LOW_IMPROVEMENT_THRESHOLD:
            strategy["mode"] = "balanced"
            strategy["exploration_temperature"] = self.EXPLORATION_TEMP_DEFAULT
            strategy["sampling_weights"] = {"latin_hypercube": 0.4, "gaussian": 0.4, "edge_corner": 0.2}
            strategy["confidence"] = 0.8
            self.current_temperature = self.EXPLORATION_TEMP_DEFAULT

        else:
            strategy["mode"] = "mild_exploration"
            strategy["exploration_temperature"] = 1.2
            strategy["sampling_weights"] = {"latin_hypercube": 0.5, "gaussian": 0.3, "edge_corner": 0.2}
            strategy["confidence"] = 0.6
            self.current_temperature = 1.2

        return strategy

    def get_current_strategy(self) -> Dict[str, Any]:
        """Get current strategy"""
        if self.strategy_history:
            return {"exploration_temperature": self.current_temperature, "last_decision": self.strategy_history[-1]}
        else:
            return {"exploration_temperature": self.EXPLORATION_TEMP_DEFAULT, "last_decision": None}
