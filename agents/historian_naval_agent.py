"""
HistorianNavalAgent: History compression and pattern extraction agent for MAGNET
=================================================================================

The Historian agent compresses experimental history, identifies patterns,
infers constraints from failures, and tracks performance trends over time.

Responsibilities:
- Compress large experiment histories to prevent memory explosion
- Identify parameter correlations and patterns
- Infer constraints from failed experiments
- Track performance trends and improvement rates
- Detect stagnation and trigger exploration
- Update knowledge base with insights
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

from agents.base_naval_agent import BaseNavalAgent, NavalAgentResponse, NavalAgentConfig


class HistorianNavalAgent(BaseNavalAgent):
    """
    Historian agent for history compression and pattern extraction in naval domain.

    Takes experiment results and history as input, outputs compressed history
    and identified patterns.
    """

    # Compression settings
    KEEP_RECENT_CYCLES = 3  # Keep full detail for last N cycles
    KEEP_BEST_DESIGNS = 15  # Always keep top N designs
    MAX_HISTORY_SIZE = 50  # Target compressed history size

    # Pattern detection thresholds
    CORRELATION_THRESHOLD = 0.5  # |r| > 0.5 considered strong correlation
    STAGNATION_CYCLES = 5  # No improvement for N cycles = stagnation

    def __init__(self, config: NavalAgentConfig, llm_client):
        """
        Initialize Historian agent.

        Args:
            config: Agent configuration
            llm_client: LLM client instance
        """
        super().__init__(config, llm_client)
        self.analysis_count = 0

    def autonomous_cycle(self, context: Dict[str, Any]) -> NavalAgentResponse:
        """
        Run one autonomous cycle to compress history and extract patterns.

        Args:
            context: Dictionary containing:
                - new_results: Latest batch of experiment results
                - current_history: Existing compressed history
                - knowledge_base: Current KB state
                - cycle_number: Current research cycle

        Returns:
            NavalAgentResponse with updated history and patterns
        """
        self.state = self.state.__class__.BUSY
        self.current_task = "analyzing_history"

        try:
            # Extract context
            new_results = context.get("new_results", [])
            current_history = context.get("current_history", {})
            knowledge_base = context.get("knowledge_base", {})
            cycle_number = context.get("cycle_number", 0)

            # Process new results
            analysis = self.analyze_batch_results(
                new_results,
                current_history,
                knowledge_base,
                cycle_number
            )

            self.analysis_count += 1
            self.state = self.state.__class__.ACTIVE
            self.current_task = None

            return NavalAgentResponse(
                agent_id=self.agent_id,
                action="update_history",
                reasoning=analysis.get("summary", "History updated"),
                confidence=analysis.get("confidence", 0.85),
                data=analysis
            )

        except Exception as e:
            self.state = self.state.__class__.FAILED
            raise e

    def analyze_batch_results(
        self,
        new_results: List[Dict[str, Any]],
        current_history: Dict[str, Any],
        knowledge_base: Dict[str, Any],
        cycle_number: int
    ) -> Dict[str, Any]:
        """
        Analyze a batch of new results and update history.

        Args:
            new_results: Latest experimental results
            current_history: Existing compressed history
            knowledge_base: Current KB state
            cycle_number: Current cycle number

        Returns:
            Analysis dictionary with compressed history and patterns
        """
        # Get full history (current + new)
        all_experiments = current_history.get("experiments", []) + new_results

        # Compress history
        compressed = self.compress_history(all_experiments, cycle_number)

        # Identify patterns
        patterns = self.identify_patterns(all_experiments)

        # Infer constraints from failures
        constraints = self.infer_constraints(all_experiments)

        # Track performance trends
        trends = self.track_performance_trends(all_experiments, knowledge_base)

        # Generate summary with LLM
        summary = self._generate_summary(
            len(new_results),
            len(compressed["experiments"]),
            patterns,
            constraints,
            trends
        )

        return {
            "compressed_history": compressed,
            "new_patterns": patterns,
            "inferred_constraints": constraints,
            "performance_trends": trends,
            "summary": summary,
            "confidence": 0.90,
            "timestamp": datetime.now().isoformat()
        }

    def compress_history(
        self,
        experiments: List[Dict[str, Any]],
        cycle_number: int
    ) -> Dict[str, Any]:
        """
        Compress experiment history to prevent memory explosion.

        Strategy:
        - Keep all experiments from last KEEP_RECENT_CYCLES cycles (full detail)
        - Keep KEEP_BEST_DESIGNS best designs ever
        - Keep statistical summary of older experiments
        - Discard redundant/low-value experiments

        Args:
            experiments: All experiments to compress
            cycle_number: Current cycle number

        Returns:
            Compressed history dictionary
        """
        if not experiments:
            return {
                "experiments": [],
                "summary_statistics": {},
                "compression_ratio": 1.0
            }

        # Separate valid and invalid experiments
        valid = [e for e in experiments if e.get("results", {}).get("is_valid", False)]
        invalid = [e for e in experiments if not e.get("results", {}).get("is_valid", False)]

        # 1. Keep recent experiments (full detail)
        recent_cutoff_cycle = max(0, cycle_number - self.KEEP_RECENT_CYCLES)
        recent = [e for e in experiments if e.get("cycle_number", 0) >= recent_cutoff_cycle]

        # 2. Keep best designs ever
        if valid:
            sorted_valid = sorted(
                valid,
                key=lambda e: e.get("results", {}).get("overall_score", 0),
                reverse=True
            )
            best_designs = sorted_valid[:self.KEEP_BEST_DESIGNS]
        else:
            best_designs = []

        # 3. Combine (deduplicate)
        kept_ids = set()
        kept_experiments = []

        for exp in recent + best_designs:
            exp_id = exp.get("design_id", str(id(exp)))
            if exp_id not in kept_ids:
                kept_ids.add(exp_id)
                kept_experiments.append(exp)

        # 4. Create statistical summary of discarded experiments
        discarded = [e for e in experiments if e.get("design_id") not in kept_ids]
        summary_stats = self._create_summary_statistics(discarded)

        # 5. Calculate compression ratio
        compression_ratio = len(kept_experiments) / len(experiments) if experiments else 1.0

        return {
            "experiments": kept_experiments,
            "summary_statistics": summary_stats,
            "total_experiments": len(experiments),
            "kept_experiments": len(kept_experiments),
            "compression_ratio": compression_ratio,
            "timestamp": datetime.now().isoformat()
        }

    def identify_patterns(
        self,
        experiments: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Identify parameter patterns and correlations.

        Args:
            experiments: Experiment list

        Returns:
            List of pattern descriptions
        """
        patterns = []

        # Get valid experiments only
        valid = [e for e in experiments if e.get("results", {}).get("is_valid", False)]

        if len(valid) < 5:
            return ["Insufficient data for pattern detection (need ≥5 valid experiments)"]

        # Extract parameter vectors and scores
        param_names = [
            "length_overall", "beam", "hull_spacing", "hull_depth",
            "deadrise_angle", "freeboard", "lcb_position", "prismatic_coefficient"
        ]

        param_vectors = []
        overall_scores = []
        stability_scores = []
        speed_scores = []

        for exp in valid:
            params = exp.get("parameters", {})
            results = exp.get("results", {})

            # Extract parameters
            param_vec = [params.get(p, 0) for p in param_names]
            param_vectors.append(param_vec)

            # Extract scores
            overall_scores.append(results.get("overall_score", 0))
            stability_scores.append(results.get("stability_score", 0))
            speed_scores.append(results.get("speed_score", 0))

        # Convert to numpy arrays
        X = np.array(param_vectors)
        y_overall = np.array(overall_scores)
        y_stability = np.array(stability_scores)
        y_speed = np.array(speed_scores)

        # Calculate correlations for each parameter
        for i, param_name in enumerate(param_names):
            param_values = X[:, i]

            # Skip if no variation
            if np.std(param_values) < 0.01:
                continue

            # Correlation with overall score
            corr_overall = np.corrcoef(param_values, y_overall)[0, 1]

            if abs(corr_overall) > self.CORRELATION_THRESHOLD:
                direction = "increases" if corr_overall > 0 else "decreases"
                patterns.append(
                    f"Parameter '{param_name}' strongly {direction} overall performance "
                    f"(r={corr_overall:.2f})"
                )

            # Correlation with stability
            corr_stability = np.corrcoef(param_values, y_stability)[0, 1]
            if abs(corr_stability) > self.CORRELATION_THRESHOLD:
                direction = "improves" if corr_stability > 0 else "reduces"
                patterns.append(
                    f"Parameter '{param_name}' {direction} stability "
                    f"(r={corr_stability:.2f})"
                )

            # Correlation with speed
            corr_speed = np.corrcoef(param_values, y_speed)[0, 1]
            if abs(corr_speed) > self.CORRELATION_THRESHOLD:
                direction = "improves" if corr_speed > 0 else "reduces"
                patterns.append(
                    f"Parameter '{param_name}' {direction} speed "
                    f"(r={corr_speed:.2f})"
                )

        # Find optimal parameter ranges
        optimal_ranges = self._find_optimal_ranges(valid)
        for param, (min_val, max_val, avg_score) in optimal_ranges.items():
            patterns.append(
                f"Optimal '{param}' range: {min_val:.2f}-{max_val:.2f}m "
                f"(avg score: {avg_score:.1f})"
            )

        return patterns[:10]  # Return top 10 patterns

    def infer_constraints(
        self,
        experiments: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Infer constraints from failed experiments.

        Args:
            experiments: Experiment list

        Returns:
            List of inferred constraint descriptions
        """
        constraints = []

        # Get failed experiments
        failed = [e for e in experiments if not e.get("results", {}).get("is_valid", False)]

        if len(failed) < 3:
            return ["Insufficient failures for constraint inference"]

        # Analyze failure patterns
        failure_regions = defaultdict(list)

        for exp in failed:
            params = exp.get("parameters", {})
            reasons = exp.get("results", {}).get("failure_reasons", ["unknown"])

            # Group by failure reason
            for reason in reasons:
                failure_regions[reason].append(params)

        # Identify parameter ranges that always fail
        for reason, param_sets in failure_regions.items():
            if len(param_sets) < 3:
                continue

            # Find common parameter ranges
            param_ranges = self._find_common_ranges(param_sets)

            for param, (min_val, max_val) in param_ranges.items():
                if max_val - min_val < 0.5:  # Narrow range = likely constraint
                    constraints.append(
                        f"Avoid '{param}' in range {min_val:.2f}-{max_val:.2f}m "
                        f"(causes {reason})"
                    )

        # Check for interaction constraints
        interaction_constraints = self._find_interaction_constraints(failed)
        constraints.extend(interaction_constraints)

        return constraints[:8]  # Return top 8 constraints

    def track_performance_trends(
        self,
        experiments: List[Dict[str, Any]],
        knowledge_base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Track performance trends over time.

        Args:
            experiments: All experiments
            knowledge_base: Current KB

        Returns:
            Trends dictionary with metrics
        """
        # Get valid experiments sorted by cycle
        valid = [e for e in experiments if e.get("results", {}).get("is_valid", False)]

        if len(valid) < 2:
            return {
                "improvement_rate": 0.0,
                "best_so_far": {},
                "stagnation_detected": False,
                "cycles_without_improvement": 0
            }

        # Sort by cycle number
        sorted_valid = sorted(valid, key=lambda e: e.get("cycle_number", 0))

        # Track best score over time
        best_scores_by_cycle = {}
        current_best = 0

        for exp in sorted_valid:
            cycle = exp.get("cycle_number", 0)
            score = exp.get("results", {}).get("overall_score", 0)

            if score > current_best:
                current_best = score

            if cycle not in best_scores_by_cycle or current_best > best_scores_by_cycle[cycle]:
                best_scores_by_cycle[cycle] = current_best

        # Calculate improvement rate (linear regression on best scores)
        if len(best_scores_by_cycle) >= 2:
            cycles = np.array(list(best_scores_by_cycle.keys()))
            scores = np.array(list(best_scores_by_cycle.values()))

            # Linear fit
            if len(cycles) > 1:
                coeffs = np.polyfit(cycles, scores, 1)
                improvement_rate = coeffs[0]  # Slope = improvement per cycle
            else:
                improvement_rate = 0.0
        else:
            improvement_rate = 0.0

        # Detect stagnation
        recent_cycles = list(best_scores_by_cycle.keys())[-self.STAGNATION_CYCLES:]
        recent_scores = [best_scores_by_cycle[c] for c in recent_cycles]

        if len(recent_scores) >= self.STAGNATION_CYCLES:
            score_range = max(recent_scores) - min(recent_scores)
            stagnation_detected = score_range < 1.0  # Less than 1 point improvement
            cycles_without_improvement = self.STAGNATION_CYCLES if stagnation_detected else 0
        else:
            stagnation_detected = False
            cycles_without_improvement = 0

        # Best design ever
        best_design = max(valid, key=lambda e: e.get("results", {}).get("overall_score", 0))

        return {
            "improvement_rate": float(improvement_rate),
            "best_so_far": {
                "design_id": best_design.get("design_id"),
                "score": best_design.get("results", {}).get("overall_score", 0),
                "parameters": best_design.get("parameters", {})
            },
            "stagnation_detected": stagnation_detected,
            "cycles_without_improvement": cycles_without_improvement,
            "total_valid_experiments": len(valid),
            "best_scores_by_cycle": best_scores_by_cycle
        }

    def _create_summary_statistics(
        self,
        experiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create statistical summary of experiments.

        Args:
            experiments: Experiments to summarize

        Returns:
            Summary statistics dictionary
        """
        if not experiments:
            return {}

        valid = [e for e in experiments if e.get("results", {}).get("is_valid", False)]

        if not valid:
            return {"num_experiments": len(experiments), "num_valid": 0}

        scores = [e.get("results", {}).get("overall_score", 0) for e in valid]

        return {
            "num_experiments": len(experiments),
            "num_valid": len(valid),
            "num_invalid": len(experiments) - len(valid),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores))
        }

    def _find_optimal_ranges(
        self,
        experiments: List[Dict[str, Any]]
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Find parameter ranges that produce best results.

        Args:
            experiments: Valid experiments

        Returns:
            Dictionary of param -> (min, max, avg_score)
        """
        # Get top 25% performers
        sorted_exps = sorted(
            experiments,
            key=lambda e: e.get("results", {}).get("overall_score", 0),
            reverse=True
        )
        top_quartile = sorted_exps[:max(1, len(sorted_exps) // 4)]

        optimal_ranges = {}

        param_names = ["length_overall", "beam", "hull_spacing", "hull_depth"]

        for param in param_names:
            values = [e.get("parameters", {}).get(param, 0) for e in top_quartile]
            scores = [e.get("results", {}).get("overall_score", 0) for e in top_quartile]

            if values:
                optimal_ranges[param] = (
                    float(np.min(values)),
                    float(np.max(values)),
                    float(np.mean(scores))
                )

        return optimal_ranges

    def _find_common_ranges(
        self,
        param_sets: List[Dict[str, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Find common parameter ranges across failed experiments.

        Args:
            param_sets: List of parameter dictionaries

        Returns:
            Dictionary of param -> (min, max) for common ranges
        """
        common_ranges = {}

        param_names = ["length_overall", "beam", "hull_spacing", "hull_depth",
                       "deadrise_angle", "freeboard"]

        for param in param_names:
            values = [ps.get(param) for ps in param_sets if param in ps]

            if len(values) >= 3:
                common_ranges[param] = (float(np.min(values)), float(np.max(values)))

        return common_ranges

    def _find_interaction_constraints(
        self,
        failed_experiments: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Find interaction constraints (e.g., "if X > a AND Y < b, then fails").

        Args:
            failed_experiments: Failed experiments

        Returns:
            List of interaction constraint descriptions
        """
        constraints = []

        # Simple heuristic: look for common parameter combinations
        # In a real implementation, would use decision tree or rule mining

        if len(failed_experiments) < 5:
            return []

        # Check for common spacing/beam violations
        spacing_beam_violations = []
        for exp in failed_experiments:
            params = exp.get("parameters", {})
            spacing = params.get("hull_spacing", 0)
            beam = params.get("beam", 1)

            if spacing / beam > 0.9:  # Close to constraint boundary
                spacing_beam_violations.append((spacing, beam))

        if len(spacing_beam_violations) >= 3:
            constraints.append(
                "High hull_spacing/beam ratio (>0.9) frequently causes failures"
            )

        return constraints

    def _generate_summary(
        self,
        num_new: int,
        num_kept: int,
        patterns: List[str],
        constraints: List[str],
        trends: Dict[str, Any]
    ) -> str:
        """
        Generate natural language summary of analysis.

        Args:
            num_new: Number of new experiments
            num_kept: Number of experiments kept in compressed history
            patterns: Identified patterns
            constraints: Inferred constraints
            trends: Performance trends

        Returns:
            Summary string
        """
        parts = [f"Processed {num_new} new experiments."]
        parts.append(f"Compressed history to {num_kept} key experiments.")

        if patterns:
            parts.append(f"Identified {len(patterns)} new patterns.")

        if constraints:
            parts.append(f"Inferred {len(constraints)} constraints from failures.")

        improvement = trends.get("improvement_rate", 0)
        if improvement > 0:
            parts.append(f"Performance improving at {improvement:.2f} points/cycle.")
        elif trends.get("stagnation_detected"):
            parts.append("Stagnation detected - may need paradigm shift.")

        return " ".join(parts)
