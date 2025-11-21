"""
CriticNavalAgent: Experimental review and critique agent for MAGNET
====================================================================

The Critic agent reviews experimental designs before simulation and critiques
results after simulation. It provides safety checks, redundancy detection,
and recommendations for improvement.

Responsibilities:
- Pre-simulation: Review designs for safety and redundancy
- Post-simulation: Critique results and extract insights
- Validate hypothesis coherence
- Flag dangerous or implausible designs
- Detect duplicate experiments
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
from datetime import datetime

from agents.base_naval_agent import BaseNavalAgent, NavalAgentResponse, NavalAgentConfig


class CriticNavalAgent(BaseNavalAgent):
    """
    Critic agent for experimental review and result analysis in naval domain.

    Takes designs/results as input, outputs critiques and recommendations.
    """

    # Safety thresholds
    SAFETY_THRESHOLDS = {
        "min_gm": 0.5,  # Minimum metacentric height (m) for stability
        "min_freeboard": 0.8,  # Minimum freeboard (m)
        "max_beam_to_length": 0.5,  # Maximum B/L ratio
        "min_beam_to_length": 0.15,  # Minimum B/L ratio
        "max_displacement_to_length": 150.0,  # Max disp/L^3 ratio
    }

    # Redundancy detection threshold (Euclidean distance in normalized parameter space)
    REDUNDANCY_THRESHOLD = 0.10  # 10% similarity threshold

    def __init__(self, config: NavalAgentConfig, llm_client):
        """
        Initialize Critic agent.

        Args:
            config: Agent configuration
            llm_client: LLM client instance
        """
        super().__init__(config, llm_client)
        self.critique_count = 0

    def autonomous_cycle(self, context: Dict[str, Any]) -> NavalAgentResponse:
        """
        Run one autonomous cycle to critique experiments or results.

        Args:
            context: Dictionary containing either:
                - For pre-simulation review:
                    - designs: List of designs to review
                    - hypothesis: Hypothesis being tested
                    - experiment_history: Past experiments
                - For post-simulation critique:
                    - experiment_results: Results from physics simulation
                    - hypothesis: Hypothesis being tested
                    - experiment_history: Past experiments

        Returns:
            NavalAgentResponse with critique
        """
        self.state = self.state.__class__.BUSY
        self.current_task = "critiquing"

        try:
            # Determine if this is pre-simulation or post-simulation
            if "designs" in context:
                # Pre-simulation review
                critique = self.review_experiments(
                    context.get("designs", []),
                    context.get("hypothesis", {}),
                    context.get("experiment_history", [])
                )
                action = "submit_pre_review"
            elif "experiment_results" in context:
                # Post-simulation critique
                critique = self.analyze_results(
                    context.get("experiment_results", []),
                    context.get("hypothesis", {}),
                    context.get("experiment_history", [])
                )
                action = "submit_critique"
            else:
                raise ValueError("Context must contain either 'designs' or 'experiment_results'")

            self.critique_count += 1
            self.state = self.state.__class__.ACTIVE
            self.current_task = None

            return NavalAgentResponse(
                agent_id=self.agent_id,
                action=action,
                reasoning=critique.get("summary", "Critique completed"),
                confidence=critique.get("confidence", 0.8),
                data=critique
            )

        except Exception as e:
            self.state = self.state.__class__.FAILED
            raise e

    def review_experiments(
        self,
        designs: List[Dict[str, Any]],
        hypothesis: Dict[str, Any],
        experiment_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Review experimental designs before simulation (pre-simulation check).

        Args:
            designs: List of design dictionaries to review
            hypothesis: Hypothesis being tested
            experiment_history: Past experiments for redundancy checking

        Returns:
            Review dictionary with verdict and concerns
        """
        concerns = []
        safety_flags = []
        recommendations = []
        redundancies = []

        # Check each design
        for design in designs:
            design_id = design.get("design_id", "unknown")
            params = design.get("parameters", {})

            # Safety checks
            safety_issues = self._check_physical_constraints(params)
            if safety_issues:
                safety_flags.append({
                    "design_id": design_id,
                    "issues": safety_issues
                })

            # Hypothesis coherence
            coherence_issues = self._validate_against_hypothesis(design, hypothesis)
            if coherence_issues:
                concerns.append({
                    "design_id": design_id,
                    "type": "coherence",
                    "issues": coherence_issues
                })

        # Redundancy detection across all designs
        redundancies = self._detect_redundancy(designs, experiment_history)

        # Generate overall verdict
        verdict = self._determine_verdict(safety_flags, concerns, redundancies)

        # Generate recommendations
        if safety_flags:
            recommendations.append("Revise designs with safety violations before simulation")
        if redundancies:
            recommendations.append(f"Remove {len(redundancies)} redundant designs to save computation")
        if len(concerns) > len(designs) * 0.5:
            recommendations.append("Many designs lack hypothesis coherence - review test protocol")

        # LLM-based summary (optional)
        summary = self._generate_critique_summary(
            verdict, len(designs), safety_flags, concerns, redundancies
        )

        return {
            "verdict": verdict,
            "summary": summary,
            "confidence": 0.85,
            "designs_reviewed": len(designs),
            "safety_flags": safety_flags,
            "concerns": concerns,
            "redundancies": redundancies,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_results(
        self,
        experiment_results: List[Dict[str, Any]],
        hypothesis: Dict[str, Any],
        experiment_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze experimental results after simulation (post-simulation critique).

        Args:
            experiment_results: Results from physics simulation
            hypothesis: Hypothesis being tested
            experiment_history: Past experiments for comparison

        Returns:
            Critique dictionary with analysis and insights
        """
        insights = []
        concerns = []
        recommendations = []

        # Extract success metrics
        successful = [r for r in experiment_results if r.get("results", {}).get("is_valid", False)]
        failed = [r for r in experiment_results if not r.get("results", {}).get("is_valid", False)]

        # Analyze performance
        if successful:
            scores = [r["results"]["overall_score"] for r in successful]
            best_design = max(successful, key=lambda r: r["results"]["overall_score"])
            worst_design = min(successful, key=lambda r: r["results"]["overall_score"])

            insights.append(f"Success rate: {len(successful)}/{len(experiment_results)} ({len(successful)/len(experiment_results)*100:.1f}%)")
            insights.append(f"Score range: {min(scores):.1f} - {max(scores):.1f}")
            insights.append(f"Best design: {best_design['design_id']} (score: {best_design['results']['overall_score']:.1f})")

        # Analyze failures
        if failed:
            failure_reasons = self._analyze_failures(failed)
            concerns.append(f"{len(failed)} designs failed simulation")
            for reason, count in failure_reasons.items():
                concerns.append(f"  - {reason}: {count} occurrences")

        # Compare to hypothesis expectations
        hypothesis_met = self._evaluate_hypothesis_outcome(
            experiment_results,
            hypothesis
        )

        if hypothesis_met:
            insights.append(f"Hypothesis '{hypothesis.get('statement', 'N/A')[:60]}...' appears CONFIRMED")
        else:
            insights.append(f"Hypothesis '{hypothesis.get('statement', 'N/A')[:60]}...' appears REFUTED or INCONCLUSIVE")
            recommendations.append("Consider revising hypothesis or test protocol")

        # Identify breakthrough designs
        breakthroughs = self._identify_breakthroughs(experiment_results, experiment_history)
        if breakthroughs:
            insights.append(f"Identified {len(breakthroughs)} breakthrough designs")
            for b in breakthroughs:
                insights.append(f"  - {b['design_id']}: {b['reason']}")

        # Generate LLM-based critique
        llm_critique = self._generate_result_critique(
            experiment_results,
            hypothesis,
            insights,
            concerns
        )

        # Overall verdict
        if len(successful) == 0:
            verdict = "reject"
        elif len(successful) >= len(experiment_results) * 0.7:
            verdict = "approve"
        else:
            verdict = "revise"

        return {
            "verdict": verdict,
            "summary": llm_critique,
            "confidence": 0.80,
            "results_analyzed": len(experiment_results),
            "successful_designs": len(successful),
            "failed_designs": len(failed),
            "hypothesis_met": hypothesis_met,
            "insights": insights,
            "concerns": concerns,
            "breakthroughs": breakthroughs,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }

    def _check_physical_constraints(self, params: Dict[str, float]) -> List[str]:
        """
        Check design for physical constraint violations.

        Args:
            params: Design parameters

        Returns:
            List of safety issues (empty if safe)
        """
        issues = []

        # Extract key parameters
        loa = params.get("length_overall", 18.0)
        beam = params.get("beam", 6.0)
        hull_spacing = params.get("hull_spacing", 4.5)
        hull_depth = params.get("hull_depth", 2.5)
        freeboard = params.get("freeboard", 1.5)
        displacement = params.get("displacement", 45000.0)

        # Check beam-to-length ratio
        bl_ratio = beam / loa
        if bl_ratio > self.SAFETY_THRESHOLDS["max_beam_to_length"]:
            issues.append(f"Beam/Length ratio too high: {bl_ratio:.3f} > {self.SAFETY_THRESHOLDS['max_beam_to_length']}")
        elif bl_ratio < self.SAFETY_THRESHOLDS["min_beam_to_length"]:
            issues.append(f"Beam/Length ratio too low: {bl_ratio:.3f} < {self.SAFETY_THRESHOLDS['min_beam_to_length']}")

        # Check hull spacing vs beam
        if hull_spacing >= beam:
            issues.append(f"Hull spacing ({hull_spacing:.2f}m) >= beam ({beam:.2f}m) - physically implausible")

        # Check freeboard
        if freeboard < self.SAFETY_THRESHOLDS["min_freeboard"]:
            issues.append(f"Freeboard too low: {freeboard:.2f}m < {self.SAFETY_THRESHOLDS['min_freeboard']}m")

        # Check freeboard vs depth
        if freeboard >= hull_depth:
            issues.append(f"Freeboard ({freeboard:.2f}m) >= depth ({hull_depth:.2f}m) - invalid")

        # Check displacement-to-length ratio
        disp_to_length = displacement / (loa ** 3)
        if disp_to_length > self.SAFETY_THRESHOLDS["max_displacement_to_length"]:
            issues.append(f"Displacement/Length^3 ratio very high: {disp_to_length:.1f}")

        return issues

    def _validate_against_hypothesis(
        self,
        design: Dict[str, Any],
        hypothesis: Dict[str, Any]
    ) -> List[str]:
        """
        Validate design coherence with hypothesis.

        Args:
            design: Design dictionary
            hypothesis: Hypothesis being tested

        Returns:
            List of coherence issues (empty if coherent)
        """
        issues = []

        # Get test protocol
        test_protocol = hypothesis.get("test_protocol", {})
        params_to_vary = test_protocol.get("parameters_to_vary", [])
        ranges = test_protocol.get("ranges", [])

        # Check if varied parameters are actually being varied
        params = design.get("parameters", {})

        for i, param_name in enumerate(params_to_vary):
            if param_name in params:
                value = params[param_name]
                if i < len(ranges):
                    min_val, max_val = ranges[i]
                    # Check if value is outside expected range (might be constraint-adjusted)
                    if value < min_val * 0.9 or value > max_val * 1.1:
                        issues.append(
                            f"Parameter '{param_name}' = {value:.2f} outside expected range [{min_val:.2f}, {max_val:.2f}]"
                        )

        return issues

    def _detect_redundancy(
        self,
        designs: List[Dict[str, Any]],
        experiment_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect redundant designs (duplicates or near-duplicates).

        Args:
            designs: New designs to check
            experiment_history: Past experiments

        Returns:
            List of redundancy warnings
        """
        redundancies = []

        # Extract parameter vectors from history
        historical_params = []
        for exp in experiment_history[-100:]:  # Check last 100 experiments
            if "parameters" in exp:
                historical_params.append(exp["parameters"])

        # Check each design
        for design in designs:
            params = design.get("parameters", {})
            design_id = design.get("design_id", "unknown")

            # Check against history
            for hist_params in historical_params:
                distance = self._parameter_distance(params, hist_params)
                if distance < self.REDUNDANCY_THRESHOLD:
                    redundancies.append({
                        "design_id": design_id,
                        "similarity": f"{(1 - distance) * 100:.1f}%",
                        "message": f"Very similar to historical experiment (distance: {distance:.3f})"
                    })
                    break

        # Check for duplicates within new batch
        for i, design1 in enumerate(designs):
            for j, design2 in enumerate(designs[i+1:], i+1):
                params1 = design1.get("parameters", {})
                params2 = design2.get("parameters", {})
                distance = self._parameter_distance(params1, params2)

                if distance < self.REDUNDANCY_THRESHOLD:
                    redundancies.append({
                        "design_id": f"{design1.get('design_id', 'unknown')} & {design2.get('design_id', 'unknown')}",
                        "similarity": f"{(1 - distance) * 100:.1f}%",
                        "message": f"Designs are near-duplicates (distance: {distance:.3f})"
                    })

        return redundancies

    def _parameter_distance(
        self,
        params1: Dict[str, float],
        params2: Dict[str, float]
    ) -> float:
        """
        Calculate normalized Euclidean distance between parameter sets.

        Args:
            params1: First parameter set
            params2: Second parameter set

        Returns:
            Distance value (0 = identical, 1 = maximally different)
        """
        # Parameter ranges for normalization
        from agents.experimental_architect_agent import ExperimentalArchitectAgent
        ranges = ExperimentalArchitectAgent.PARAMETER_RANGES

        # Get common parameters
        common_params = set(params1.keys()) & set(params2.keys()) & set(ranges.keys())

        if not common_params:
            return 1.0  # Completely different

        # Calculate normalized squared differences
        squared_diffs = []
        for param in common_params:
            val1 = params1[param]
            val2 = params2[param]
            min_val, max_val = ranges[param]
            range_size = max_val - min_val

            if range_size > 0:
                normalized_diff = (val1 - val2) / range_size
                squared_diffs.append(normalized_diff ** 2)

        # Euclidean distance
        if squared_diffs:
            return np.sqrt(np.mean(squared_diffs))
        else:
            return 1.0

    def _determine_verdict(
        self,
        safety_flags: List[Dict],
        concerns: List[Dict],
        redundancies: List[Dict]
    ) -> str:
        """
        Determine overall verdict for pre-simulation review.

        Args:
            safety_flags: List of safety issues
            concerns: List of concerns
            redundancies: List of redundant designs

        Returns:
            Verdict: "approve", "reject", or "revise"
        """
        if len(safety_flags) > 0:
            return "reject"  # Safety issues must be fixed
        elif len(redundancies) > 3:
            return "revise"  # Too many redundant designs
        elif len(concerns) > 5:
            return "revise"  # Many concerns
        else:
            return "approve"

    def _analyze_failures(
        self,
        failed_designs: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Analyze failure reasons from failed designs.

        Args:
            failed_designs: List of designs that failed simulation

        Returns:
            Dictionary of failure reason counts
        """
        failure_counts = {}

        for design in failed_designs:
            reasons = design.get("results", {}).get("failure_reasons", ["unknown"])
            for reason in reasons:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1

        return failure_counts

    def _evaluate_hypothesis_outcome(
        self,
        experiment_results: List[Dict[str, Any]],
        hypothesis: Dict[str, Any]
    ) -> bool:
        """
        Evaluate if hypothesis success criteria were met.

        Args:
            experiment_results: Results from simulation
            hypothesis: Hypothesis with success criteria

        Returns:
            True if hypothesis appears confirmed
        """
        success_criteria = hypothesis.get("success_criteria", "")

        # Simple heuristic: check if any design met the criteria
        # In a real implementation, would parse and evaluate the criteria string

        successful = [r for r in experiment_results if r.get("results", {}).get("is_valid", False)]

        if not successful:
            return False  # No valid results

        # Check if best design improved over expected
        best_score = max(r["results"]["overall_score"] for r in successful)

        # Heuristic: consider hypothesis met if best score > 70
        return best_score > 70.0

    def _identify_breakthroughs(
        self,
        experiment_results: List[Dict[str, Any]],
        experiment_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify breakthrough designs (significantly better than history).

        Args:
            experiment_results: New results
            experiment_history: Historical results

        Returns:
            List of breakthrough design descriptions
        """
        breakthroughs = []

        # Get historical best score
        if experiment_history:
            historical_scores = [
                exp.get("results", {}).get("overall_score", 0)
                for exp in experiment_history
                if exp.get("results", {}).get("is_valid", False)
            ]
            historical_best = max(historical_scores) if historical_scores else 0
        else:
            historical_best = 0

        # Check new results
        for result in experiment_results:
            if result.get("results", {}).get("is_valid", False):
                score = result["results"]["overall_score"]
                improvement = score - historical_best

                # Consider breakthrough if >5% improvement
                if improvement > 5.0:
                    breakthroughs.append({
                        "design_id": result.get("design_id"),
                        "score": score,
                        "improvement": f"+{improvement:.1f}",
                        "reason": f"New best design ({improvement:.1f}% improvement over historical best)"
                    })

        return breakthroughs

    def _generate_critique_summary(
        self,
        verdict: str,
        num_designs: int,
        safety_flags: List[Dict],
        concerns: List[Dict],
        redundancies: List[Dict]
    ) -> str:
        """
        Generate natural language summary of pre-simulation review.

        Args:
            verdict: Overall verdict
            num_designs: Number of designs reviewed
            safety_flags: Safety issues found
            concerns: Concerns found
            redundancies: Redundant designs found

        Returns:
            Summary string
        """
        summary_parts = [f"Reviewed {num_designs} experimental designs."]

        if verdict == "approve":
            summary_parts.append("All designs are safe and appropriate for simulation.")
        elif verdict == "reject":
            summary_parts.append(f"Found {len(safety_flags)} designs with safety violations. Must revise before simulation.")
        elif verdict == "revise":
            summary_parts.append("Designs have concerns that should be addressed.")

        if redundancies:
            summary_parts.append(f"Detected {len(redundancies)} redundant designs.")

        return " ".join(summary_parts)

    def _generate_result_critique(
        self,
        experiment_results: List[Dict[str, Any]],
        hypothesis: Dict[str, Any],
        insights: List[str],
        concerns: List[str]
    ) -> str:
        """
        Generate natural language critique of results.

        Args:
            experiment_results: Simulation results
            hypothesis: Hypothesis tested
            insights: Key insights
            concerns: Concerns identified

        Returns:
            Critique string
        """
        summary_parts = [f"Analyzed {len(experiment_results)} simulation results."]

        if insights:
            summary_parts.append("Key insights: " + "; ".join(insights[:3]))

        if concerns:
            summary_parts.append("Concerns: " + "; ".join(concerns[:2]))

        return " ".join(summary_parts)
