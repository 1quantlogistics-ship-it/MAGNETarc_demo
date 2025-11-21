"""
ExplorerAgent: Hypothesis generation agent for MAGNET
======================================================

The Explorer agent analyzes past experimental results, identifies patterns and gaps,
and generates novel hypotheses for testing. It acts as Agent 1 in the research cycle.

Responsibilities:
- Extract insights from experimental history
- Identify unexplored regions of design space
- Generate testable hypotheses with clear test protocols
- Rank hypotheses by novelty, feasibility, and potential impact
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from agents.base_naval_agent import BaseNavalAgent, NavalAgentResponse, NavalAgentConfig


class ExplorerAgent(BaseNavalAgent):
    """
    Explorer agent for hypothesis generation in naval design space.

    Takes knowledge base and experimental history as input,
    generates ranked hypotheses with test protocols.
    """

    def __init__(self, config: NavalAgentConfig, llm_client):
        """
        Initialize Explorer agent.

        Args:
            config: Agent configuration
            llm_client: LLM client instance
        """
        super().__init__(config, llm_client)
        self.hypothesis_count = 0

    def autonomous_cycle(self, context: Dict[str, Any]) -> NavalAgentResponse:
        """
        Run one autonomous cycle to generate hypotheses.

        Args:
            context: Dictionary containing:
                - knowledge_base: Accumulated insights and patterns
                - experiment_history: Past experiments with results
                - current_best: Current best design
                - cycle_number: Current research cycle

        Returns:
            NavalAgentResponse with hypothesis
        """
        self.state = self.state.__class__.BUSY
        self.current_task = "generating_hypothesis"

        try:
            # Extract context
            knowledge_base = context.get("knowledge_base", {})
            experiment_history = context.get("experiment_history", [])
            current_best = context.get("current_best", {})
            cycle_number = context.get("cycle_number", 0)

            # Generate hypothesis using LLM
            hypothesis = self._generate_hypothesis(
                knowledge_base,
                experiment_history,
                current_best,
                cycle_number
            )

            self.state = self.state.__class__.ACTIVE
            self.current_task = None

            return NavalAgentResponse(
                agent_id=self.agent_id,
                action="submit_hypothesis",
                reasoning=hypothesis.get("reasoning", "Generated novel hypothesis"),
                confidence=hypothesis.get("confidence", 0.7),
                data={"hypothesis": hypothesis}
            )

        except Exception as e:
            self.state = self.state.__class__.FAILED
            raise e

    def _generate_hypothesis(
        self,
        knowledge_base: Dict[str, Any],
        experiment_history: List[Dict[str, Any]],
        current_best: Dict[str, Any],
        cycle_number: int
    ) -> Dict[str, Any]:
        """
        Generate a hypothesis using LLM-based reasoning.

        Args:
            knowledge_base: Accumulated insights
            experiment_history: Past experiments
            current_best: Current best design
            cycle_number: Current cycle number

        Returns:
            Hypothesis dictionary with test protocol
        """
        # Extract insights from history
        insights = self._extract_insights_from_history(experiment_history)

        # Identify gaps in design space
        gaps = self._identify_gaps(experiment_history, knowledge_base)

        # Build prompt for LLM
        prompt = self._create_hypothesis_prompt(
            insights,
            gaps,
            current_best,
            cycle_number
        )

        # Generate hypotheses from LLM
        response = self.generate(prompt, max_tokens=1500, temperature=0.8)

        # Parse LLM response
        try:
            hypothesis_data = self.parse_llm_response(response)
        except ValueError:
            # Fallback: create a simple exploration hypothesis
            hypothesis_data = self._create_fallback_hypothesis(current_best)

        # Add unique ID
        self.hypothesis_count += 1
        hypothesis_data["id"] = f"hyp_{cycle_number:03d}_{self.hypothesis_count:02d}"
        hypothesis_data["cycle_number"] = cycle_number
        hypothesis_data["timestamp"] = datetime.now().isoformat()

        return hypothesis_data

    def _extract_insights_from_history(
        self,
        experiment_history: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract key insights from experimental history using LLM.

        Args:
            experiment_history: List of past experiments with results

        Returns:
            List of insight strings
        """
        if not experiment_history:
            return ["No prior experiments - this is initial exploration"]

        # Take last 20 experiments for analysis
        recent_experiments = experiment_history[-20:]

        # Build summary prompt
        prompt = f"""Analyze the following naval design experiments and extract 3-5 key insights:

Experiments (showing parameters and scores):
{self._format_experiments_for_prompt(recent_experiments)}

Provide insights in JSON format:
{{
    "insights": [
        "Insight 1: ...",
        "Insight 2: ...",
        ...
    ]
}}

Focus on:
- Parameter correlations (e.g., "increasing X improves Y")
- Trade-offs discovered (e.g., "stability vs speed")
- Unexpected results
- Regions of design space that perform well/poorly
"""

        try:
            response = self.generate(prompt, max_tokens=800, temperature=0.6)
            parsed = self.parse_llm_response(response)
            return parsed.get("insights", [])
        except Exception:
            # Fallback: basic statistical insights
            return self._compute_basic_insights(recent_experiments)

    def _identify_gaps(
        self,
        experiment_history: List[Dict[str, Any]],
        knowledge_base: Dict[str, Any]
    ) -> List[str]:
        """
        Identify unexplored regions of design space.

        Args:
            experiment_history: Past experiments
            knowledge_base: Accumulated knowledge

        Returns:
            List of gap descriptions
        """
        if not experiment_history:
            return ["Entire design space is unexplored"]

        # Simple heuristic: identify parameter ranges not well-explored
        gaps = []

        # Define parameter ranges
        parameter_ranges = {
            "length_overall": (14.0, 22.0),
            "beam": (4.0, 8.0),
            "hull_spacing": (3.5, 6.0),
            "hull_depth": (2.0, 3.5),
            "deadrise_angle": (8.0, 18.0),
            "freeboard": (1.0, 2.0),
            "lcb_position": (0.45, 0.55),
            "prismatic_coefficient": (0.55, 0.70)
        }

        # Check which ranges are under-explored
        for param, (min_val, max_val) in parameter_ranges.items():
            values = [
                exp.get("parameters", {}).get(param, (min_val + max_val) / 2)
                for exp in experiment_history
            ]

            if values:
                coverage = (max(values) - min(values)) / (max_val - min_val)
                if coverage < 0.5:
                    gaps.append(f"Parameter '{param}' is under-explored (only {coverage*100:.0f}% coverage)")

        return gaps[:5]  # Return top 5 gaps

    def _create_hypothesis_prompt(
        self,
        insights: List[str],
        gaps: List[str],
        current_best: Dict[str, Any],
        cycle_number: int
    ) -> str:
        """
        Create prompt for LLM to generate hypothesis.

        Args:
            insights: Extracted insights
            gaps: Identified gaps
            current_best: Current best design
            cycle_number: Current cycle

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a naval architecture research AI designing catamaran hulls. Generate ONE novel hypothesis to test.

**Current Knowledge:**
Insights from past experiments:
{self._format_list(insights)}

Gaps in design space:
{self._format_list(gaps)}

Current best design:
- LOA: {current_best.get('length_overall', 18.0)}m
- Beam: {current_best.get('beam', 6.0)}m
- Hull spacing: {current_best.get('hull_spacing', 4.5)}m
- Stability score: {current_best.get('stability_score', 75.0)}
- Speed score: {current_best.get('speed_score', 70.0)}

**Your Task:**
Generate a hypothesis that is:
1. Novel and testable
2. Based on naval architecture principles
3. Exploits a gap or insight
4. Has clear success criteria

**Output Format (JSON):**
{{
    "statement": "Clear hypothesis statement (e.g., 'Increasing hull spacing from 4.5m to 5.2m will improve stability without significant speed penalty')",
    "type": "exploration" | "exploitation" | "counter-intuitive",
    "test_protocol": {{
        "parameters_to_vary": ["param1", "param2"],
        "ranges": [[min1, max1], [min2, max2]],
        "num_samples": 8,
        "fixed_parameters": {{"param3": value3}}
    }},
    "expected_outcome": "What success looks like",
    "success_criteria": "Quantitative criteria (e.g., stability_score > 78 AND speed_score > 68)",
    "reasoning": "Why this hypothesis is worth testing",
    "confidence": 0.75
}}

**Hypothesis Types:**
- exploration: Broad search in under-explored region
- exploitation: Refinement around current best
- counter-intuitive: Test unconventional idea
"""

        return prompt

    def _create_fallback_hypothesis(
        self,
        current_best: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a simple exploration hypothesis as fallback.

        Args:
            current_best: Current best design

        Returns:
            Hypothesis dictionary
        """
        return {
            "statement": "Explore the effect of varying hull spacing on overall performance",
            "type": "exploration",
            "test_protocol": {
                "parameters_to_vary": ["hull_spacing"],
                "ranges": [[3.5, 6.0]],
                "num_samples": 10,
                "fixed_parameters": {}
            },
            "expected_outcome": "Identify optimal hull spacing range",
            "success_criteria": "Find configuration with stability_score > 75",
            "reasoning": "Hull spacing is a critical parameter affecting both stability and drag",
            "confidence": 0.6
        }

    def _format_experiments_for_prompt(
        self,
        experiments: List[Dict[str, Any]]
    ) -> str:
        """Format experiments for prompt (concise)."""
        lines = []
        for i, exp in enumerate(experiments[-10:], 1):  # Last 10 only
            params = exp.get("parameters", {})
            results = exp.get("results", {})
            lines.append(
                f"  {i}. LOA={params.get('length_overall', 0):.1f}, "
                f"Beam={params.get('beam', 0):.1f}, "
                f"Spacing={params.get('hull_spacing', 0):.1f} → "
                f"Stability={results.get('stability_score', 0):.1f}, "
                f"Speed={results.get('speed_score', 0):.1f}"
            )
        return "\n".join(lines)

    def _format_list(self, items: List[str]) -> str:
        """Format list for prompt."""
        if not items:
            return "  (none)"
        return "\n".join(f"  - {item}" for item in items)

    def _compute_basic_insights(
        self,
        experiments: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Compute basic statistical insights as fallback.

        Args:
            experiments: List of experiments

        Returns:
            List of insight strings
        """
        if not experiments:
            return []

        insights = []

        # Find best experiment
        best_exp = max(
            experiments,
            key=lambda e: e.get("results", {}).get("stability_score", 0)
        )
        best_params = best_exp.get("parameters", {})
        insights.append(
            f"Best stability ({best_exp.get('results', {}).get('stability_score', 0):.1f}) "
            f"at LOA={best_params.get('length_overall', 0):.1f}m, "
            f"Spacing={best_params.get('hull_spacing', 0):.1f}m"
        )

        return insights
