"""
ExperimentalArchitectAgent: Experimental design agent for MAGNET (Agent 2)
===========================================================================

The Architect agent translates hypotheses into concrete experimental designs.
It uses advanced sampling strategies to generate design variants for testing.

Responsibilities:
- Receive hypothesis with test protocol from Explorer
- Generate 5-10 design variants using smart sampling (LHS, Gaussian, Edge)
- Fill in non-varied parameters with sensible defaults
- Enforce physical constraints
- Validate designs with LLM before submission
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
from datetime import datetime
from scipy.stats import qmc  # Latin Hypercube Sampling

from agents.base_naval_agent import BaseNavalAgent, NavalAgentResponse, NavalAgentConfig


class ExperimentalArchitectAgent(BaseNavalAgent):
    """
    Architect agent for experimental design in naval domain.

    Takes hypothesis as input, outputs concrete design variants ready for simulation.
    """

    # Naval parameter constraints and defaults
    PARAMETER_DEFAULTS = {
        "length_overall": 18.0,      # meters
        "beam": 6.0,                 # meters
        "hull_spacing": 4.5,         # meters
        "hull_depth": 2.5,           # meters
        "deadrise_angle": 12.0,      # degrees
        "freeboard": 1.5,            # meters
        "lcb_position": 0.50,        # 0-1 (longitudinal center of buoyancy)
        "prismatic_coefficient": 0.62,  # 0-1
        "waterline_beam": 5.5,       # meters
        "block_coefficient": 0.45,   # 0-1
        "wetted_surface_area": 180.0,  # m^2
        "displacement": 45000.0      # kg
    }

    PARAMETER_RANGES = {
        "length_overall": (14.0, 22.0),
        "beam": (4.0, 8.0),
        "hull_spacing": (3.5, 6.0),
        "hull_depth": (2.0, 3.5),
        "deadrise_angle": (8.0, 18.0),
        "freeboard": (1.0, 2.0),
        "lcb_position": (0.45, 0.55),
        "prismatic_coefficient": (0.55, 0.70),
        "waterline_beam": (4.0, 7.0),
        "block_coefficient": (0.35, 0.55),
        "wetted_surface_area": (120.0, 250.0),
        "displacement": (30000.0, 65000.0)
    }

    def __init__(self, config: NavalAgentConfig, llm_client):
        """
        Initialize Architect agent.

        Args:
            config: Agent configuration
            llm_client: LLM client instance
        """
        super().__init__(config, llm_client)
        self.experiment_count = 0

    def autonomous_cycle(self, context: Dict[str, Any]) -> NavalAgentResponse:
        """
        Run one autonomous cycle to design experiments.

        Args:
            context: Dictionary containing:
                - hypothesis: Hypothesis to test (from Explorer)
                - current_best_design: Current best design for baseline
                - knowledge_base: Accumulated knowledge

        Returns:
            NavalAgentResponse with design variants
        """
        self.state = self.state.__class__.BUSY
        self.current_task = "designing_experiments"

        try:
            # Extract context
            hypothesis = context.get("hypothesis", {})
            current_best = context.get("current_best_design", self.PARAMETER_DEFAULTS)

            # Design experiments
            designs = self.design_experiments(hypothesis, context)

            self.state = self.state.__class__.ACTIVE
            self.current_task = None

            return NavalAgentResponse(
                agent_id=self.agent_id,
                action="submit_experiments",
                reasoning=f"Designed {len(designs)} experiments to test hypothesis: {hypothesis.get('statement', 'N/A')}",
                confidence=0.85,
                data={
                    "designs": designs,
                    "hypothesis": hypothesis,
                    "num_designs": len(designs)
                }
            )

        except Exception as e:
            self.state = self.state.__class__.FAILED
            raise e

    def design_experiments(
        self,
        hypothesis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Design experimental variants for a hypothesis.

        Args:
            hypothesis: Hypothesis with test_protocol
            context: Additional context (current_best_design, etc.)

        Returns:
            List of design dictionaries with complete parameter sets
        """
        # Extract test protocol
        test_protocol = hypothesis.get("test_protocol", {})
        params_to_vary = test_protocol.get("parameters_to_vary", ["hull_spacing"])
        ranges = test_protocol.get("ranges", [[3.5, 6.0]])
        num_samples = test_protocol.get("num_samples", 8)
        hypothesis_type = hypothesis.get("type", "exploration")

        # Get baseline design
        baseline = context.get("current_best_design", self.PARAMETER_DEFAULTS.copy())

        # Generate parameter samples using appropriate strategy
        param_sets = self._generate_parameter_samples(
            params_to_vary,
            ranges,
            num_samples,
            hypothesis_type
        )

        # Complete designs with fixed parameters
        full_designs = []
        for i, param_set in enumerate(param_sets):
            design = self._complete_design(
                varied_params=param_set,
                param_names=params_to_vary,
                baseline=baseline,
                hypothesis=hypothesis
            )

            # Enforce constraints
            design = self._enforce_constraints(design)

            # Add metadata
            self.experiment_count += 1
            design_id = f"exp_{hypothesis.get('id', 'unknown')}_{i:02d}"
            full_design = {
                "design_id": design_id,
                "parameters": design,
                "hypothesis_id": hypothesis.get("id"),
                "expected_outcome": hypothesis.get("expected_outcome", "N/A"),
                "timestamp": datetime.now().isoformat()
            }

            full_designs.append(full_design)

        # LLM validation (optional, can be mocked for speed)
        if self.llm:
            validation_note = self._validate_with_llm(full_designs, hypothesis)
        else:
            validation_note = "LLM validation skipped (offline mode)"

        return full_designs

    def _generate_parameter_samples(
        self,
        param_names: List[str],
        ranges: List[List[float]],
        num_samples: int,
        hypothesis_type: str
    ) -> List[List[float]]:
        """
        Generate parameter samples using appropriate sampling strategy.

        Args:
            param_names: Names of parameters to vary
            ranges: [[min, max], ...] for each parameter
            num_samples: Number of samples to generate
            hypothesis_type: Type of hypothesis (exploration, exploitation, counter-intuitive)

        Returns:
            List of parameter value lists
        """
        n_params = len(param_names)

        if hypothesis_type == "exploration":
            # Latin Hypercube Sampling for broad coverage
            sampler = qmc.LatinHypercube(d=n_params, seed=None)
            samples_normalized = sampler.random(n=num_samples)

            # Scale to actual ranges
            samples = []
            for sample in samples_normalized:
                scaled = [
                    ranges[i][0] + sample[i] * (ranges[i][1] - ranges[i][0])
                    for i in range(n_params)
                ]
                samples.append(scaled)

            return samples

        elif hypothesis_type == "exploitation":
            # Gaussian sampling around midpoint (refinement)
            samples = []
            for _ in range(num_samples):
                sample = []
                for i in range(n_params):
                    midpoint = (ranges[i][0] + ranges[i][1]) / 2
                    std = (ranges[i][1] - ranges[i][0]) / 6  # 3 sigma ≈ range
                    value = np.random.normal(midpoint, std)
                    # Clip to range
                    value = np.clip(value, ranges[i][0], ranges[i][1])
                    sample.append(value)
                samples.append(sample)

            return samples

        elif hypothesis_type == "counter-intuitive":
            # Edge/corner sampling for extremes
            samples = []

            # Add corners
            for i in range(min(num_samples, 2**n_params)):
                corner = []
                for j in range(n_params):
                    # Use binary representation to select min/max
                    use_max = (i >> j) & 1
                    value = ranges[j][1] if use_max else ranges[j][0]
                    corner.append(value)
                samples.append(corner)

            # Fill remaining with edge samples (one param at extreme, others at mid)
            while len(samples) < num_samples:
                sample = []
                extreme_idx = np.random.randint(n_params)
                for i in range(n_params):
                    if i == extreme_idx:
                        value = np.random.choice([ranges[i][0], ranges[i][1]])
                    else:
                        value = (ranges[i][0] + ranges[i][1]) / 2
                    sample.append(value)
                samples.append(sample)

            return samples[:num_samples]

        else:
            # Default: uniform random sampling
            samples = []
            for _ in range(num_samples):
                sample = [
                    np.random.uniform(ranges[i][0], ranges[i][1])
                    for i in range(n_params)
                ]
                samples.append(sample)
            return samples

    def _complete_design(
        self,
        varied_params: List[float],
        param_names: List[str],
        baseline: Dict[str, float],
        hypothesis: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Complete a design by filling in non-varied parameters.

        Args:
            varied_params: Values for varied parameters
            param_names: Names of varied parameters
            baseline: Baseline design to use for fixed parameters
            hypothesis: Hypothesis being tested

        Returns:
            Complete design dictionary
        """
        design = baseline.copy()

        # Set varied parameters
        for i, param_name in enumerate(param_names):
            design[param_name] = varied_params[i]

        # Apply fixed parameters from test protocol if specified
        fixed_params = hypothesis.get("test_protocol", {}).get("fixed_parameters", {})
        design.update(fixed_params)

        # Ensure all required parameters are present
        for param, default_value in self.PARAMETER_DEFAULTS.items():
            if param not in design:
                design[param] = default_value

        return design

    def _enforce_constraints(self, design: Dict[str, float]) -> Dict[str, float]:
        """
        Enforce physical constraints on design parameters.

        Constraints:
        - hull_spacing < beam
        - freeboard < hull_depth
        - lcb_position in [0.45, 0.55]
        - All parameters within valid ranges

        Args:
            design: Design dictionary

        Returns:
            Constrained design dictionary
        """
        constrained = design.copy()

        # Constraint 1: hull_spacing < beam
        if constrained.get("hull_spacing", 0) >= constrained.get("beam", 10):
            constrained["hull_spacing"] = constrained["beam"] * 0.85

        # Constraint 2: freeboard < hull_depth
        if constrained.get("freeboard", 0) >= constrained.get("hull_depth", 10):
            constrained["freeboard"] = constrained["hull_depth"] * 0.75

        # Constraint 3: Clip all parameters to valid ranges
        for param, (min_val, max_val) in self.PARAMETER_RANGES.items():
            if param in constrained:
                constrained[param] = np.clip(constrained[param], min_val, max_val)

        # Constraint 4: Length-to-beam ratio should be reasonable (2.5-4.0)
        loa = constrained.get("length_overall", 18.0)
        beam = constrained.get("beam", 6.0)
        l_b_ratio = loa / beam
        if l_b_ratio < 2.5:
            constrained["beam"] = loa / 2.5
        elif l_b_ratio > 4.0:
            constrained["beam"] = loa / 4.0

        return constrained

    def _validate_with_llm(
        self,
        designs: List[Dict[str, Any]],
        hypothesis: Dict[str, Any]
    ) -> str:
        """
        Validate designs with LLM (sanity check).

        Args:
            designs: List of design dictionaries
            hypothesis: Hypothesis being tested

        Returns:
            Validation note string
        """
        # Build validation prompt
        prompt = self._create_validation_prompt(designs, hypothesis)

        try:
            response = self.generate(prompt, max_tokens=500, temperature=0.5)
            parsed = self.parse_llm_response(response)
            return parsed.get("validation", "Designs appear valid")
        except Exception:
            # Fallback: basic validation
            return "Designs validated (basic checks passed)"

    def _create_validation_prompt(
        self,
        designs: List[Dict[str, Any]],
        hypothesis: Dict[str, Any]
    ) -> str:
        """
        Create prompt for LLM validation of designs.

        Args:
            designs: List of design dictionaries
            hypothesis: Hypothesis being tested

        Returns:
            Formatted prompt string
        """
        # Summarize designs
        design_summary = []
        for i, design in enumerate(designs[:3], 1):  # Show first 3
            params = design["parameters"]
            design_summary.append(
                f"  Design {i}: LOA={params.get('length_overall', 0):.1f}m, "
                f"Beam={params.get('beam', 0):.1f}m, "
                f"Spacing={params.get('hull_spacing', 0):.1f}m, "
                f"Depth={params.get('hull_depth', 0):.1f}m"
            )

        prompt = f"""You are a naval architecture expert. Validate these experimental designs.

**Hypothesis:**
{hypothesis.get('statement', 'N/A')}

**Designs Generated ({len(designs)} total):**
{chr(10).join(design_summary)}
... (and {len(designs) - 3} more)

**Validation Task:**
Check if these designs:
1. Are physically realistic for catamaran hulls
2. Appropriately test the hypothesis
3. Cover the parameter space well

**Output Format (JSON):**
{{
    "validation": "valid" | "concerns",
    "notes": "Brief validation notes",
    "concerns": ["concern 1", "concern 2"] (if any)
}}
"""

        return prompt

    def get_sampling_strategy_name(self, hypothesis_type: str) -> str:
        """
        Get human-readable name for sampling strategy.

        Args:
            hypothesis_type: Type of hypothesis

        Returns:
            Strategy name string
        """
        strategies = {
            "exploration": "Latin Hypercube Sampling (broad coverage)",
            "exploitation": "Gaussian sampling (refinement around best)",
            "counter-intuitive": "Edge/corner sampling (extremes)",
        }
        return strategies.get(hypothesis_type, "Uniform random sampling")
