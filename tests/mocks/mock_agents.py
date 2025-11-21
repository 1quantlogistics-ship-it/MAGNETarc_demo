"""
Mock Agent Implementations for Testing

Provides simple mock versions of LLM-based agents for integration testing
without requiring real LLM infrastructure.

Mock Agents:
- MockExplorer: Generates simple hypotheses
- MockArchitect: Creates design batches via random sampling
- MockCritic: Simple validation (always approves for testing)
- MockHistorian: Formats results into structured insights

All agents return proper dict formats compatible with real LLM agents.
"""

import random
from typing import Dict, List, Any, Optional


class MockExplorer:
    """
    Mock Explorer agent for hypothesis generation.

    Generates simple hypotheses by randomly selecting parameter ranges to explore.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize mock explorer.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.cycle_count = 0

    def autonomous_cycle(self, knowledge_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate hypothesis for next research cycle.

        Args:
            knowledge_context: Historical context from knowledge base

        Returns:
            Hypothesis dict with description and parameter ranges
        """
        self.cycle_count += 1

        # Simple strategy: Pick a random parameter to explore
        parameters = [
            ('length_overall', 16.0, 22.0, 'length'),
            ('hull_spacing', 4.5, 7.0, 'spacing'),
            ('beam', 1.8, 2.4, 'beam'),
            ('design_speed', 20.0, 32.0, 'speed'),
            ('prismatic_coefficient', 0.56, 0.68, 'Cp'),
            ('block_coefficient', 0.38, 0.48, 'Cb'),
        ]

        # Select random parameter
        param_name, min_val, max_val, label = random.choice(parameters)

        hypothesis = {
            'cycle': self.cycle_count,
            'type': 'parameter_sweep',
            'description': f"Exploring effect of {label} on overall performance",
            'parameter': param_name,
            'parameter_range': {
                param_name: (min_val, max_val)
            },
            'expected_outcome': f"Higher {label} may improve performance",
            'confidence': random.uniform(0.5, 0.8),
        }

        return hypothesis


class MockArchitect:
    """
    Mock Architect agent for experimental design.

    Generates batches of designs using simple random sampling within
    hypothesis-specified parameter ranges.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize mock architect.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def design_experiments(
        self,
        hypothesis: Dict[str, Any],
        baseline_design: Dict[str, Any],
        n_designs: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Generate experimental designs based on hypothesis.

        Args:
            hypothesis: Hypothesis from Explorer
            baseline_design: Baseline design to vary
            n_designs: Number of designs to generate

        Returns:
            List of design parameter dicts
        """
        designs = []

        # Get parameter to vary from hypothesis
        param_ranges = hypothesis.get('parameter_range', {})

        for i in range(n_designs):
            design = baseline_design.copy()

            # Remove non-physics keys
            design.pop('name', None)
            design.pop('description', None)

            # Vary parameters specified in hypothesis
            for param_name, (min_val, max_val) in param_ranges.items():
                # Linear sweep or random sampling
                if n_designs > 1:
                    # Linear sweep
                    fraction = i / (n_designs - 1)
                    value = min_val + fraction * (max_val - min_val)
                else:
                    # Random
                    value = random.uniform(min_val, max_val)

                design[param_name] = value

            # Add small random perturbations to other parameters (exploration)
            if random.random() < 0.3:  # 30% chance
                # Perturb another parameter slightly
                other_params = ['beam', 'deadrise_angle', 'freeboard']
                perturb_param = random.choice(other_params)

                if perturb_param in design:
                    current_value = design[perturb_param]
                    # ±5% perturbation
                    perturbation = current_value * random.uniform(-0.05, 0.05)
                    design[perturb_param] = current_value + perturbation

            designs.append(design)

        return designs


class MockCritic:
    """
    Mock Critic agent for validation.

    Performs simple rule-based validation. For testing, mostly approves
    designs unless obviously invalid.
    """

    def __init__(self):
        """Initialize mock critic."""
        pass

    def review_experiments(
        self,
        designs: List[Dict[str, Any]],
        hypothesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review experimental designs for safety and coherence.

        Args:
            designs: Proposed designs
            hypothesis: Hypothesis being tested

        Returns:
            Review dict with approval/rejection
        """
        # Simple validation: check if designs are coherent
        issues = []

        for i, design in enumerate(designs):
            # Check basic constraints
            if design.get('length_overall', 0) < 8.0:
                issues.append(f"Design {i}: Length too small")

            if design.get('hull_spacing', 0) < 2.0:
                issues.append(f"Design {i}: Hull spacing too narrow")

            if design.get('design_speed', 0) > 50.0:
                issues.append(f"Design {i}: Speed unrealistic")

        # Approve if fewer than 20% have issues
        approval = len(issues) < len(designs) * 0.2

        review = {
            'approved': approval,
            'n_designs': len(designs),
            'n_issues': len(issues),
            'issues': issues[:10],  # First 10 issues only
            'recommendation': 'approved' if approval else 'revise_designs',
            'confidence': 0.9 if approval else 0.5,
        }

        return review

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze physics simulation results.

        Args:
            results: Physics results

        Returns:
            Analysis dict with insights
        """
        valid_results = [r for r in results if r and r.get('is_valid', False)]

        if not valid_results:
            return {
                'status': 'failed',
                'message': 'No valid results to analyze',
                'insights': [],
            }

        # Calculate statistics
        scores = [r['overall_score'] for r in valid_results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)

        insights = []

        if avg_score > 70.0:
            insights.append("Hypothesis appears promising (avg score > 70)")
        elif avg_score < 50.0:
            insights.append("Hypothesis may need revision (avg score < 50)")
        else:
            insights.append("Hypothesis shows mixed results")

        if max_score > 80.0:
            insights.append(f"Found high-performing design (score {max_score:.1f})")

        analysis = {
            'status': 'success',
            'n_valid': len(valid_results),
            'n_total': len(results),
            'avg_score': avg_score,
            'max_score': max_score,
            'min_score': min_score,
            'insights': insights,
        }

        return analysis


class MockHistorian:
    """
    Mock Historian agent for results summarization.

    Formats physics results into structured insights for knowledge base.
    """

    def __init__(self):
        """Initialize mock historian."""
        pass

    def analyze_batch_results(
        self,
        designs: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        hypothesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze batch results and extract insights.

        Args:
            designs: Design parameters
            results: Physics results
            hypothesis: Hypothesis being tested

        Returns:
            Structured insights dict
        """
        valid_results = [r for r in results if r and r.get('is_valid', False)]

        if not valid_results:
            return {
                'summary': 'Batch failed - no valid results',
                'hypothesis_outcome': 'inconclusive',
                'insights': [],
                'recommendations': ['Review parameter constraints'],
            }

        # Calculate statistics
        scores = [r['overall_score'] for r in valid_results]
        stability_scores = [r['stability_score'] for r in valid_results]
        speed_scores = [r['speed_score'] for r in valid_results]
        efficiency_scores = [r['efficiency_score'] for r in valid_results]

        avg_overall = sum(scores) / len(scores)

        # Determine hypothesis outcome
        # Simple rule: if avg score > 65, hypothesis confirmed
        if avg_overall > 65.0:
            outcome = 'confirmed'
        elif avg_overall < 55.0:
            outcome = 'refuted'
        else:
            outcome = 'inconclusive'

        # Generate insights
        insights = []

        if max(scores) > 75.0:
            best_idx = scores.index(max(scores))
            best_design = designs[best_idx]
            insights.append(
                f"Best design achieved {max(scores):.1f}/100 with "
                f"LOA={best_design['length_overall']:.1f}m"
            )

        if sum(s > 70 for s in stability_scores) > len(stability_scores) * 0.5:
            insights.append("Majority of designs show good stability")

        if sum(s > 70 for s in efficiency_scores) > len(efficiency_scores) * 0.5:
            insights.append("Majority of designs show good efficiency")

        # Recommendations
        recommendations = []

        if outcome == 'confirmed':
            recommendations.append("Continue exploration in this parameter range")
        elif outcome == 'refuted':
            recommendations.append("Consider different parameter ranges")
        else:
            recommendations.append("Refine hypothesis with more targeted experiments")

        summary = {
            'summary': f"Analyzed {len(results)} designs, {len(valid_results)} valid",
            'hypothesis_outcome': outcome,
            'avg_overall_score': avg_overall,
            'avg_stability_score': sum(stability_scores) / len(stability_scores),
            'avg_speed_score': sum(speed_scores) / len(speed_scores),
            'avg_efficiency_score': sum(efficiency_scores) / len(efficiency_scores),
            'insights': insights,
            'recommendations': recommendations,
        }

        return summary

    def compare_to_baseline(
        self,
        results: List[Dict[str, Any]],
        baseline_score: float = 65.0
    ) -> Dict[str, Any]:
        """
        Compare results to baseline performance.

        Args:
            results: Physics results
            baseline_score: Baseline score to compare against

        Returns:
            Comparison dict
        """
        valid_results = [r for r in results if r and r.get('is_valid', False)]

        if not valid_results:
            return {'status': 'no_valid_results'}

        scores = [r['overall_score'] for r in valid_results]
        avg_score = sum(scores) / len(scores)

        better_than_baseline = sum(s > baseline_score for s in scores)
        percentage_better = (better_than_baseline / len(scores)) * 100

        comparison = {
            'baseline_score': baseline_score,
            'avg_new_score': avg_score,
            'improvement': avg_score - baseline_score,
            'n_better': better_than_baseline,
            'percentage_better': percentage_better,
            'status': 'improved' if avg_score > baseline_score else 'declined',
        }

        return comparison


# === Convenience function for creating all mock agents ===

def create_mock_agents(seed: Optional[int] = 42) -> Dict[str, Any]:
    """
    Create all mock agents for testing.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Dict with all mock agents
    """
    return {
        'explorer': MockExplorer(seed=seed),
        'architect': MockArchitect(seed=seed),
        'critic': MockCritic(),
        'historian': MockHistorian(),
    }


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("MOCK AGENTS DEMONSTRATION")
    print("=" * 70)
    print()

    # Create agents
    agents = create_mock_agents(seed=42)

    # Simulate research cycle
    print("1. Explorer generates hypothesis:")
    hypothesis = agents['explorer'].autonomous_cycle({'total_experiments': 0})
    print(f"   {hypothesis['description']}")
    print(f"   Parameter: {hypothesis['parameter']}")
    print()

    # Architect designs experiments
    print("2. Architect designs experiments:")
    baseline = {
        'length_overall': 18.0,
        'beam': 2.0,
        'hull_depth': 2.2,
        'hull_spacing': 5.4,
        'deadrise_angle': 12.0,
        'freeboard': 1.4,
        'lcb_position': 48.0,
        'prismatic_coefficient': 0.60,
        'waterline_beam': 1.8,
        'block_coefficient': 0.42,
        'design_speed': 25.0,
        'displacement': 35.0,
        'draft': 0.8,
    }

    designs = agents['architect'].design_experiments(hypothesis, baseline, n_designs=10)
    print(f"   Generated {len(designs)} designs")
    print(f"   First design {hypothesis['parameter']}: {designs[0][hypothesis['parameter']]:.2f}")
    print(f"   Last design {hypothesis['parameter']}: {designs[-1][hypothesis['parameter']]:.2f}")
    print()

    # Critic reviews
    print("3. Critic reviews experiments:")
    review = agents['critic'].review_experiments(designs, hypothesis)
    print(f"   Approved: {review['approved']}")
    print(f"   Issues: {review['n_issues']}")
    print()

    # Mock results (for demonstration)
    print("4. Historian analyzes results (mock):")
    mock_results = [
        {
            'is_valid': True,
            'overall_score': 70.0 + i,
            'stability_score': 75.0,
            'speed_score': 68.0,
            'efficiency_score': 65.0
        }
        for i in range(10)
    ]

    insights = agents['historian'].analyze_batch_results(designs, mock_results, hypothesis)
    print(f"   Outcome: {insights['hypothesis_outcome']}")
    print(f"   Avg Score: {insights['avg_overall_score']:.1f}/100")
    print(f"   Insights: {len(insights['insights'])}")
    for insight in insights['insights']:
        print(f"      - {insight}")

    print()
    print("=" * 70)
