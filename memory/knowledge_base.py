"""
Knowledge Base for Autonomous Naval Design Research

This module implements persistent storage and retrieval of design experiments,
results, and extracted principles. Supports the autonomous research cycle by
providing historical context and tracking learning progress.

Key Features:
- JSON-based persistence (no database required)
- Experiment history tracking
- Design principle extraction
- Pareto frontier maintenance
- Statistical analysis and reporting
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np


class KnowledgeBase:
    """
    Persistent knowledge storage for autonomous design research.

    Stores:
    - Experiment history (hypotheses, designs, results)
    - Extracted design principles
    - Best designs (Pareto frontier)
    - Research statistics

    All data persisted to JSON files for portability.
    """

    def __init__(self, storage_path: str = "memory/knowledge"):
        """
        Initialize knowledge base.

        Args:
            storage_path: Directory for JSON storage files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self.experiments: List[Dict[str, Any]] = []
        self.principles: List[Dict[str, Any]] = []
        self.best_designs: List[Dict[str, Any]] = []
        self.statistics: Dict[str, Any] = self._init_statistics()

        # File paths
        self.experiments_file = self.storage_path / "experiments.json"
        self.principles_file = self.storage_path / "principles.json"
        self.best_designs_file = self.storage_path / "best_designs.json"
        self.statistics_file = self.storage_path / "statistics.json"

        # Load existing data if available
        self.load()

    def _init_statistics(self) -> Dict[str, Any]:
        """Initialize statistics dictionary."""
        return {
            'total_experiments': 0,
            'total_designs_evaluated': 0,
            'total_cycles': 0,
            'avg_overall_score': 0.0,
            'avg_stability_score': 0.0,
            'avg_speed_score': 0.0,
            'avg_efficiency_score': 0.0,
            'best_overall_score': 0.0,
            'successful_hypotheses': 0,
            'failed_hypotheses': 0,
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }

    def add_experiment_results(
        self,
        hypothesis: Dict[str, Any],
        designs: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        cycle_number: int
    ) -> None:
        """
        Add experiment results to knowledge base.

        Args:
            hypothesis: Hypothesis that generated this experiment
            designs: List of design parameter dicts
            results: List of physics result dicts
            cycle_number: Current research cycle number
        """
        # Create experiment record
        experiment = {
            'cycle': cycle_number,
            'timestamp': datetime.now().isoformat(),
            'hypothesis': hypothesis,
            'n_designs': len(designs),
            'designs': designs,
            'results': results,
        }

        # Extract valid results
        valid_results = [r for r in results if r and r.get('is_valid', False)]

        if valid_results:
            # Calculate experiment statistics
            scores = [r['overall_score'] for r in valid_results]
            experiment['avg_score'] = float(np.mean(scores))
            experiment['max_score'] = float(np.max(scores))
            experiment['min_score'] = float(np.min(scores))
            experiment['std_score'] = float(np.std(scores))

            # Check if hypothesis was successful (improvement over baseline)
            baseline_score = 60.0  # Approximate baseline score
            if experiment['avg_score'] > baseline_score:
                experiment['hypothesis_outcome'] = 'confirmed'
                self.statistics['successful_hypotheses'] += 1
            else:
                experiment['hypothesis_outcome'] = 'refuted'
                self.statistics['failed_hypotheses'] += 1
        else:
            experiment['avg_score'] = 0.0
            experiment['hypothesis_outcome'] = 'failed'
            self.statistics['failed_hypotheses'] += 1

        # Add to experiments history
        self.experiments.append(experiment)

        # Update best designs (Pareto frontier)
        self._update_best_designs(designs, results)

        # Extract principles from this batch
        if valid_results:
            principles = self._extract_principles_from_batch(designs, valid_results)
            self.principles.extend(principles)

        # Update statistics
        self._update_statistics(results, cycle_number)

        # Auto-save
        self.save()

    def get_context_for_explorer(self, max_entries: int = 10) -> Dict[str, Any]:
        """
        Package knowledge base context for Explorer agent.

        Args:
            max_entries: Maximum number of recent experiments to include

        Returns:
            Dictionary with historical context for LLM
        """
        recent_experiments = self.experiments[-max_entries:] if self.experiments else []

        # Summarize recent experiments
        experiment_summaries = []
        for exp in recent_experiments:
            summary = {
                'cycle': exp['cycle'],
                'hypothesis': exp['hypothesis'],
                'n_designs': exp['n_designs'],
                'avg_score': exp.get('avg_score', 0.0),
                'outcome': exp.get('hypothesis_outcome', 'unknown'),
            }
            experiment_summaries.append(summary)

        # Top principles
        top_principles = self.principles[-20:] if self.principles else []

        # Best designs
        best_designs = self.best_designs[:10]  # Top 10

        context = {
            'recent_experiments': experiment_summaries,
            'extracted_principles': top_principles,
            'best_designs': best_designs,
            'statistics': self.statistics,
            'total_experiments': len(self.experiments),
        }

        return context

    def get_best_designs(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N designs by overall score.

        Args:
            n: Number of designs to return

        Returns:
            List of best design dicts with results
        """
        return self.best_designs[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current research statistics.

        Returns:
            Dictionary of statistics
        """
        return self.statistics.copy()

    def export_markdown_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate human-readable Markdown report.

        Args:
            output_path: Optional path to save report file

        Returns:
            Markdown report string
        """
        if output_path is None:
            output_path = self.storage_path / "research_report.md"

        lines = [
            "# Autonomous Naval Design Research Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            f"- **Total Research Cycles:** {self.statistics['total_cycles']}",
            f"- **Total Experiments:** {self.statistics['total_experiments']}",
            f"- **Total Designs Evaluated:** {self.statistics['total_designs_evaluated']}",
            f"- **Started:** {self.statistics['started_at']}",
            "",
            "## Performance Summary",
            "",
            f"- **Average Overall Score:** {self.statistics['avg_overall_score']:.2f}/100",
            f"- **Best Overall Score:** {self.statistics['best_overall_score']:.2f}/100",
            f"- **Average Stability:** {self.statistics['avg_stability_score']:.2f}/100",
            f"- **Average Speed:** {self.statistics['avg_speed_score']:.2f}/100",
            f"- **Average Efficiency:** {self.statistics['avg_efficiency_score']:.2f}/100",
            "",
            "## Hypothesis Success Rate",
            "",
            f"- **Confirmed Hypotheses:** {self.statistics['successful_hypotheses']}",
            f"- **Refuted Hypotheses:** {self.statistics['failed_hypotheses']}",
        ]

        if self.statistics['successful_hypotheses'] + self.statistics['failed_hypotheses'] > 0:
            success_rate = (self.statistics['successful_hypotheses'] /
                          (self.statistics['successful_hypotheses'] + self.statistics['failed_hypotheses']) * 100)
            lines.append(f"- **Success Rate:** {success_rate:.1f}%")

        lines.extend([
            "",
            "## Top 10 Designs",
            "",
            "| Rank | LOA (m) | Spacing (m) | Speed (kts) | Overall Score |",
            "|------|---------|-------------|-------------|---------------|",
        ])

        for i, design_data in enumerate(self.best_designs[:10], 1):
            design = design_data['design']
            result = design_data['result']
            lines.append(
                f"| {i} | {design['length_overall']:.1f} | "
                f"{design['hull_spacing']:.1f} | {design['design_speed']:.1f} | "
                f"{result['overall_score']:.1f} |"
            )

        lines.extend([
            "",
            "## Extracted Design Principles",
            "",
        ])

        for i, principle in enumerate(self.principles[-10:], 1):
            lines.append(f"### Principle {i}")
            lines.append(f"- **Parameter:** {principle.get('parameter', 'Unknown')}")
            lines.append(f"- **Correlation:** {principle.get('correlation', 0.0):.3f}")
            lines.append(f"- **Insight:** {principle.get('insight', 'No insight available')}")
            lines.append("")

        report = "\n".join(lines)

        # Save to file
        with open(output_path, 'w') as f:
            f.write(report)

        return report

    def _update_best_designs(
        self,
        designs: List[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ) -> None:
        """
        Update Pareto frontier of best designs.

        Args:
            designs: New designs
            results: New results
        """
        # Add new designs to pool
        for design, result in zip(designs, results):
            if result and result.get('is_valid', False):
                self.best_designs.append({
                    'design': design,
                    'result': result,
                    'overall_score': result['overall_score'],
                })

        # Sort by overall score (descending)
        self.best_designs.sort(key=lambda x: x['overall_score'], reverse=True)

        # Keep top 100
        self.best_designs = self.best_designs[:100]

    def _extract_principles_from_batch(
        self,
        designs: List[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract design principles from batch using correlation analysis.

        Args:
            designs: Design parameter dicts
            results: Physics result dicts

        Returns:
            List of extracted principle dicts
        """
        if len(designs) < 3:
            return []  # Need minimum 3 samples for correlation

        principles = []

        # Extract parameter arrays
        param_names = [
            'length_overall', 'beam', 'hull_spacing', 'deadrise_angle',
            'prismatic_coefficient', 'block_coefficient', 'design_speed'
        ]

        scores = np.array([r['overall_score'] for r in results])

        for param_name in param_names:
            try:
                param_values = np.array([d[param_name] for d in designs])

                # Calculate correlation with overall score
                correlation = np.corrcoef(param_values, scores)[0, 1]

                # Extract principle if correlation is significant
                if abs(correlation) > 0.5:  # Moderate to strong correlation
                    insight = self._generate_insight(param_name, correlation, param_values, scores)

                    principle = {
                        'parameter': param_name,
                        'correlation': float(correlation),
                        'insight': insight,
                        'timestamp': datetime.now().isoformat(),
                        'n_samples': len(designs),
                    }

                    principles.append(principle)
            except (KeyError, ValueError):
                # Skip if parameter missing or invalid
                continue

        return principles

    def _generate_insight(
        self,
        param_name: str,
        correlation: float,
        param_values: np.ndarray,
        scores: np.ndarray
    ) -> str:
        """
        Generate human-readable insight from correlation.

        Args:
            param_name: Parameter name
            correlation: Correlation coefficient
            param_values: Parameter value array
            scores: Score array

        Returns:
            Insight string
        """
        if correlation > 0.5:
            direction = "increases"
            effect = "improves"
        elif correlation < -0.5:
            direction = "decreases"
            effect = "reduces"
        else:
            return f"{param_name} has weak correlation with performance"

        mean_param = float(np.mean(param_values))
        optimal_idx = int(np.argmax(scores))
        optimal_value = float(param_values[optimal_idx])

        return (
            f"As {param_name} {direction}, overall score {effect} "
            f"(corr={correlation:.2f}). Optimal value appears near {optimal_value:.2f} "
            f"(mean={mean_param:.2f})"
        )

    def _update_statistics(self, results: List[Dict[str, Any]], cycle_number: int) -> None:
        """
        Update running statistics.

        Args:
            results: New physics results
            cycle_number: Current cycle number
        """
        valid_results = [r for r in results if r and r.get('is_valid', False)]

        if not valid_results:
            return

        # Update counts
        self.statistics['total_experiments'] += 1
        self.statistics['total_designs_evaluated'] += len(results)
        self.statistics['total_cycles'] = cycle_number

        # Extract scores
        overall_scores = [r['overall_score'] for r in valid_results]
        stability_scores = [r['stability_score'] for r in valid_results]
        speed_scores = [r['speed_score'] for r in valid_results]
        efficiency_scores = [r['efficiency_score'] for r in valid_results]

        # Update averages (running average)
        n_total = self.statistics['total_designs_evaluated']
        n_new = len(valid_results)

        # Weighted average update
        old_weight = (n_total - n_new) / n_total if n_total > 0 else 0
        new_weight = n_new / n_total if n_total > 0 else 1

        self.statistics['avg_overall_score'] = (
            old_weight * self.statistics['avg_overall_score'] +
            new_weight * np.mean(overall_scores)
        )
        self.statistics['avg_stability_score'] = (
            old_weight * self.statistics['avg_stability_score'] +
            new_weight * np.mean(stability_scores)
        )
        self.statistics['avg_speed_score'] = (
            old_weight * self.statistics['avg_speed_score'] +
            new_weight * np.mean(speed_scores)
        )
        self.statistics['avg_efficiency_score'] = (
            old_weight * self.statistics['avg_efficiency_score'] +
            new_weight * np.mean(efficiency_scores)
        )

        # Update best score
        max_score = float(np.max(overall_scores))
        if max_score > self.statistics['best_overall_score']:
            self.statistics['best_overall_score'] = max_score

        # Update timestamp
        self.statistics['last_updated'] = datetime.now().isoformat()

    def save(self) -> None:
        """Save knowledge base to JSON files."""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

        with open(self.principles_file, 'w') as f:
            json.dump(self.principles, f, indent=2)

        with open(self.best_designs_file, 'w') as f:
            json.dump(self.best_designs, f, indent=2)

        with open(self.statistics_file, 'w') as f:
            json.dump(self.statistics, f, indent=2)

    def load(self) -> None:
        """Load knowledge base from JSON files if they exist."""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                self.experiments = json.load(f)

        if self.principles_file.exists():
            with open(self.principles_file, 'r') as f:
                self.principles = json.load(f)

        if self.best_designs_file.exists():
            with open(self.best_designs_file, 'r') as f:
                self.best_designs = json.load(f)

        if self.statistics_file.exists():
            with open(self.statistics_file, 'r') as f:
                self.statistics = json.load(f)

    def clear(self) -> None:
        """Clear all knowledge (useful for testing)."""
        self.experiments = []
        self.principles = []
        self.best_designs = []
        self.statistics = self._init_statistics()
        self.save()


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("KNOWLEDGE BASE DEMONSTRATION")
    print("=" * 70)
    print()

    # Add parent directory to path
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # Initialize knowledge base
    kb = KnowledgeBase(storage_path="memory/knowledge_demo")

    # Simulate experiment results
    from naval_domain.hull_parameters import get_baseline_catamaran
    from naval_domain.physics_engine import simulate_design
    import random

    baseline = get_baseline_catamaran()
    baseline_dict = baseline.to_dict()

    # Create mock experiment
    hypothesis = {
        'description': 'Increasing hull spacing improves stability',
        'parameter_ranges': {
            'hull_spacing': (5.0, 7.0),
        }
    }

    designs = []
    results = []

    for i in range(10):
        design = baseline_dict.copy()
        design['hull_spacing'] = 5.0 + i * 0.2
        designs.append(design)

        # Simulate
        from naval_domain.hull_parameters import HullParameters
        hp = HullParameters(**design)
        result = simulate_design(hp)
        results.append(result.to_dict())

    # Add to knowledge base
    kb.add_experiment_results(hypothesis, designs, results, cycle_number=1)

    print("Added experiment to knowledge base")
    print()

    # Get statistics
    stats = kb.get_statistics()
    print("Statistics:")
    print(f"  Total Experiments: {stats['total_experiments']}")
    print(f"  Total Designs: {stats['total_designs_evaluated']}")
    print(f"  Average Score: {stats['avg_overall_score']:.2f}/100")
    print(f"  Best Score: {stats['best_overall_score']:.2f}/100")
    print()

    # Get context for Explorer
    context = kb.get_context_for_explorer(max_entries=5)
    print(f"Context for Explorer: {len(context['recent_experiments'])} experiments")
    print(f"Extracted Principles: {len(context['extracted_principles'])}")
    print()

    # Export report
    report_path = kb.storage_path / "demo_report.md"
    kb.export_markdown_report(report_path)
    print(f"Exported report to: {report_path}")
    print()

    print("=" * 70)
    print("Knowledge base saved to:", kb.storage_path)
