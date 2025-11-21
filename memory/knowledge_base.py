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

    # ========== Visualization Methods ==========

    def export_html_report(self, output_path: str) -> str:
        """
        Generate comprehensive HTML dashboard with embedded visualizations.

        Creates a self-contained HTML file with:
        - Research statistics summary
        - Performance over time charts
        - Pareto frontier visualization
        - Top 10 designs table
        - Extracted principles list

        Args:
            output_path: Path where HTML file should be saved

        Returns:
            Path to generated HTML file

        Example:
            >>> kb = KnowledgeBase()
            >>> # ... add experiments ...
            >>> kb.export_html_report("results/dashboard.html")
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        # Helper function to convert figure to base64
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(fig)
            return img_base64

        # Generate visualizations
        plots = {}

        # Improvement over time plot
        if self.experiments:
            cycles = [exp['cycle'] for exp in self.experiments]
            avg_scores = [exp.get('avg_score', 0.0) for exp in self.experiments]
            max_scores = [exp.get('max_score', 0.0) for exp in self.experiments]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(cycles, avg_scores, marker='o', linewidth=2, label='Average')
            ax.plot(cycles, max_scores, marker='^', linewidth=1.5, label='Max', linestyle='--')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Score')
            ax.set_title('Performance Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plots['improvement'] = fig_to_base64(fig)

        # Pareto frontier plot
        if self.best_designs:
            stability = [d['result']['stability_score'] for d in self.best_designs[:50]]
            speed = [d['result']['speed_score'] for d in self.best_designs[:50]]
            overall = [d['overall_score'] for d in self.best_designs[:50]]

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(stability, speed, c=overall, cmap='viridis',
                               s=100, alpha=0.6, edgecolors='black')
            ax.set_xlabel('Stability Score')
            ax.set_ylabel('Speed Score')
            ax.set_title('Pareto Frontier: Stability vs Speed')
            plt.colorbar(scatter, ax=ax, label='Overall Score')
            ax.grid(True, alpha=0.3)
            plots['pareto'] = fig_to_base64(fig)

        # Build HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAGNET Research Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #764ba2;
            margin-top: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌊 MAGNET Autonomous Naval Design Research Dashboard</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>📊 Research Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Cycles</div>
                <div class="stat-value">{self.statistics['total_cycles']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Designs Evaluated</div>
                <div class="stat-value">{self.statistics['total_designs_evaluated']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Best Score</div>
                <div class="stat-value">{self.statistics['best_overall_score']:.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Score</div>
                <div class="stat-value">{self.statistics['avg_overall_score']:.1f}</div>
            </div>
        </div>

        <h2>📈 Performance Over Time</h2>
        {f'<div class="plot"><img src="data:image/png;base64,{plots["improvement"]}" /></div>' if 'improvement' in plots else '<p>No data available</p>'}

        <h2>🎯 Pareto Frontier</h2>
        {f'<div class="plot"><img src="data:image/png;base64,{plots["pareto"]}" /></div>' if 'pareto' in plots else '<p>No data available</p>'}

        <h2>🏆 Top 10 Designs</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>LOA (m)</th>
                <th>Spacing (m)</th>
                <th>Speed (kts)</th>
                <th>Overall Score</th>
            </tr>
"""

        for i, design_data in enumerate(self.best_designs[:10], 1):
            design = design_data['design']
            result = design_data['result']
            html += f"""
            <tr>
                <td>{i}</td>
                <td>{design['length_overall']:.1f}</td>
                <td>{design['hull_spacing']:.1f}</td>
                <td>{design['design_speed']:.1f}</td>
                <td>{result['overall_score']:.1f}</td>
            </tr>
"""

        html += f"""
        </table>

        <h2>💡 Extracted Principles ({len(self.principles)})</h2>
        <ul>
"""

        for principle in self.principles[-10:]:
            html += f"""
            <li>
                <strong>{principle.get('parameter', 'Unknown')}:</strong>
                {principle.get('insight', 'No insight')}
                (correlation: {principle.get('correlation', 0.0):.3f})
            </li>
"""

        html += """
        </ul>

        <div class="footer">
            <p>MAGNET Autonomous Research System | Generated by KnowledgeBase</p>
        </div>
    </div>
</body>
</html>
"""

        # Save HTML file
        with open(output_path, 'w') as f:
            f.write(html)

        return output_path

    def plot_improvement_over_time(self, output_path: str) -> str:
        """
        Generate improvement over time plot showing learning progression.

        Creates a line chart showing average, max, and min scores across
        research cycles, with shaded region between min/max.

        Args:
            output_path: Path where plot image should be saved

        Returns:
            Path to generated plot file

        Raises:
            ValueError: If no experiments available for visualization

        Example:
            >>> kb = KnowledgeBase()
            >>> # ... run cycles ...
            >>> kb.plot_improvement_over_time("results/improvement.png")
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not self.experiments:
            raise ValueError("No experiments available for visualization")

        # Extract cycle data
        cycles = []
        avg_scores = []
        max_scores = []
        min_scores = []

        for exp in self.experiments:
            cycles.append(exp['cycle'])
            avg_scores.append(exp.get('avg_score', 0.0))
            max_scores.append(exp.get('max_score', 0.0))
            min_scores.append(exp.get('min_score', 0.0))

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot lines
        ax.plot(cycles, avg_scores, marker='o', linewidth=2, label='Average Score',
               color='blue', markersize=6)
        ax.plot(cycles, max_scores, marker='^', linewidth=1.5, label='Max Score',
               color='green', markersize=6, linestyle='--')
        ax.plot(cycles, min_scores, marker='v', linewidth=1.5, label='Min Score',
               color='red', markersize=6, linestyle='--')

        # Fill between min and max
        ax.fill_between(cycles, min_scores, max_scores, alpha=0.2, color='blue')

        ax.set_xlabel('Research Cycle', fontsize=12)
        ax.set_ylabel('Overall Score', fontsize=12)
        ax.set_title('Design Performance Improvement Over Time', fontsize=14)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Baseline reference line
        ax.axhline(y=60.0, color='gray', linestyle=':', linewidth=2, label='Baseline')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def visualize_design_space_2d(
        self,
        output_path: str,
        param_x: str = 'length_overall',
        param_y: str = 'hull_spacing'
    ) -> str:
        """
        Generate 2D design space scatter plot.

        Creates a scatter plot showing the relationship between two design
        parameters, color-coded by overall score.

        Args:
            output_path: Path where plot should be saved
            param_x: X-axis parameter name (default: 'length_overall')
            param_y: Y-axis parameter name (default: 'hull_spacing')

        Returns:
            Path to generated plot file

        Raises:
            ValueError: If no designs available or parameters not found

        Example:
            >>> kb = KnowledgeBase()
            >>> # ... add experiments ...
            >>> kb.visualize_design_space_2d("results/design_space.png",
            ...                               param_x='length_overall',
            ...                               param_y='beam')
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not self.best_designs:
            raise ValueError("No designs available for visualization")

        # Extract parameter values and scores
        x_values = []
        y_values = []
        scores = []

        for design_data in self.best_designs:
            design = design_data['design']
            if param_x in design and param_y in design:
                x_values.append(design[param_x])
                y_values.append(design[param_y])
                scores.append(design_data['overall_score'])

        if not x_values:
            raise ValueError(f"Parameters {param_x} and {param_y} not found in designs")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot with color map
        scatter = ax.scatter(x_values, y_values, c=scores, cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black')

        ax.set_xlabel(param_x.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(param_y.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Design Space: {param_x} vs {param_y}', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Overall Score', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def visualize_pareto_frontier(self, output_path: str) -> str:
        """
        Generate Pareto frontier visualization showing multi-objective trade-offs.

        Creates a 3-panel plot showing trade-offs between:
        - Stability vs Speed
        - Stability vs Efficiency
        - Speed vs Efficiency

        Args:
            output_path: Path where plot should be saved

        Returns:
            Path to generated plot file

        Raises:
            ValueError: If no designs available for visualization

        Example:
            >>> kb = KnowledgeBase()
            >>> # ... add experiments ...
            >>> kb.visualize_pareto_frontier("results/pareto.png")
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not self.best_designs:
            raise ValueError("No designs available for visualization")

        # Extract scores
        stability_scores = []
        speed_scores = []
        efficiency_scores = []
        overall_scores = []

        for design_data in self.best_designs:
            result = design_data['result']
            stability_scores.append(result['stability_score'])
            speed_scores.append(result['speed_score'])
            efficiency_scores.append(result['efficiency_score'])
            overall_scores.append(result['overall_score'])

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Stability vs Speed
        axes[0].scatter(stability_scores, speed_scores, c=overall_scores,
                       cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        axes[0].set_xlabel('Stability Score', fontsize=11)
        axes[0].set_ylabel('Speed Score', fontsize=11)
        axes[0].set_title('Stability vs Speed', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Stability vs Efficiency
        axes[1].scatter(stability_scores, efficiency_scores, c=overall_scores,
                       cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        axes[1].set_xlabel('Stability Score', fontsize=11)
        axes[1].set_ylabel('Efficiency Score', fontsize=11)
        axes[1].set_title('Stability vs Efficiency', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # Speed vs Efficiency
        scatter = axes[2].scatter(speed_scores, efficiency_scores, c=overall_scores,
                                 cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        axes[2].set_xlabel('Speed Score', fontsize=11)
        axes[2].set_ylabel('Efficiency Score', fontsize=11)
        axes[2].set_title('Speed vs Efficiency', fontsize=12)
        axes[2].grid(True, alpha=0.3)

        # Colorbar
        fig.colorbar(scatter, ax=axes, label='Overall Score')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path


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
