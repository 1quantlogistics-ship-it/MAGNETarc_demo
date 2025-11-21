"""
Visualization Extensions for KnowledgeBase

Provides visualization methods for the Knowledge Base:
- 2D design space scatter plots
- Pareto frontier visualization
- Improvement over time plots
- Self-contained HTML dashboards

All plots use matplotlib with Agg backend (Mac-compatible, no display needed).
"""

import base64
from io import BytesIO
from typing import Optional
from pathlib import Path
from datetime import datetime


def visualize_design_space_2d(kb, param_x: str, param_y: str, output_path: Optional[str] = None) -> str:
    """
    Create 2D scatter plot of design space.

    Args:
        kb: KnowledgeBase instance
        param_x: Parameter name for X axis
        param_y: Parameter name for Y axis
        output_path: Optional path to save plot

    Returns:
        Path to saved plot
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    if not kb.best_designs:
        raise ValueError("No designs available for visualization")

    # Extract parameter values and scores
    x_values = []
    y_values = []
    scores = []

    for design_data in kb.best_designs:
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

    # Save
    if output_path is None:
        viz_dir = kb.storage_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        output_path = viz_dir / f'design_space_{param_x}_{param_y}.png'

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def visualize_pareto_frontier(kb, output_path: Optional[str] = None) -> str:
    """
    Create Pareto frontier visualization.

    Shows trade-offs between stability, speed, and efficiency.

    Args:
        kb: KnowledgeBase instance
        output_path: Optional path to save plot

    Returns:
        Path to saved plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not kb.best_designs:
        raise ValueError("No designs available for visualization")

    # Extract scores
    stability_scores = []
    speed_scores = []
    efficiency_scores = []
    overall_scores = []

    for design_data in kb.best_designs:
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

    # Save
    if output_path is None:
        viz_dir = kb.storage_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        output_path = viz_dir / 'pareto_frontier.png'

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def plot_improvement_over_time(kb, output_path: Optional[str] = None) -> str:
    """
    Plot score progression over research cycles.

    Args:
        kb: KnowledgeBase instance
        output_path: Optional path to save plot

    Returns:
        Path to saved plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not kb.experiments:
        raise ValueError("No experiments available for visualization")

    # Extract cycle data
    cycles = []
    avg_scores = []
    max_scores = []
    min_scores = []

    for exp in kb.experiments:
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

    # Save
    if output_path is None:
        viz_dir = kb.storage_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        output_path = viz_dir / 'improvement_over_time.png'

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def export_html_report(kb, output_path: Optional[str] = None) -> str:
    """
    Export self-contained HTML dashboard with embedded visualizations.

    Args:
        kb: KnowledgeBase instance
        output_path: Optional path to save HTML file

    Returns:
        Path to saved HTML file
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = kb.storage_path / "research_dashboard.html"

    # Generate visualizations to base64
    def fig_to_base64(fig):
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_base64

    # Create visualizations
    plots = {}

    # Improvement over time
    if kb.experiments:
        cycles = [exp['cycle'] for exp in kb.experiments]
        avg_scores = [exp.get('avg_score', 0.0) for exp in kb.experiments]
        max_scores = [exp.get('max_score', 0.0) for exp in kb.experiments]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cycles, avg_scores, marker='o', linewidth=2, label='Average')
        ax.plot(cycles, max_scores, marker='^', linewidth=1.5, label='Max', linestyle='--')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Score')
        ax.set_title('Performance Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plots['improvement'] = fig_to_base64(fig)

    # Pareto frontier
    if kb.best_designs:
        stability = [d['result']['stability_score'] for d in kb.best_designs[:50]]
        speed = [d['result']['speed_score'] for d in kb.best_designs[:50]]
        overall = [d['overall_score'] for d in kb.best_designs[:50]]

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
                <div class="stat-value">{kb.statistics['total_cycles']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Designs Evaluated</div>
                <div class="stat-value">{kb.statistics['total_designs_evaluated']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Best Score</div>
                <div class="stat-value">{kb.statistics['best_overall_score']:.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Score</div>
                <div class="stat-value">{kb.statistics['avg_overall_score']:.1f}</div>
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

    for i, design_data in enumerate(kb.best_designs[:10], 1):
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

        <h2>💡 Extracted Principles ({len(kb.principles)})</h2>
        <ul>
"""

    for principle in kb.principles[-10:]:
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

    # Save
    with open(output_path, 'w') as f:
        f.write(html)

    return str(output_path)
