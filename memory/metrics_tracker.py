"""
Performance Metrics Tracker for Autonomous Research System

Tracks system performance metrics during autonomous research cycles:
- Designs per second (throughput)
- Agent decision time (LLM latency)
- Physics simulation time
- Cycle completion time
- Hypothesis success rate

Provides real-time performance monitoring and optimization insights.
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import numpy as np


class MetricsTracker:
    """
    Track and analyze performance metrics for autonomous research system.

    Monitors:
    - Physics engine throughput
    - LLM agent latency
    - Cycle timing
    - Success rates
    - Resource utilization
    """

    def __init__(self, storage_path: str = "memory/metrics"):
        """
        Initialize metrics tracker.

        Args:
            storage_path: Directory for metrics storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.cycle_metrics: List[Dict[str, Any]] = []
        self.agent_metrics: Dict[str, List[float]] = {
            'explorer': [],
            'architect': [],
            'critic': [],
            'historian': [],
            'supervisor': []
        }
        self.physics_metrics: List[Dict[str, Any]] = []

        # Cumulative statistics
        self.total_designs_simulated = 0
        self.total_llm_calls = 0
        self.total_time_elapsed = 0.0
        self.session_start_time = time.time()

        # File paths
        self.metrics_file = self.storage_path / "metrics.json"
        self.summary_file = self.storage_path / "summary.json"

        # Load existing data
        self.load()

    def start_cycle(self, cycle_number: int) -> float:
        """
        Mark start of research cycle.

        Args:
            cycle_number: Current cycle number

        Returns:
            Start timestamp
        """
        start_time = time.time()
        self.current_cycle = {
            'cycle': cycle_number,
            'start_time': start_time,
            'start_datetime': datetime.now().isoformat(),
            'agent_times': {},
            'physics_time': 0.0,
            'total_time': 0.0,
            'designs_evaluated': 0,
        }
        return start_time

    def end_cycle(self, start_time: float, designs_evaluated: int) -> Dict[str, Any]:
        """
        Mark end of research cycle and calculate metrics.

        Args:
            start_time: Cycle start timestamp
            designs_evaluated: Number of designs evaluated this cycle

        Returns:
            Cycle metrics dict
        """
        end_time = time.time()
        cycle_duration = end_time - start_time

        self.current_cycle['total_time'] = cycle_duration
        self.current_cycle['designs_evaluated'] = designs_evaluated
        self.current_cycle['throughput'] = designs_evaluated / cycle_duration if cycle_duration > 0 else 0

        self.cycle_metrics.append(self.current_cycle.copy())
        self.total_designs_simulated += designs_evaluated
        self.total_time_elapsed += cycle_duration

        self.save()
        return self.current_cycle

    def record_agent_time(self, agent_name: str, duration: float) -> None:
        """
        Record time taken by an agent.

        Args:
            agent_name: Name of the agent (explorer, architect, etc.)
            duration: Time in seconds
        """
        if agent_name in self.agent_metrics:
            self.agent_metrics[agent_name].append(duration)
            self.total_llm_calls += 1

        if hasattr(self, 'current_cycle'):
            self.current_cycle['agent_times'][agent_name] = duration

    def record_physics_time(self, n_designs: int, duration: float, device: str = 'cpu') -> None:
        """
        Record physics simulation time.

        Args:
            n_designs: Number of designs simulated
            duration: Time in seconds
            device: Device used (cpu or cuda)
        """
        throughput = n_designs / duration if duration > 0 else 0

        metric = {
            'timestamp': datetime.now().isoformat(),
            'n_designs': n_designs,
            'duration': duration,
            'throughput': throughput,
            'device': device,
        }

        self.physics_metrics.append(metric)

        if hasattr(self, 'current_cycle'):
            self.current_cycle['physics_time'] = duration
            self.current_cycle['physics_throughput'] = throughput

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.

        Returns:
            Performance metrics dict
        """
        summary = {
            'session_duration': time.time() - self.session_start_time,
            'total_cycles': len(self.cycle_metrics),
            'total_designs_simulated': self.total_designs_simulated,
            'total_llm_calls': self.total_llm_calls,
            'total_time_elapsed': self.total_time_elapsed,
        }

        # Average cycle time
        if self.cycle_metrics:
            cycle_times = [c['total_time'] for c in self.cycle_metrics]
            summary['avg_cycle_time'] = float(np.mean(cycle_times))
            summary['min_cycle_time'] = float(np.min(cycle_times))
            summary['max_cycle_time'] = float(np.max(cycle_times))

        # Agent latency statistics
        summary['agent_latency'] = {}
        for agent_name, times in self.agent_metrics.items():
            if times:
                summary['agent_latency'][agent_name] = {
                    'avg': float(np.mean(times)),
                    'min': float(np.min(times)),
                    'max': float(np.max(times)),
                    'std': float(np.std(times)),
                    'calls': len(times),
                }

        # Physics throughput
        if self.physics_metrics:
            throughputs = [p['throughput'] for p in self.physics_metrics]
            summary['physics_throughput'] = {
                'avg': float(np.mean(throughputs)),
                'min': float(np.min(throughputs)),
                'max': float(np.max(throughputs)),
                'designs_per_second': float(np.mean(throughputs)),
            }

        # Overall system throughput
        if self.total_time_elapsed > 0:
            summary['overall_throughput'] = self.total_designs_simulated / self.total_time_elapsed

        return summary

    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """
        Get performance statistics for specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent performance dict
        """
        if agent_name not in self.agent_metrics or not self.agent_metrics[agent_name]:
            return {'error': f'No data for agent: {agent_name}'}

        times = self.agent_metrics[agent_name]

        return {
            'agent': agent_name,
            'total_calls': len(times),
            'avg_latency': float(np.mean(times)),
            'min_latency': float(np.min(times)),
            'max_latency': float(np.max(times)),
            'std_latency': float(np.std(times)),
            'p50_latency': float(np.percentile(times, 50)),
            'p95_latency': float(np.percentile(times, 95)),
            'p99_latency': float(np.percentile(times, 99)),
        }

    def get_cycle_performance(self, cycle_number: int) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for specific cycle.

        Args:
            cycle_number: Cycle number

        Returns:
            Cycle metrics dict or None if not found
        """
        for cycle in self.cycle_metrics:
            if cycle['cycle'] == cycle_number:
                return cycle
        return None

    def export_metrics_report(self, output_path: Optional[str] = None) -> str:
        """
        Export metrics as markdown report.

        Args:
            output_path: Optional path to save report

        Returns:
            Path to saved report
        """
        if output_path is None:
            output_path = self.storage_path / "metrics_report.md"

        summary = self.get_performance_summary()

        lines = [
            "# Performance Metrics Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            f"- **Session Duration:** {summary.get('session_duration', 0):.1f}s",
            f"- **Total Cycles:** {summary.get('total_cycles', 0)}",
            f"- **Total Designs Simulated:** {summary.get('total_designs_simulated', 0)}",
            f"- **Total LLM Calls:** {summary.get('total_llm_calls', 0)}",
            "",
            "## Cycle Performance",
            "",
            f"- **Average Cycle Time:** {summary.get('avg_cycle_time', 0):.2f}s",
            f"- **Min Cycle Time:** {summary.get('min_cycle_time', 0):.2f}s",
            f"- **Max Cycle Time:** {summary.get('max_cycle_time', 0):.2f}s",
            "",
            "## Agent Latency",
            "",
            "| Agent | Avg (s) | Min (s) | Max (s) | Calls |",
            "|-------|---------|---------|---------|-------|",
        ]

        for agent_name, stats in summary.get('agent_latency', {}).items():
            lines.append(
                f"| {agent_name.title()} | {stats['avg']:.3f} | {stats['min']:.3f} | "
                f"{stats['max']:.3f} | {stats['calls']} |"
            )

        lines.extend([
            "",
            "## Physics Engine Performance",
            "",
        ])

        physics = summary.get('physics_throughput', {})
        if physics:
            lines.extend([
                f"- **Average Throughput:** {physics.get('designs_per_second', 0):.1f} designs/sec",
                f"- **Min Throughput:** {physics.get('min', 0):.1f} designs/sec",
                f"- **Max Throughput:** {physics.get('max', 0):.1f} designs/sec",
            ])

        lines.extend([
            "",
            "## Overall System Throughput",
            "",
            f"- **Designs/Second:** {summary.get('overall_throughput', 0):.2f}",
            "",
        ])

        report = "\n".join(lines)

        with open(output_path, 'w') as f:
            f.write(report)

        return str(output_path)

    def save(self) -> None:
        """Save metrics to JSON files."""
        # Save detailed metrics
        metrics_data = {
            'cycles': self.cycle_metrics,
            'agents': {k: v for k, v in self.agent_metrics.items() if v},  # Only non-empty
            'physics': self.physics_metrics,
            'totals': {
                'designs_simulated': self.total_designs_simulated,
                'llm_calls': self.total_llm_calls,
                'time_elapsed': self.total_time_elapsed,
                'session_start': self.session_start_time,
            }
        }

        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        # Save summary
        summary = self.get_performance_summary()
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def load(self) -> None:
        """Load metrics from JSON files if they exist."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)

            self.cycle_metrics = data.get('cycles', [])
            self.agent_metrics = data.get('agents', {
                'explorer': [],
                'architect': [],
                'critic': [],
                'historian': [],
                'supervisor': []
            })
            self.physics_metrics = data.get('physics', [])

            totals = data.get('totals', {})
            self.total_designs_simulated = totals.get('designs_simulated', 0)
            self.total_llm_calls = totals.get('llm_calls', 0)
            self.total_time_elapsed = totals.get('time_elapsed', 0.0)
            self.session_start_time = totals.get('session_start', time.time())

    def clear(self) -> None:
        """Clear all metrics (useful for testing)."""
        self.cycle_metrics = []
        self.agent_metrics = {k: [] for k in self.agent_metrics.keys()}
        self.physics_metrics = []
        self.total_designs_simulated = 0
        self.total_llm_calls = 0
        self.total_time_elapsed = 0.0
        self.session_start_time = time.time()
        self.save()


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("METRICS TRACKER DEMONSTRATION")
    print("=" * 70)
    print()

    tracker = MetricsTracker(storage_path="memory/metrics_demo")

    # Simulate cycles
    for cycle in range(1, 4):
        print(f"Cycle {cycle}:")

        start = tracker.start_cycle(cycle)

        # Simulate agent calls
        tracker.record_agent_time('explorer', 0.5 + cycle * 0.1)
        tracker.record_agent_time('architect', 0.3 + cycle * 0.05)
        tracker.record_agent_time('critic', 0.2)

        # Simulate physics
        tracker.record_physics_time(n_designs=10 * cycle, duration=0.05 * cycle, device='cpu')

        # End cycle
        time.sleep(0.1)  # Small delay
        tracker.end_cycle(start, designs_evaluated=10 * cycle)

        print(f"  Duration: {tracker.current_cycle['total_time']:.2f}s")
        print(f"  Throughput: {tracker.current_cycle['throughput']:.1f} designs/sec")
        print()

    # Get summary
    summary = tracker.get_performance_summary()

    print("Performance Summary:")
    print(f"  Total Cycles: {summary['total_cycles']}")
    print(f"  Total Designs: {summary['total_designs_simulated']}")
    print(f"  Avg Cycle Time: {summary.get('avg_cycle_time', 0):.2f}s")
    print()

    # Agent performance
    print("Agent Latency:")
    for agent_name in ['explorer', 'architect', 'critic']:
        perf = tracker.get_agent_performance(agent_name)
        print(f"  {agent_name.title()}: {perf['avg_latency']:.3f}s avg ({perf['total_calls']} calls)")

    print()

    # Export report
    report_path = tracker.export_metrics_report()
    print(f"Metrics report saved to: {report_path}")
    print()

    print("=" * 70)
