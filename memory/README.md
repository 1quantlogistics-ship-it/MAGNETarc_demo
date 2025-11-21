# MAGNET Memory & Knowledge Systems

Persistent knowledge management and performance tracking for autonomous naval design research.

## Overview

The `memory` module provides two core components:

1. **KnowledgeBase** - Stores experiments, extracts principles, tracks best designs
2. **MetricsTracker** - Monitors system performance and agent latency

Both components use JSON-based persistence for portability and simplicity.

---

## KnowledgeBase

### Purpose

The KnowledgeBase is the system's long-term memory, storing all experimental data and learning from patterns across research cycles.

### Core Functionality

- **Experiment Storage**: Persistent record of all hypotheses, designs, and results
- **Principle Extraction**: Automatic mining of design principles via correlation analysis
- **Pareto Frontier**: Maintains top 100 designs across all objectives
- **Progress Tracking**: Statistics on research effectiveness and learning
- **Visualization**: Charts, plots, and HTML dashboards

### API Reference

#### Initialization

```python
from memory import KnowledgeBase

# Create knowledge base
kb = KnowledgeBase(storage_path="memory/knowledge")
```

**Parameters:**
- `storage_path` (str): Directory for JSON storage files (default: "memory/knowledge")

**Storage Files:**
- `experiments.json` - All experiment records
- `principles.json` - Extracted design principles
- `best_designs.json` - Pareto frontier (top 100)
- `statistics.json` - Cumulative statistics

---

#### Core Methods

##### `add_experiment_results()`

Store results from a research cycle.

```python
kb.add_experiment_results(
    hypothesis={
        'description': 'Hull spacing affects stability',
        'parameter_range': {'hull_spacing': (5.0, 7.0)}
    },
    designs=[design1_dict, design2_dict, ...],
    results=[result1_dict, result2_dict, ...],
    cycle_number=1
)
```

**Parameters:**
- `hypothesis` (dict): Hypothesis that generated this experiment
- `designs` (list[dict]): Design parameter dictionaries
- `results` (list[dict]): Physics simulation results
- `cycle_number` (int): Current research cycle

**Side Effects:**
- Stores experiment in history
- Updates best designs (Pareto frontier)
- Extracts principles if correlations found
- Updates statistics
- Auto-saves to disk

---

##### `get_context_for_explorer()`

Package historical context for the Explorer agent to generate new hypotheses.

```python
context = kb.get_context_for_explorer(max_entries=10)

# Returns:
# {
#     'recent_experiments': [...],  # Last N experiments
#     'extracted_principles': [...], # Top design principles
#     'best_designs': [...],         # Top 10 designs
#     'statistics': {...},           # Overall stats
#     'total_experiments': 42
# }
```

**Parameters:**
- `max_entries` (int): Maximum recent experiments to include (default: 10)

**Returns:**
- dict: Structured context for LLM agent

---

##### `get_best_designs()`

Retrieve top-performing designs.

```python
best = kb.get_best_designs(n=10)

# Each entry:
# {
#     'design': {...},         # Hull parameters
#     'result': {...},         # Physics results
#     'overall_score': 85.3
# }
```

**Parameters:**
- `n` (int): Number of designs to return (default: 10)

**Returns:**
- list[dict]: Best designs sorted by overall_score

---

##### `export_markdown_report()`

Generate human-readable research report.

```python
report = kb.export_markdown_report("results/report.md")
```

**Parameters:**
- `output_path` (str, optional): Path to save report

**Returns:**
- str: Path to saved report

**Report Contents:**
- Overview statistics
- Performance summary
- Hypothesis success rate
- Top 10 designs table
- Extracted principles

---

### Visualization Methods

#### `export_html_report()`

Generate comprehensive interactive HTML dashboard.

```python
kb.export_html_report("results/dashboard.html")
```

**Features:**
- Self-contained (no external dependencies)
- Embedded base64-encoded charts
- Responsive design
- Beautiful gradient UI
- Statistics cards
- Top designs table
- Principles list

**Output Size:** ~500KB-2MB depending on data

---

#### `plot_improvement_over_time()`

Learning progression chart showing score evolution.

```python
kb.plot_improvement_over_time("results/improvement.png")
```

**Chart Shows:**
- Average score per cycle (blue line)
- Max score per cycle (green dashed)
- Min score per cycle (red dashed)
- Shaded region (min to max)
- Baseline reference line (60.0)

**Resolution:** 1800x900 px at 150 DPI

---

#### `visualize_design_space_2d()`

2D scatter plot of design parameter relationships.

```python
kb.visualize_design_space_2d(
    "results/design_space.png",
    param_x='length_overall',
    param_y='hull_spacing'
)
```

**Parameters:**
- `output_path` (str): Path to save plot
- `param_x` (str): X-axis parameter (default: 'length_overall')
- `param_y` (str): Y-axis parameter (default: 'hull_spacing')

**Features:**
- Color-coded by overall score (viridis colormap)
- Automatic axis labels
- Colorbar legend
- Grid for readability

---

#### `visualize_pareto_frontier()`

Multi-objective trade-off visualization.

```python
kb.visualize_pareto_frontier("results/pareto.png")
```

**Chart Shows:**
- 3 panels side-by-side:
  1. Stability vs Speed
  2. Stability vs Efficiency
  3. Speed vs Efficiency
- Color-coded by overall score
- Reveals design compromises

**Resolution:** 2250x750 px at 150 DPI

---

### Usage Example

```python
from memory import KnowledgeBase
from naval_domain.physics_engine import PhysicsEngine
from naval_domain.hull_parameters import HullParameters
from naval_domain.baseline_designs import get_baseline_general

# Initialize
kb = KnowledgeBase()
physics = PhysicsEngine()
baseline = get_baseline_general()

# Simulate research cycle
hypothesis = {
    'description': 'Wider hull spacing improves stability',
    'parameter_range': {'hull_spacing': (5.0, 8.0)}
}

designs = []
results = []

for spacing in [5.0, 6.0, 7.0, 8.0]:
    design = baseline.copy()
    design.pop('name', None)
    design.pop('description', None)
    design['hull_spacing'] = spacing
    designs.append(design)

    hp = HullParameters(**design)
    result = physics.simulate(hp)
    results.append(result.to_dict())

# Store results
kb.add_experiment_results(hypothesis, designs, results, cycle_number=1)

# Generate visualizations
kb.export_html_report("dashboard.html")
kb.plot_improvement_over_time("improvement.png")
kb.visualize_pareto_frontier("pareto.png")

# Get statistics
stats = kb.get_statistics()
print(f"Best score: {stats['best_overall_score']:.1f}")
print(f"Avg score: {stats['avg_overall_score']:.1f}")
```

---

## MetricsTracker

### Purpose

Monitors system performance in real-time, tracking throughput, latency, and success rates.

### Core Functionality

- **Cycle Timing**: Tracks duration of each research cycle
- **Agent Latency**: Records decision time for each agent
- **Physics Throughput**: Measures designs/sec simulation rate
- **Success Rates**: Calculates hypothesis confirmation rates
- **Report Generation**: Markdown performance reports

### API Reference

#### Initialization

```python
from memory import MetricsTracker

tracker = MetricsTracker(storage_path="memory/metrics")
```

**Parameters:**
- `storage_path` (str): Directory for metrics storage (default: "memory/metrics")

**Storage Files:**
- `metrics.json` - Detailed metrics history
- `summary.json` - Performance summary

---

#### Core Methods

##### `start_cycle()` / `end_cycle()`

Track cycle timing.

```python
# Start
start_time = tracker.start_cycle(cycle_number=1)

# ... research cycle executes ...

# End
tracker.end_cycle(start_time, designs_evaluated=50)
```

**Returns:**
- `start_cycle`: float (timestamp)
- `end_cycle`: dict (cycle metrics)

---

##### `record_agent_time()`

Record agent execution time.

```python
tracker.record_agent_time('explorer', duration=0.523)
tracker.record_agent_time('architect', duration=1.234)
tracker.record_agent_time('critic', duration=0.321)
```

**Parameters:**
- `agent_name` (str): Agent identifier
- `duration` (float): Time in seconds

**Tracked Agents:**
- `explorer`, `architect`, `critic`, `historian`, `supervisor`

---

##### `record_physics_time()`

Record physics simulation performance.

```python
tracker.record_physics_time(
    n_designs=50,
    duration=0.156,
    device='cpu'
)
```

**Parameters:**
- `n_designs` (int): Number of designs simulated
- `duration` (float): Time in seconds
- `device` (str): 'cpu' or 'cuda'

**Calculates:**
- Throughput (designs/sec)
- Per-design latency

---

##### `get_performance_summary()`

Get comprehensive performance metrics.

```python
summary = tracker.get_performance_summary()

# Returns:
# {
#     'total_cycles': 5,
#     'total_designs_simulated': 250,
#     'avg_cycle_time': 2.34,
#     'agent_latency': {
#         'explorer': {'avg': 0.52, 'min': 0.48, 'max': 0.58, 'calls': 5},
#         'architect': {'avg': 1.23, ...},
#         ...
#     },
#     'physics_throughput': {
#         'avg': 320.5,  # designs/sec
#         'min': 280.2,
#         'max': 350.8
#     },
#     'overall_throughput': 106.8  # designs/sec including all overhead
# }
```

---

##### `export_metrics_report()`

Generate Markdown performance report.

```python
tracker.export_metrics_report("results/metrics.md")
```

**Report Contents:**
- Session overview
- Cycle performance statistics
- Agent latency table
- Physics engine throughput
- Overall system throughput

---

### Usage Example

```python
from memory import MetricsTracker
import time

tracker = MetricsTracker()

# Simulate research cycle
start = tracker.start_cycle(1)

# Mock agent executions
tracker.record_agent_time('explorer', 0.5)
time.sleep(0.5)

tracker.record_agent_time('architect', 1.2)
time.sleep(1.2)

tracker.record_agent_time('critic', 0.3)
time.sleep(0.3)

# Mock physics simulation
tracker.record_physics_time(n_designs=50, duration=0.2, device='cpu')
time.sleep(0.2)

# End cycle
tracker.end_cycle(start, designs_evaluated=50)

# Get summary
summary = tracker.get_performance_summary()
print(f"Cycle time: {summary['avg_cycle_time']:.2f}s")
print(f"Throughput: {summary['overall_throughput']:.1f} designs/sec")

# Export report
tracker.export_metrics_report("metrics_report.md")
```

---

## Data Persistence

Both components automatically save to disk after updates.

### Storage Format

All data stored as human-readable JSON for:
- Easy inspection
- Version control compatibility
- No database dependencies
- Cross-platform portability

### Manual Save/Load

```python
# Explicit save (normally automatic)
kb.save()
tracker.save()

# Explicit load (normally automatic on init)
kb.load()
tracker.load()
```

---

## Performance Characteristics

### KnowledgeBase

- **Experiment storage**: O(1) append
- **Best designs update**: O(n log n) sort, limited to top 100
- **Principle extraction**: O(n²) correlation analysis
- **Context generation**: O(1) with max_entries limit

**Memory Usage:**
- ~1KB per experiment
- ~500 bytes per principle
- ~2KB per best design entry

**Typical Storage:**
- 100 experiments: ~500KB
- 1000 experiments: ~5MB

### MetricsTracker

- **Metric recording**: O(1) append
- **Summary generation**: O(n) over all metrics

**Memory Usage:**
- ~200 bytes per cycle metric
- ~50 bytes per agent timing

**Typical Storage:**
- 100 cycles: ~50KB
- 1000 cycles: ~500KB

---

## Integration with Orchestrator

```python
from memory import KnowledgeBase, MetricsTracker

class AutonomousOrchestrator:
    def __init__(self, config):
        self.kb = KnowledgeBase(config['knowledge_path'])
        self.metrics = MetricsTracker(config['metrics_path'])

    async def research_cycle(self, cycle_num):
        # Start timing
        start = self.metrics.start_cycle(cycle_num)

        # Get context from knowledge base
        context = self.kb.get_context_for_explorer()

        # Run agents (record timing)
        hypothesis = await self.explorer.execute(context)
        self.metrics.record_agent_time('explorer', explorer_time)

        designs = await self.architect.execute(hypothesis)
        self.metrics.record_agent_time('architect', architect_time)

        # Physics simulation
        results = self.physics.simulate_batch(designs)
        self.metrics.record_physics_time(len(designs), physics_time)

        # Store results
        self.kb.add_experiment_results(hypothesis, designs, results, cycle_num)

        # End timing
        self.metrics.end_cycle(start, len(designs))

        # Periodic visualization
        if cycle_num % 10 == 0:
            self.kb.export_html_report(f"results/cycle_{cycle_num}.html")
            self.metrics.export_metrics_report(f"results/metrics_{cycle_num}.md")
```

---

## Testing

### Unit Tests

```bash
pytest tests/integration/test_knowledge_base.py -v
```

### System Validation

```bash
pytest tests/integration/test_system_validation.py -v
```

---

## Mac Development Notes

All visualization methods use matplotlib Agg backend:
- No display server required
- Works on headless Mac
- Safe for automated testing

All components are CPU-compatible:
- No GPU dependencies in memory module
- Works on Mac M1/M2/M3 chips
- PyTorch optional (only for physics batching)

---

## Version History

### v0.1.0 (Current)
- Initial release
- KnowledgeBase with 4 visualization methods
- MetricsTracker with comprehensive performance monitoring
- JSON persistence
- Full Mac compatibility

---

## See Also

- [Naval Domain Documentation](../naval_domain/README.md)
- [Orchestration Guide](../orchestration/README.md)
- [Agent System](../agents/README.md)
