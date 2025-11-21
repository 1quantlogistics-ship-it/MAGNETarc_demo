# MAGNET CLI User Guide

Complete guide to using the MAGNET command-line interface.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Command-Line Options](#command-line-options)
- [Common Workflows](#common-workflows)
- [Advanced Features](#advanced-features)
- [Examples](#examples)

## Basic Usage

```bash
python run_magnet.py [options]
```

### Quick Start

```bash
# Run 5 cycles in mock mode (CPU-only)
python run_magnet.py --cycles 5 --mock

# Run with visualizations
python run_magnet.py --cycles 10 --mock --visualize --auto-open

# Continuous monitoring
python run_magnet.py --watch --metrics-report
```

## Command-Line Options

### Core Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cycles` | int | infinite | Number of research cycles to run |
| `--mock` | flag | false | Use mock LLM and physics (CPU-only mode) |
| `--memory` | path | `memory/knowledge` | Path to knowledge base storage |
| `--log-level` | choice | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Visualization Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--visualize` | flag | false | Generate visualizations after run completes |
| `--export-html` | path | auto | Export HTML dashboard to specified path |
| `--auto-open` | flag | false | Auto-open dashboard in browser |

### Reporting Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--metrics-report` | flag | false | Print detailed metrics summary at end |

### State Management Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--resume` | path | - | Resume from previous state file |
| `--save-state-every` | int | 1 | Save state every N cycles |

### Monitoring Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--watch` | flag | false | Run continuously, monitoring for experiments |

## Common Workflows

### 1. Quick Research Run

Run a short research session with automatic visualizations:

```bash
python run_magnet.py --cycles 5 --mock --visualize --auto-open
```

**What happens:**
1. Runs 5 autonomous research cycles
2. Generates HTML dashboard and plots
3. Opens dashboard in your default browser
4. State saved after each cycle

### 2. Long-Running Research

Start a 24-hour continuous research session:

```bash
python run_magnet.py --watch --mock --metrics-report --log-level WARNING
```

**What happens:**
1. Runs continuously until interrupted (Ctrl+C)
2. Shows metrics after each cycle
3. Minimal logging (warnings only)
4. State persisted continuously

### 3. Resume Previous Session

Continue a previous research session:

```bash
python run_magnet.py --resume memory/orchestrator_state.json --cycles 10 --mock
```

**What happens:**
1. Loads previous state (cycle count, best designs, etc.)
2. Continues from where you left off
3. Runs 10 more cycles

### 4. Generate Report from Existing Data

Create visualizations from a completed run:

```bash
python run_magnet.py --cycles 0 --mock --visualize --export-html results/final_report.html
```

**What happens:**
1. Loads existing knowledge base
2. Generates fresh visualizations
3. Exports to custom HTML path

## Advanced Features

### Custom Dashboard Location

```bash
python run_magnet.py --cycles 10 --mock --export-html ~/Desktop/magnet_dashboard.html --auto-open
```

Generates dashboard at a custom location and opens it.

### State Persistence Control

```bash
python run_magnet.py --watch --save-state-every 5
```

Saves orchestrator state every 5 cycles instead of every cycle (reduces I/O for long runs).

### Debug Mode

```bash
python run_magnet.py --cycles 3 --mock --log-level DEBUG --metrics-report
```

Runs with full debug logging for troubleshooting.

## Examples

### Example 1: Development Testing

Quick validation during development:

```bash
python run_magnet.py --cycles 2 --mock --log-level WARNING --metrics-report
```

### Example 2: Production Research Run

24-hour autonomous research with periodic checkpoints:

```bash
python run_magnet.py --cycles 100 --save-state-every 10 --visualize --export-html production_run.html
```

### Example 3: Presentation Mode

Generate impressive visualizations for a presentation:

```bash
# Run research
python run_magnet.py --cycles 20 --mock

# Generate visualizations
python run_magnet.py --cycles 0 --mock --visualize --export-html presentation_dashboard.html --auto-open
```

### Example 4: Continuous Monitoring

Monitor MAGNET performance in real-time:

```bash
python run_magnet.py --watch --metrics-report --log-level WARNING
```

Press Ctrl+C to stop. State is saved automatically.

### Example 5: Resume After Crash

If MAGNET crashes or is interrupted:

```bash
# Check what cycle you were on
cat memory/orchestrator_state.json | grep cycle_number

# Resume from that state
python run_magnet.py --resume memory/orchestrator_state.json --cycles 50
```

## Metrics Report Format

When using `--metrics-report`, you'll see:

```
======================================================================
 📈 MAGNET METRICS REPORT
======================================================================
  Cycles completed:      10
  Total experiments:     80
  Valid designs:         78 (97.5%)
  Best score:            82.45

  KB Statistics:
    Total experiments:   10
    Designs evaluated:   80
    Cycles tracked:      10
    Avg overall score:   75.23
    Hypotheses confirmed:6
    Hypotheses refuted:  4

  Top 3 Designs:
    1. Score: 82.45
       Length: 19.5m, Beam: 6.8m, Spacing: 5.2m
    2. Score: 81.92
       Length: 18.7m, Beam: 6.5m, Spacing: 5.0m
    3. Score: 81.34
       Length: 20.1m, Beam: 7.0m, Spacing: 5.4m
======================================================================
```

## Visualization Outputs

When using `--visualize` or `--export-html`, MAGNET generates:

- **HTML Dashboard** (`results/dashboard_TIMESTAMP.html`)
  - Interactive research overview
  - Performance trends
  - Design space exploration
  - Best designs table

- **Improvement Plot** (`results/improvement_TIMESTAMP.png`)
  - Best score vs. cycle
  - Average score trends
  - Valid design rate

- **Design Space Plot** (`results/design_space_TIMESTAMP.png`)
  - 2D parameter visualization
  - Color-coded by performance
  - Exploration coverage

- **Pareto Frontier** (`results/pareto_TIMESTAMP.png`)
  - Multi-objective trade-offs
  - Stability vs. Speed vs. Efficiency

## Tips and Best Practices

### Performance

- Use `--log-level WARNING` for faster execution (less I/O)
- Set `--save-state-every 10` for long runs to reduce disk writes
- Run `--mock` mode for development and testing

### Reliability

- Always use `--metrics-report` to verify results
- Check state files periodically: `cat memory/orchestrator_state.json`
- Use `--resume` to recover from interruptions

### Debugging

- Use `--log-level DEBUG` to diagnose issues
- Check `memory/orchestrator_state.json` for error counts
- Review visualization plots to identify problems

### Production

- Use `--watch` for continuous operation
- Combine `--visualize` with `--save-state-every 50` for daily reports
- Set up cron jobs to generate periodic reports

## Troubleshooting

### "State file not found"

```bash
# Create fresh state
rm memory/orchestrator_state.json
python run_magnet.py --cycles 5 --mock
```

### "Visualization generation failed"

Ensure you have matplotlib and seaborn installed:

```bash
pip install matplotlib seaborn
```

### Watch mode won't stop

Press Ctrl+C to gracefully stop watch mode. State will be saved automatically.

### Dashboard won't open

Try opening manually:

```bash
open results/dashboard_TIMESTAMP.html  # macOS
xdg-open results/dashboard_TIMESTAMP.html  # Linux
```

## Getting Help

```bash
python run_magnet.py --help
```

Shows all available options and examples.

## Next Steps

- Read [README.md](README.md) for system architecture
- Check [CHANGELOG.md](CHANGELOG.md) for recent updates
- Review [memory/README.md](memory/README.md) for knowledge base details
