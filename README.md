# ARC - Autonomous Research Collective

**Version:** 0.9.0  
**Status:** Post-Smoketest Verified  
**License:** MIT

## Overview

ARC (Autonomous Research Collective) is a multi-agent autonomous ML research framework that uses LLM-based reasoning agents to design, execute, and learn from machine learning experiments.

### Key Features

- **Multi-Agent Architecture**: Director, Architect, Critic, Historian, Executor roles
- **LLM-Powered Reasoning**: DeepSeek R1 integration via vLLM
- **Safety-First Design**: SEMI/AUTO/FULL autonomy modes with human oversight
- **File-Based Protocol Memory**: JSON-based inter-agent communication
- **Real GPU Training**: PyTorch integration with experiment tracking
- **Snapshot & Rollback**: State preservation and restoration
- **Web Dashboard**: Streamlit-based monitoring interface

## Architecture

```
┌─────────────┐
│  Director   │  Strategic planning
└──────┬──────┘
       │
┌──────▼──────┐
│  Architect  │  Experiment proposals
└──────┬──────┘
       │
┌──────▼──────┐
│   Critic    │  Safety evaluation
└──────┬──────┘
       │
┌──────▼──────┐
│  Executor   │  GPU training
└──────┬──────┘
       │
┌──────▼──────┐
│  Historian  │  Learning & memory
└─────────────┘
```

## Components

### Control Plane (`api/control_plane.py`)
FastAPI service for orchestration, safety validation, and state management.
- **Port:** 8002
- **Endpoints:** `/status`, `/exec`, `/train`, `/archive`, `/rollback`, `/mode`

### Dashboard (`api/dashboard.py`)
Streamlit web interface for monitoring and control.
- **Port:** 8501
- **Features:** Memory visualization, experiment tracking, live metrics

### Orchestrators
- **full_cycle_orchestrator.py**: Director → Architect → Critic loop
- **training_cycle_orchestrator.py**: Executor → Historian with real GPU training
- **complete_research_loop.py**: End-to-end research cycle

### Training Stub (`api/training_stub.py`)
Minimal PyTorch training for pipeline validation and testing.

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (for training)
- vLLM server with DeepSeek R1 or compatible model

### Setup

```bash
# Clone repository
git clone <your-repo-url>
cd arc

# Install dependencies
pip install -r requirements.txt

# Create directory structure
mkdir -p memory experiments logs config snapshots

# Initialize memory files
python scripts/init_memory.py
```

## Usage

### 1. Start LLM Server (vLLM)
```bash
python -m vllm.entrypoints.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 2
```

### 2. Start Control Plane
```bash
python api/control_plane.py
```

### 3. Start Dashboard
```bash
streamlit run api/dashboard.py --server.port 8501
```

### 4. Run Research Cycle
```bash
# Dry-run (no training)
python api/full_cycle_orchestrator.py 1

# With training
python api/complete_research_loop.py 1
```

## Operating Modes

- **SEMI**: Human approval required for all actions (default, safest)
- **AUTO**: Automatic reasoning, human approval for training
- **FULL**: Full autonomy (use with caution)

Change mode via API:
```bash
curl -X POST http://localhost:8002/mode?mode=SEMI
```

## Memory Protocol

ARC uses file-based JSON protocol for all agent communication:

- `memory/directive.json` - Strategic directives from Director
- `memory/proposals.json` - Experiment ideas from Architect
- `memory/reviews.json` - Safety evaluations from Critic
- `memory/history_summary.json` - Learning history from Historian
- `memory/constraints.json` - Forbidden parameter ranges
- `memory/system_state.json` - Global ARC state

## Validation Status

✅ **Smoketest #1 (Structural)** - PASSED  
✅ **Smoketest #2 (Training Pipeline)** - PASSED

- LLM backend operational
- Control plane functional
- Memory layer verified
- Multi-agent reasoning validated
- Real GPU training successful
- Historian metrics ingestion confirmed
- Dashboard integration working
- Full research loop complete

## Development

### Project Structure
```
arc/
├── api/                    # Core services
│   ├── control_plane.py
│   ├── dashboard.py
│   ├── *_orchestrator.py
│   └── training_stub.py
├── config/                 # Configuration templates
├── scripts/                # Utility scripts
├── requirements.txt
└── README.md
```

### Running Tests
```bash
# Test LLM endpoint
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Test", "max_tokens": 50}'

# Test Control Plane
curl http://localhost:8002/status

# Run validation cycle
python api/full_cycle_orchestrator.py 0
```

## Safety Features

- **Command Allowlist**: Only safe commands can execute
- **SEMI Mode**: Human-in-the-loop by default
- **Snapshot System**: State preservation before risky operations
- **Rollback**: Restore to previous stable state
- **Constraint Tracking**: Forbidden parameter ranges enforced

## Performance

- Reasoning cycle: ~20-30 seconds
- Training per experiment: ~0.7-0.8 seconds (minimal stub)
- Full research loop: ~40-50 seconds
- GPU memory: Returns to baseline after training

## Contributing

This is a research prototype. Contributions welcome for:
- Additional agent roles
- Enhanced safety mechanisms
- Real model training integration
- Distributed execution
- Advanced experiment design

## Citation

If you use ARC in your research, please cite:

```
@software{arc2025,
  title={ARC: Autonomous Research Collective},
  author={Your Name},
  year={2025},
  version={0.9.0},
  url={https://github.com/yourusername/arc}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with DeepSeek R1 and vLLM
- Inspired by autonomous research systems
- Validated on RunPod infrastructure
