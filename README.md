# ARC - Autonomous Research Collective

**Version:** 1.1.0-alpha (Phase D)
**Status:** Multi-Agent Architecture - Production Ready
**License:** MIT

## Overview

ARC (Autonomous Research Collective) is a multi-agent autonomous ML research framework that uses LLM-based reasoning agents to design, execute, and learn from machine learning experiments.

### Key Features

- **🆕 True Multi-Agent Architecture**: 9 specialized agents with democratic voting
- **🆕 Heterogeneous Models**: Different LLMs for different roles (Claude, DeepSeek, Qwen, Llama)
- **🆕 Democratic Consensus**: Weighted voting with supervisor oversight
- **🆕 Supervisor Veto Power**: Final safety gatekeeper with override authority
- **🆕 FDA-Aligned Development Logging**: Automatic traceability and provenance tracking
- **🆕 Role-Specific Timeouts**: Historian gets 600s for deep reasoning (configurable)
- **🆕 RunPod Deployment Ready**: Production Docker configuration with GPU support
- **Offline Operation**: Full functionality without network/models (mock mode)
- **Safety-First Design**: SEMI/AUTO/FULL autonomy modes with human oversight
- **File-Based Protocol Memory**: JSON-based inter-agent communication
- **Real GPU Training**: PyTorch integration with experiment tracking
- **Enhanced Dashboard**: 8 tabs including Agents, Supervisor, and Insights
- **Snapshot & Rollback**: State preservation and restoration

## Architecture

### Phase D: Multi-Agent Architecture (v1.1.0)

```
┌──────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                       │
│     Multi-Agent Orchestrator + Consensus Engine              │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼───────────────────────────────────┐
│                     AGENT REGISTRY (9 Agents)                │
│                                                               │
│  Strategic:        │ Proposal:           │ Safety:           │
│  • Director (2.0)  │ • Architect (1.5)   │ • Critic (2.0)    │
│                    │ • Explorer (1.2)    │ • Critic 2 (1.8)  │
│                    │ • Param Sci (1.5)   │ • Supervisor (3.0)│
│                                                               │
│  Memory:           │ Execution:                               │
│  • Historian (1.0) │ • Executor (1.0)                        │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼───────────────────────────────────┐
│                     LLM ROUTING LAYER                        │
│  Claude Sonnet 4.5 │ DeepSeek R1 │ Qwen 2.5 │ Llama 3 8B    │
│  (Strategy)        │ (Analysis)  │ (Safety) │ (Validator)   │
│  Role-Specific Timeouts: Historian=600s, Others=120s         │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼───────────────────────────────────┐
│                  FILE-BASED PROTOCOL MEMORY                  │
│  directive.json │ proposals.json │ reviews.json │ votes.jsonl│
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼───────────────────────────────────┐
│              FDA DEVELOPMENT LOGGING (NEW)                   │
│  experiments/ │ cycles/ │ data/ │ risk/ │ git_commits/       │
└──────────────────────────────────────────────────────────────┘

Numbers in parentheses = Voting weights
```

### 🆕 Phase D Agent Roles

| Agent               | Model            | Weight | Responsibility                       | Timeout |
|---------------------|------------------|--------|--------------------------------------|---------|
| Director            | Claude Sonnet    | 2.0    | Strategic planning, mode control     | 120s    |
| Architect           | DeepSeek R1      | 1.5    | Experiment design                    | 120s    |
| **Explorer** ⭐      | Qwen 2.5         | 1.2    | Parameter space exploration          | 120s    |
| **Param Scientist** ⭐| DeepSeek R1     | 1.5    | Hyperparameter optimization          | 120s    |
| Critic              | Qwen 2.5         | 2.0    | Primary safety review                | 120s    |
| **Critic Secondary** ⭐| DeepSeek R1    | 1.8    | Secondary safety, prevent groupthink | 120s    |
| **Supervisor** ⭐    | Llama 3 (Local)  | **3.0**| **Final validation, veto power**     | 120s    |
| **Historian** 🔧    | DeepSeek R1      | 1.0    | Memory management                    | **600s**|
| Executor            | DeepSeek R1      | 1.0    | Training execution                   | 120s    |

⭐ = New in Phase D | 🔧 = Enhanced timeout support

## 🆕 FDA-Aligned Development Logging

ARC now includes automatic development logging that demonstrates professional, methodical development for regulatory contexts (FDA, ISO 13485, GMLP Principle 9).

### What Gets Logged Automatically

**Experiment Logging** (`dev_logs/experiments/`)
- Complete config (model, dataset, hyperparameters)
- All metrics (AUC, sensitivity, specificity, accuracy)
- Model and dataset versions
- Reasoning summaries
- Execution status and duration
- Checkpoint paths

**Research Cycle Logging** (`dev_logs/cycles/`)
- Agents involved in each cycle
- Proposals generated and approved
- Decision reasoning
- Failures and warnings
- Supervisor vetoes and conflicts
- Cycle duration

**Risk Event Logging** (`dev_logs/risk/`)
- Cycle crashes (high severity)
- LLM timeouts (medium severity)
- Supervisor vetoes (low severity)
- Experiment failures (medium severity)
- Training errors with context

**Data Provenance Logging** (`dev_logs/data/`)
- Dataset preprocessing operations
- Input/output checksums (MD5)
- Transformations applied
- File counts and validation
- Processing metadata

**Git Commit Tracking** (`dev_logs/git_commits/`)
- Automatic commit logging
- Code change tracking

**System Snapshots** (`dev_logs/system_snapshots/`)
- Per-cycle system state
- Configuration snapshots
- Reproducibility support

### Log Formats

All logs written in dual format:
- **JSONL** (`.jsonl`): Machine-readable, line-delimited JSON
- **TXT** (`.txt`): Human-readable summaries

### FDA Compliance Features

✅ **Traceability**: Every decision tracked from proposal to result
✅ **Structured Iteration**: Cycle-by-cycle progression documented
✅ **Controlled Changes**: Git commits + system snapshots
✅ **Reproducibility**: Full config + checksums captured
✅ **Process Awareness**: Agent reasoning and decisions logged
✅ **Risk Awareness**: Timeouts, crashes, vetoes tracked

**Note**: This is *lightweight documentation* showing professional development, NOT full QMS/DHF/ISO compliance. Demonstrates methodical approach and traceability for regulatory review.

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
- **multi_agent_orchestrator.py**: Full 9-agent democratic research cycle
- **training_executor.py**: GPU training with experiment tracking
- **complete_research_loop.py**: End-to-end autonomous research

### Training Integration (`tools/acuvue_tools.py`)
AcuVue medical imaging tools with:
- Dataset preprocessing with provenance tracking
- PyTorch training with GPU support
- Evaluation and metrics calculation
- Checkpoint management
- CAM visualization generation

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (for training)
- Docker (for RunPod deployment)
- vLLM server with DeepSeek R1 or compatible model

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/1quantlogistics-ship-it/arc-autonomous-research.git
cd arc-autonomous-research

# Install dependencies
pip install -r requirements.txt

# Create directory structure
mkdir -p memory experiments logs config snapshots dev_logs

# Initialize memory files
python scripts/init_memory.py

# Copy production environment template
cp .env.production .env

# Edit .env with your configuration
nano .env
```

### RunPod Deployment

```bash
# Build Docker image
docker build -t arc-autonomous-research .

# Run with GPU support
docker run --gpus all \
  -p 8000:8000 \
  -p 8501:8501 \
  -v $(pwd)/workspace:/workspace/arc \
  arc-autonomous-research

# Or use docker-compose
docker-compose up
```

See `RUNPOD_DEPLOYMENT.md` for complete deployment guide.

## Usage

### 1. Start LLM Server (vLLM)

```bash
# Local deployment
./start_vllm.sh

# Or manual start
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
# Single cycle (with FDA logging)
python api/multi_agent_orchestrator.py 1

# Continuous autonomous research
python scripts/run_continuous_research.py

# Dry-run mode (Mac development)
python scripts/run_continuous_research.py --dry-run --max-cycles 3
```

## Operating Modes

- **SEMI**: Human approval required for all actions (default, safest)
- **AUTO**: Automatic reasoning, human approval for training
- **FULL**: Full autonomy (use with caution)

Change mode via API:
```bash
curl -X POST http://localhost:8002/mode?mode=SEMI
```

## Configuration

### Environment Variables (.env.production)

**LLM Timeouts**:
```bash
ARC_LLM_TIMEOUT=120              # Default LLM timeout (seconds)
ARC_HISTORIAN_TIMEOUT=600        # Historian timeout (DeepSeek needs longer)
ARC_LLM_MAX_RETRIES=3            # Retry attempts on timeout
ARC_LLM_RETRY_DELAY=2.0          # Delay between retries
```

**GPU Configuration**:
```bash
ARC_MAX_CONCURRENT_EXPERIMENTS=3  # Parallel training jobs
TRAINING_GPUS=0,1                 # GPUs for training
INFERENCE_GPU=2                   # GPU for LLM inference
ARC_DEFAULT_EPOCHS=5              # Training epochs
ARC_BASE_BATCH_SIZE=16            # Base batch size
```

**Polling**:
```bash
JOB_POLL_INTERVAL=2              # Job status polling (seconds)
CYCLE_POLL_INTERVAL=1            # Cycle polling (seconds)
```

See `.env.production` for complete configuration template.

## Memory Protocol

ARC uses file-based JSON protocol for all agent communication:

**Core Protocol Files:**
- `memory/directive.json` - Strategic directives from Director
- `memory/proposals.json` - Experiment ideas from Architect
- `memory/reviews.json` - Safety evaluations from Critic
- `memory/history_summary.json` - Learning history from Historian
- `memory/constraints.json` - Forbidden parameter ranges
- `memory/system_state.json` - Global ARC state

**Phase D Decision Logs:**
- `memory/decisions/voting_history.jsonl` - Multi-agent vote records
- `memory/decisions/supervisor_decisions.jsonl` - Supervisor decisions
- `memory/decisions/overrides.jsonl` - Consensus override log

**🆕 FDA Development Logs:**
- `dev_logs/experiments/experiment_history.jsonl` - All experiments
- `dev_logs/cycles/cycle_history.jsonl` - All research cycles
- `dev_logs/risk/risk_events.jsonl` - Risk tracking
- `dev_logs/data/data_provenance.jsonl` - Dataset operations
- `dev_logs/git_commits/commit_history.jsonl` - Code changes
- `dev_logs/system_snapshots/` - System state snapshots

## Validation Status

### Phase C (v0.9.0)
✅ **Smoketest #1 (Structural)** - PASSED
✅ **Smoketest #2 (Training Pipeline)** - PASSED
- Single-LLM architecture validated
- All 5 agents operational
- Real GPU training successful
- Full research loop complete

### Phase D (v1.1.0-alpha)
✅ **Multi-Agent Infrastructure** - COMPLETE
- 9 specialized agent classes implemented
- Agent registry and discovery system
- Democratic voting mechanism
- Supervisor veto power
- Offline operation (mock mode)
- Enhanced dashboard (8 tabs)
- Configuration system (YAML)

✅ **Production Enhancements** - COMPLETE
- FDA-aligned development logging
- Role-specific timeout support (Historian 600s)
- Data provenance tracking with checksums
- Risk event monitoring
- RunPod deployment configuration
- Docker containerization

🔧 **In Progress:**
- Multi-GPU training infrastructure
- GPU monitoring dashboard
- Async cycle timing optimization
- Retry-on-timeout logic

## Development

### Project Structure
```
arc_clean/
├── agents/                 # 🆕 Agent infrastructure
│   ├── base.py             # BaseAgent class
│   ├── registry.py         # Agent discovery
│   ├── protocol.py         # Communication schemas
│   ├── director_agent.py   # Strategic agent
│   ├── architect_agent.py  # Proposal agent
│   ├── critic_agent.py     # Primary safety
│   ├── critic_secondary.py # 🆕 Secondary safety
│   ├── historian_agent.py  # Memory agent
│   ├── executor_agent.py   # Execution agent
│   ├── explorer.py         # 🆕 Exploration agent
│   ├── parameter_scientist.py # 🆕 Optimization agent
│   └── supervisor.py       # 🆕 Oversight agent
├── llm/                    # 🆕 LLM integration
│   ├── client.py           # LLM client
│   ├── mock_client.py      # 🆕 Offline mock
│   ├── router.py           # 🆕 Model routing (role-specific timeouts)
│   └── models.py           # 🆕 Model configs
├── consensus/              # 🆕 Voting mechanisms
│   ├── voting.py           # Democratic voting
│   └── conflict_resolution.py # Conflict handling
├── config/                 # 🆕 Configuration system
│   ├── loader.py           # Config loader
│   ├── agents.example.yaml # Agent registry template
│   ├── models.example.yaml # Model endpoints
│   └── consensus.example.yaml # Voting rules
├── tools/                  # Development tools
│   ├── acuvue_tools.py     # Medical imaging tools
│   └── dev_logger.py       # 🆕 FDA development logger
├── api/                    # Core services
│   ├── control_plane.py
│   ├── dashboard.py        # 🔄 Extended with 3 new tabs
│   ├── multi_agent_orchestrator.py # 🔄 FDA logging integrated
│   ├── training_executor.py # 🔄 FDA logging integrated
│   └── mock_data.py        # 🆕 Mock data generator
├── scripts/                # Utility scripts
│   └── run_continuous_research.py # 🆕 Continuous loop
├── .env.production         # 🔄 Production config template
├── Dockerfile              # 🆕 RunPod deployment
├── docker-compose.yml      # 🆕 Service orchestration
├── RUNPOD_DEPLOYMENT.md    # 🆕 Deployment guide
├── DEV1_IMPLEMENTATION_GUIDE.md # 🆕 Implementation reference
├── PHASE_D_PLAN.md         # Phase D documentation
├── requirements.txt
└── README.md
```

🆕 = New in v1.1.0 | 🔄 = Updated in v1.1.0

### Running Tests

```bash
# Test LLM endpoint
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Test", "max_tokens": 50}'

# Test Control Plane
curl http://localhost:8002/status

# Test Historian timeout configuration
python -c "from config import get_settings; print(f'Historian timeout: {get_settings().historian_timeout}s')"

# Run validation cycle
python api/multi_agent_orchestrator.py 1

# View FDA development logs
ls -la dev_logs/
cat dev_logs/experiments/experiment_summary.txt
cat dev_logs/cycles/cycle_summary.txt
```

## Safety Features

- **Command Allowlist**: Only safe commands can execute
- **SEMI Mode**: Human-in-the-loop by default
- **Snapshot System**: State preservation before risky operations
- **Rollback**: Restore to previous stable state
- **Constraint Tracking**: Forbidden parameter ranges enforced
- **🆕 Risk Monitoring**: Automatic logging of crashes, timeouts, vetoes
- **🆕 Data Provenance**: Checksums and transformation tracking

## Performance

- Reasoning cycle: ~20-30 seconds (with 600s Historian timeout when needed)
- Training per experiment: ~0.7-0.8 seconds (minimal stub)
- Full research loop: ~40-60 seconds
- GPU memory: Returns to baseline after training
- FDA logging overhead: <100ms per cycle

## Phase D Features

### 🆕 Democratic Consensus Voting

Proposals are approved via weighted voting:

```python
# Each agent votes: approve/reject/revise
# Weighted score = Σ(vote * weight * confidence) / Σ(weight * confidence)
# Consensus if score >= 0.66

Example:
  Director (2.0):    Approve (conf: 0.9) → +1.8
  Critic (2.0):      Approve (conf: 0.8) → +1.6
  Architect (1.5):   Approve (conf: 0.85)→ +1.275
  Supervisor (3.0):  Approve (conf: 0.95)→ +2.85
  → Weighted score: 0.85 → Consensus reached ✓
```

### 🆕 Supervisor Veto Power

The Supervisor (weight 3.0) can override any consensus:
- **Veto if critical risk detected** (forbidden parameter ranges)
- **Override if excessive caution** (low-risk proposal rejected)
- **Final validation** before all executions
- **Decision logging** for audit trail (FDA logs)

### 🆕 Role-Specific Timeouts

Different agents get different timeout values:
- **Historian**: 600 seconds (deep reasoning with DeepSeek)
- **All others**: 120 seconds (standard operations)
- **Configurable via environment variables**
- **Retry logic**: 3 attempts with 2s delay

### 🆕 Offline Operation

Full system works without network:
```bash
export ARC_OFFLINE_MODE=true
python api/dashboard.py  # Uses MockLLMClient
```

All agents return deterministic, structured responses for:
- Development without live models
- CI/CD automated testing
- Air-gapped deployments
- Demo mode

### 🆕 Configuration System

YAML-based agent and model configuration:

```yaml
# config/agents.yaml
agents:
  - id: supervisor_001
    role: supervisor
    model: llama-3-8b-local
    voting_weight: 3.0
    capabilities: [supervision, validation]
```

## Documentation

### Core Documentation
- **[README.md](README.md)**: This file - overview and quick start
- **[PHASE_D_PLAN.md](PHASE_D_PLAN.md)**: Complete Phase D architecture guide

### Deployment Documentation
- **[RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md)**: RunPod deployment guide
- **[DEV1_IMPLEMENTATION_GUIDE.md](DEV1_IMPLEMENTATION_GUIDE.md)**: Infrastructure implementation reference
- **[DEV1_STATUS.md](Desktop/ARC_RunPod_Deployment/DEV1_STATUS.md)**: Current implementation status
- **[DEV1_QUICK_START.md](Desktop/ARC_RunPod_Deployment/DEV1_QUICK_START.md)**: Quick start guide

### Configuration Templates
- **[config/agents.example.yaml](config/agents.example.yaml)**: Agent configuration
- **[config/models.example.yaml](config/models.example.yaml)**: Model endpoints
- **[config/consensus.example.yaml](config/consensus.example.yaml)**: Voting rules
- **[.env.production](.env.production)**: Production environment template

## Contributing

This is a research prototype. Contributions welcome for:
- Additional agent roles and capabilities
- Enhanced consensus mechanisms
- Multi-model optimization strategies
- Real model training integration
- Distributed multi-pod execution
- Advanced experiment design patterns
- FDA/ISO compliance features

## Citation

If you use ARC in your research, please cite:

```
@software{arc2025,
  title={ARC: Autonomous Research Collective - Phase D Multi-Agent Architecture},
  author={1Quant Logistics},
  year={2025},
  version={1.1.0-alpha},
  url={https://github.com/1quantlogistics-ship-it/arc-autonomous-research}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with DeepSeek R1 and vLLM
- Inspired by autonomous research systems
- Validated on RunPod infrastructure
- FDA logging guidance from GMLP Principle 9 and ISO 13485
