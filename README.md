# ARC - Autonomous Research Collective

**Version:** 1.1.0-alpha (Phase D)
**Status:** Multi-Agent Architecture - Development Ready
**License:** MIT

## Overview

ARC (Autonomous Research Collective) is a multi-agent autonomous ML research framework that uses LLM-based reasoning agents to design, execute, and learn from machine learning experiments.

### Key Features

- **🆕 True Multi-Agent Architecture**: 9 specialized agents with democratic voting
- **🆕 Heterogeneous Models**: Different LLMs for different roles (Claude, DeepSeek, Qwen, Llama)
- **🆕 Democratic Consensus**: Weighted voting with supervisor oversight
- **🆕 Supervisor Veto Power**: Final safety gatekeeper with override authority
- **🆕 Offline Operation**: Full functionality without network/models (mock mode)
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
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼───────────────────────────────────┐
│                  FILE-BASED PROTOCOL MEMORY                  │
│  directive.json │ proposals.json │ reviews.json │ votes.jsonl│
└──────────────────────────────────────────────────────────────┘

Numbers in parentheses = Voting weights
```

### 🆕 Phase D Agent Roles

| Agent               | Model            | Weight | Responsibility                       |
|---------------------|------------------|--------|--------------------------------------|
| Director            | Claude Sonnet    | 2.0    | Strategic planning, mode control     |
| Architect           | DeepSeek R1      | 1.5    | Experiment design                    |
| **Explorer** ⭐      | Qwen 2.5         | 1.2    | Parameter space exploration          |
| **Param Scientist** ⭐| DeepSeek R1     | 1.5    | Hyperparameter optimization          |
| Critic              | Qwen 2.5         | 2.0    | Primary safety review                |
| **Critic Secondary** ⭐| DeepSeek R1    | 1.8    | Secondary safety, prevent groupthink |
| **Supervisor** ⭐    | Llama 3 (Local)  | **3.0**| **Final validation, veto power**     |
| Historian           | DeepSeek R1      | 1.0    | Memory management                    |
| Executor            | DeepSeek R1      | 1.0    | Training execution                   |

⭐ = New in Phase D

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

**Core Protocol Files:**
- `memory/directive.json` - Strategic directives from Director
- `memory/proposals.json` - Experiment ideas from Architect
- `memory/reviews.json` - Safety evaluations from Critic
- `memory/history_summary.json` - Learning history from Historian
- `memory/constraints.json` - Forbidden parameter ranges
- `memory/system_state.json` - Global ARC state

**🆕 Phase D Decision Logs:**
- `memory/decisions/voting_history.jsonl` - Multi-agent vote records
- `memory/decisions/supervisor_decisions.jsonl` - Supervisor decisions
- `memory/decisions/overrides.jsonl` - Consensus override log

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

🔧 **In Progress:**
- Multi-model deployment testing
- Consensus quality tuning
- Heterogeneous model validation

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
│   ├── router.py           # 🆕 Model routing
│   └── models.py           # 🆕 Model configs
├── consensus/              # 🆕 Voting mechanisms
│   ├── voting.py           # Democratic voting
│   └── conflict_resolution.py # Conflict handling
├── config/                 # 🆕 Configuration system
│   ├── loader.py           # Config loader
│   ├── agents.example.yaml # Agent registry template
│   ├── models.example.yaml # Model endpoints
│   └── consensus.example.yaml # Voting rules
├── api/                    # Core services
│   ├── control_plane.py
│   ├── dashboard.py        # 🔄 Extended with 3 new tabs
│   ├── mock_data.py        # 🆕 Mock data generator
│   ├── *_orchestrator.py
│   └── training_stub.py
├── scripts/                # Utility scripts
├── PHASE_D_PLAN.md         # 🆕 Phase D documentation
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
- **Decision logging** for audit trail

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

## Phase D Documentation

For comprehensive Phase D documentation, see:
- **[PHASE_D_PLAN.md](PHASE_D_PLAN.md)**: Complete architecture guide
- **[config/agents.example.yaml](config/agents.example.yaml)**: Agent configuration
- **[config/models.example.yaml](config/models.example.yaml)**: Model endpoints
- **[config/consensus.example.yaml](config/consensus.example.yaml)**: Voting rules

## Contributing

This is a research prototype. Contributions welcome for:
- Additional agent roles and capabilities
- Enhanced consensus mechanisms
- Multi-model optimization strategies
- Real model training integration
- Distributed multi-pod execution
- Advanced experiment design patterns

## Citation

If you use ARC in your research, please cite:

```
@software{arc2025,
  title={ARC: Autonomous Research Collective - Phase D Multi-Agent Architecture},
  author={Your Name},
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
