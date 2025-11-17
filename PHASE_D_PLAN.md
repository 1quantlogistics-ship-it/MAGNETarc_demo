# PHASE D: Multi-Agent Architecture Plan

**ARC v1.1.0-alpha | Phase D Implementation**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Agent Roles](#agent-roles)
4. [Communication Protocol](#communication-protocol)
5. [Consensus Mechanism](#consensus-mechanism)
6. [Supervisor Authority](#supervisor-authority)
7. [Model Routing](#model-routing)
8. [Offline Operation](#offline-operation)
9. [Migration Path](#migration-path)
10. [Testing Strategy](#testing-strategy)
11. [Future Extensions](#future-extensions)

---

## Overview

### Transition from Phase C to Phase D

**Phase C (v0.9.0):** Single-LLM system with role-based prompts
- All agents use the same LLM (DeepSeek R1)
- Roles simulated via different prompt templates
- Sequential execution (no parallelism)
- No voting or consensus mechanisms

**Phase D (v1.1.0):** True multi-agent architecture
- ✅ Heterogeneous model support (different LLMs per role)
- ✅ Specialized agent classes (not just prompts)
- ✅ Democratic voting and consensus
- ✅ Supervisor oversight with veto power
- ✅ Agent registry and discovery
- ✅ Parallel agent execution capability
- ✅ Offline operation support

### Goals

1. **Enable Multi-Model Collaboration**: Allow different LLMs to specialize in different roles
2. **Democratic Decision-Making**: Consensus-based approval with weighted voting
3. **Safety Enhancement**: Supervisor agent with veto power over risky decisions
4. **Scalability**: Support for adding new agents without code changes
5. **Resilience**: Offline operation and graceful degradation

---

## Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATOR LAYER                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │Multi-Agent Orch│  │Consensus Engine│  │Conflict Resolv.│   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AGENT REGISTRY                            │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │Director│ │Architect│ │Critic  │ │Explorer│ │Supervis│...    │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LLM ROUTING LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │Claude Sonnet │  │DeepSeek R1   │  │Qwen 2.5 32B  │         │
│  │(Strategy)    │  │(Analysis)    │  │(Safety)      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │Llama 3 8B    │  │Mock LLM      │                            │
│  │(Validator)   │  │(Offline)     │                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MEMORY PROTOCOL                           │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │directive│ │proposals│ │reviews │ │history │ │decisions│      │
│  │.json   │ │.json   │ │.json   │ │.json   │ │/*.jsonl│      │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Agent Layer (`agents/`)
- **BaseAgent**: Abstract base class for all agents
- **Specialized Agents**: DirectorAgent, ArchitectAgent, CriticAgent, etc.
- **AgentRegistry**: Central registry for agent discovery
- **Protocol**: Inter-agent communication schemas

#### 2. LLM Layer (`llm/`)
- **LLMClient**: Generic LLM client (OpenAI-compatible)
- **MockLLMClient**: Offline mock for development
- **LLMRouter**: Routes roles to specific models
- **Models**: Model capability definitions

#### 3. Consensus Layer (`consensus/`)
- **VotingSystem**: Democratic voting mechanism
- **ConflictResolver**: Resolution of voting conflicts
- **Metrics**: Consensus quality analysis

#### 4. Configuration (`config/`)
- **agents.yaml**: Agent registry definitions
- **models.yaml**: Model endpoint configurations
- **consensus.yaml**: Voting rules and thresholds
- **ConfigLoader**: Load and validate configs

---

## Agent Roles

### Phase D Agent Ecosystem (9 Agents)

#### Strategic Agents

**1. Director (director_001)**
- **Model**: Claude Sonnet 4.5
- **Voting Weight**: 2.0
- **Responsibilities**:
  - Set research mode (explore/exploit/recover)
  - Allocate novelty budgets
  - Define strategic objectives
  - Detect stagnation
- **Capabilities**: `strategy`, `planning`
- **Veto Power**: Strategic override on mission-critical decisions

#### Proposal Generation Agents

**2. Architect (architect_001)**
- **Model**: DeepSeek R1
- **Voting Weight**: 1.5
- **Responsibilities**:
  - Design experiment proposals
  - Generate hypotheses
  - Predict metric impacts
  - Assign novelty categories
- **Capabilities**: `proposal_generation`

**3. Explorer (explorer_001)** ⭐ *NEW*
- **Model**: Qwen 2.5 32B
- **Voting Weight**: 1.2
- **Responsibilities**:
  - Systematic parameter space exploration
  - Identify high-potential regions
  - Map parameter boundaries
  - Use exploration strategies (grid, random, bayesian)
- **Capabilities**: `exploration`, `proposal_generation`

**4. Parameter Scientist (parameter_scientist_001)** ⭐ *NEW*
- **Model**: DeepSeek R1
- **Voting Weight**: 1.5
- **Responsibilities**:
  - Advanced hyperparameter optimization
  - Bayesian optimization, TPE, CMA-ES
  - Model parameter-performance relationships
  - Prune unpromising regions
- **Capabilities**: `proposal_generation`, `exploration`

#### Safety & Review Agents

**5. Critic (critic_001)**
- **Model**: Qwen 2.5 32B
- **Voting Weight**: 2.0
- **Responsibilities**:
  - Primary safety review
  - Enforce constraints
  - Detect redundancy
  - Flag risky configurations
- **Capabilities**: `safety_review`, `constraint_checking`

**6. Critic Secondary (critic_secondary_001)** ⭐ *NEW*
- **Model**: DeepSeek R1
- **Voting Weight**: 1.8
- **Responsibilities**:
  - Secondary safety opinion
  - Challenge scientific assumptions
  - Detect logical flaws
  - Prevent groupthink
- **Capabilities**: `safety_review`, `validation`

**7. Supervisor (supervisor_001)** ⭐ *NEW*
- **Model**: Llama 3 8B (Local/Offline)
- **Voting Weight**: 3.0 ⚡ **HIGHEST - VETO POWER**
- **Responsibilities**:
  - Final validation before execution
  - Risk assessment (low/medium/high/critical)
  - Override consensus if safety at risk
  - Monitor agent behavior
  - Enforce system-level constraints
- **Capabilities**: `supervision`, `validation`, `safety_review`
- **Special Powers**: Can veto any decision unilaterally

#### Memory & Execution Agents

**8. Historian (historian_001)**
- **Model**: DeepSeek R1
- **Voting Weight**: 1.0
- **Responsibilities**:
  - Maintain compressed research history
  - Track winning/failing configs
  - Infer constraints from failures
  - Analyze performance trends
- **Capabilities**: `memory_management`

**9. Executor (executor_001)**
- **Model**: DeepSeek R1
- **Voting Weight**: 1.0
- **Responsibilities**:
  - Translate proposals to training jobs
  - Execute via Control Plane
  - Monitor training progress
  - Collect metrics
- **Capabilities**: `execution`

---

## Communication Protocol

### File-Based Protocol (Maintains v0.9.0 Compatibility)

All agents communicate via **JSON files** in `/workspace/arc/memory/`:

#### Core Protocol Files

1. **`directive.json`** - Strategic directive from Director
2. **`proposals.json`** - Experiment proposals from Architects
3. **`reviews.json`** - Safety reviews from Critics
4. **`history_summary.json`** - Research history from Historian
5. **`constraints.json`** - Safety constraints
6. **`system_state.json`** - Global system state

#### New Phase D Files

7. **`decisions/voting_history.jsonl`** - Multi-agent vote records
8. **`decisions/supervisor_decisions.jsonl`** - Supervisor decisions
9. **`decisions/overrides.jsonl`** - Consensus override log
10. **`messages.jsonl`** - Inter-agent messages (optional)

### Message Schema

```python
{
  "message_id": "uuid",
  "message_type": "proposal" | "vote" | "approval" | "rejection",
  "sender_id": "agent_id",
  "recipient_id": "agent_id" or null (broadcast),
  "timestamp": "ISO-8601",
  "cycle_id": 123,
  "payload": {...},
  "metadata": {...}
}
```

---

## Consensus Mechanism

### Weighted Democratic Voting

#### Voting Process

1. **Proposal Submission**: Architect(s) submit proposals
2. **Voting Round**: All eligible agents vote
3. **Vote Collection**: Each agent provides:
   - Decision: `approve`, `reject`, `revise`, `abstain`
   - Confidence: 0.0-1.0
   - Reasoning: Text explanation
4. **Weighted Score Calculation**:
   ```
   weighted_score = Σ(vote_i * weight_i * confidence_i) / Σ(weight_i * confidence_i)
   ```
5. **Consensus Check**: If `|weighted_score| >= 0.66`, consensus reached
6. **Supervisor Validation**: Supervisor reviews and can override
7. **Final Decision**: Approve, Reject, or Revise

#### Voting Weights

| Agent Role           | Weight | Rationale                              |
|----------------------|--------|----------------------------------------|
| Supervisor           | 3.0    | Veto power, final safety gatekeeper    |
| Director             | 2.0    | Strategic authority                    |
| Critic (Primary)     | 2.0    | Safety is paramount                    |
| Critic (Secondary)   | 1.8    | Secondary safety voice                 |
| Architect            | 1.5    | Proposal expertise                     |
| Parameter Scientist  | 1.5    | Optimization expertise                 |
| Explorer             | 1.2    | Exploration insights                   |
| Historian            | 1.0    | Historical context                     |
| Executor             | 1.0    | Feasibility assessment                 |

#### Consensus Thresholds

- **Approve**: `weighted_score >= 0.66` (66% agreement)
- **Reject**: `weighted_score <= -0.66` (66% disagreement)
- **Revise**: `-0.66 < weighted_score < 0.66` (no clear consensus)

### Conflict Resolution Strategies

When consensus is not reached:

1. **Conservative** (default): Reject if unclear (safety first)
2. **Progressive**: Approve if no strong objections
3. **Majority Rule**: Simple vote count (ignore weights)
4. **Highest Confidence**: Trust most confident voter
5. **Supervisor Override**: Supervisor makes final call
6. **Mediation**: Attempt compromise via revisions

---

## Supervisor Authority

### Veto Power Rules

The Supervisor can **override any consensus** if:

1. **Critical Risk Detected**: Proposal violates safety constraints
2. **Constraint Violations**: Forbidden parameter ranges used
3. **Excessive Caution**: Low-risk proposal rejected by groupthink
4. **Anomalous Behavior**: Agents exhibit suspicious voting patterns

### Supervisor Decision Flow

```
Proposal → Consensus Vote → Supervisor Review
                              │
                              ├─ Approve (consensus + safe) → Execute
                              │
                              ├─ Reject (consensus + risky) → Reject
                              │
                              ├─ Override (safe but rejected) → Approve*
                              │
                              └─ Veto (risky but approved) → Reject*

* = Override applied
```

### Risk Assessment Matrix

| Risk Level | Criteria                              | Supervisor Action          |
|------------|---------------------------------------|----------------------------|
| Low        | Exploit mode, proven config           | Approve automatically      |
| Medium     | Explore mode, within safe bounds      | Approve if consensus       |
| High       | Wildcat mode, novel parameters        | Scrutinize heavily         |
| Critical   | Forbidden ranges, constraint violation| **VETO** (automatic reject)|

---

## Model Routing

### Default Role-to-Model Mapping

```yaml
Director           → Claude Sonnet 4.5    # Best strategic reasoning
Architect          → DeepSeek R1          # Excellent analysis
Critic             → Qwen 2.5 32B         # Strong safety review
Critic Secondary   → DeepSeek R1          # Diverse perspective
Historian          → DeepSeek R1          # Memory compression
Executor           → DeepSeek R1          # Code generation
Explorer           → Qwen 2.5 32B         # Exploration creativity
Parameter Scientist→ DeepSeek R1          # Math/optimization
Supervisor         → Llama 3 8B (Local)   # Offline validator
```

### Model Selection Rationale

**Claude Sonnet 4.5**: Strategic planning, complex reasoning, long-term thinking
**DeepSeek R1**: Analysis, constraint reasoning, scientific rigor
**Qwen 2.5 32B**: Safety-focused, data tasks, exploratory thinking
**Llama 3 8B Local**: Offline validation, schema checking, fast inference

### Heterogeneous Benefits

1. **Diversity**: Different models have different biases → better coverage
2. **Specialization**: Match model strengths to role requirements
3. **Cost Optimization**: Use expensive models only where needed
4. **Redundancy**: If one model fails, others can continue
5. **Offline Capability**: Local models enable air-gapped operation

---

## Offline Operation

### Offline-First Design

Phase D supports **full offline operation** using:

1. **MockLLMClient**: Returns deterministic, structured JSON responses
2. **Local Models**: Llama 3 8B for Supervisor (can run on CPU)
3. **No Network Calls**: All components work without internet
4. **Graceful Degradation**: Falls back to mock if models unavailable

### Offline Use Cases

- **Development**: Build and test multi-agent features without live models
- **CI/CD**: Automated testing in isolated environments
- **Air-Gapped Deployment**: High-security environments
- **Demo Mode**: Dashboard visualization without backend

### Enabling Offline Mode

```bash
# Environment variable
export ARC_OFFLINE_MODE=true

# Or in code
llm_router = LLMRouter(offline_mode=True)
```

---

## Migration Path

### From v0.9.0 (Single-LLM) → v1.1.0 (Multi-Agent)

#### Step 1: Backward Compatibility (Week 1-2)

✅ Phase D maintains **full backward compatibility**:
- Existing orchestrators (`full_cycle_orchestrator.py`) still work
- File-based protocol unchanged
- No breaking changes to Control Plane API

#### Step 2: Gradual Adoption (Week 3-4)

1. **Deploy Phase D Code**: Install new `agents/`, `llm/`, `consensus/` directories
2. **Test in Mock Mode**: Validate multi-agent infrastructure offline
3. **Single-Model Multi-Agent**: Use DeepSeek R1 for all agents initially
4. **Add Supervisor**: Enable supervisor oversight with mock LLM

#### Step 3: Multi-Model Transition (Week 5-6)

1. **Deploy Qwen 2.5 32B**: Add second model for Critics/Explorer
2. **Deploy Llama 3 8B**: Add local Supervisor model
3. **Optional: Claude API**: Add Claude Sonnet for Director (if API key available)
4. **Test Heterogeneous Cycles**: Run full multi-model research cycles

#### Step 4: Full Phase D Operation (Week 7+)

1. **Enable Democratic Voting**: Switch to consensus-based approval
2. **Activate Supervisor Veto**: Grant Supervisor override authority
3. **Monitor Consensus Metrics**: Track agreement rates, controversies
4. **Tune Voting Weights**: Adjust based on agent performance

### Configuration Changes

**Old (v0.9.0)**: Hardcoded LLM endpoint in orchestrators

**New (v1.1.0)**: YAML-based configuration

```yaml
# config/agents.yaml
agents:
  - id: director_001
    role: director
    model: claude-sonnet-4.5
    voting_weight: 2.0

# config/models.yaml
models:
  - id: claude-sonnet-4.5
    endpoint: https://api.anthropic.com/v1/messages
    provider: anthropic
```

---

## Testing Strategy

### Phase D Testing Levels

#### 1. Unit Tests (Offline)

Test individual components without network:

```bash
pytest tests/agents/test_base_agent.py
pytest tests/llm/test_mock_client.py
pytest tests/consensus/test_voting.py
```

**Coverage**:
- Agent initialization and lifecycle
- Vote calculation logic
- Conflict resolution strategies
- Mock LLM responses

#### 2. Integration Tests (Offline)

Test multi-agent interactions with mocks:

```bash
pytest tests/integration/test_multi_agent_voting.py
pytest tests/integration/test_supervisor_override.py
```

**Coverage**:
- Proposal → Vote → Consensus flow
- Supervisor veto scenarios
- Agent registry operations
- Config loading

#### 3. End-to-End Tests (Single Model)

Test full cycles with real LLM (single model):

```bash
python tests/e2e/test_single_model_cycle.py
```

**Coverage**:
- Full research cycle (Historian → Director → Architect → Critic → Supervisor → Executor)
- All agents use same model (DeepSeek R1)
- File-based protocol compliance

#### 4. Multi-Model Tests (Live Models)

Test heterogeneous model deployment:

```bash
python tests/e2e/test_multi_model_cycle.py
```

**Coverage**:
- Different LLMs for different roles
- Model routing verification
- Cross-model consensus

#### 5. Dashboard Tests (Visual Validation)

Launch dashboard with mock data:

```bash
streamlit run api/dashboard.py
```

**Coverage**:
- All 8 tabs render correctly
- Mock data displays properly
- Charts and visualizations work

### Test Scenarios

| Scenario                        | Expected Outcome                          |
|---------------------------------|-------------------------------------------|
| Unanimous approval              | Consensus reached, proposal approved      |
| Unanimous rejection             | Consensus reached, proposal rejected      |
| Split vote (55% approve)        | No consensus, mediation/revision          |
| Supervisor veto (critical risk) | Override consensus, proposal rejected     |
| Supervisor override (too cautious)| Override consensus, proposal approved   |
| Offline mode                    | All agents use MockLLMClient              |
| Model unavailable               | Graceful fallback to mock                 |

---

## Future Extensions

### Phase E: Advanced Multi-Agent Features (Month 7-9)

1. **Agent Fine-Tuning**
   - Fine-tune local models for specific roles
   - Curator agent to train new agents
   - Transfer learning from successful cycles

2. **Meta-Learning**
   - Agents learn optimal voting strategies
   - Automatic voting weight adjustment
   - Controversy detection and resolution

3. **Hierarchical Agents**
   - Sub-agents for specialized tasks
   - Committee-based decision making
   - Dynamic agent spawning

4. **External Knowledge Integration**
   - RAG (Retrieval-Augmented Generation) for Historian
   - Paper search agents
   - Codebase agents

### Phase F: Distributed ARC (Month 10-12)

1. **Multi-Pod Deployment**
   - Agents on different machines
   - Network-based protocol (gRPC/HTTP)
   - Distributed voting

2. **Blockchain Integration**
   - Immutable decision log
   - Verifiable consensus
   - Decentralized agent registry

3. **Federated Learning**
   - Multiple ARC instances collaborate
   - Shared knowledge base
   - Privacy-preserving insights

---

## Appendix

### File Structure Reference

```
arc_clean/
├── agents/
│   ├── __init__.py
│   ├── base.py                 # BaseAgent class
│   ├── registry.py             # AgentRegistry
│   ├── protocol.py             # Communication schemas
│   ├── director_agent.py       # Strategic planning
│   ├── architect_agent.py      # Proposal generation
│   ├── critic_agent.py         # Primary safety review
│   ├── critic_secondary.py     # Secondary safety review (NEW)
│   ├── historian_agent.py      # Memory management
│   ├── executor_agent.py       # Training execution
│   ├── explorer.py             # Parameter exploration (NEW)
│   ├── parameter_scientist.py  # Hyperparameter optimization (NEW)
│   └── supervisor.py           # Oversight & validation (NEW)
├── llm/
│   ├── __init__.py
│   ├── client.py               # LLM client
│   ├── mock_client.py          # Offline mock (NEW)
│   ├── router.py               # Model routing (NEW)
│   └── models.py               # Model definitions (NEW)
├── consensus/
│   ├── __init__.py
│   ├── voting.py               # Voting system (NEW)
│   └── conflict_resolution.py  # Conflict resolution (NEW)
├── config/
│   ├── loader.py               # Config loader (NEW)
│   ├── agents.example.yaml     # Agent registry template (NEW)
│   ├── models.example.yaml     # Model endpoints template (NEW)
│   └── consensus.example.yaml  # Voting rules template (NEW)
├── api/
│   ├── dashboard.py            # Extended with 3 new tabs
│   └── mock_data.py            # Mock data generator (NEW)
└── PHASE_D_PLAN.md             # This document (NEW)
```

### Key Metrics to Monitor

**Agent Performance**:
- Success rate per agent
- Average response time
- Vote agreement with consensus

**Consensus Quality**:
- Consensus rate (% of votes reaching consensus)
- Controversial rate (% of votes with high disagreement)
- Average confidence scores

**Supervisor Activity**:
- Override rate (% of decisions overridden)
- Veto count (critical interventions)
- Risk distribution (low/medium/high/critical)

**System Health**:
- Active agents count
- Failed task rate
- Model availability

---

**ARC v1.1.0-alpha | Phase D Implementation Complete**

This document describes the multi-agent architecture implemented in Phase D. All components are production-ready and fully tested in offline mode.

For deployment instructions, see [README.md](README.md).
For Phase C legacy documentation, see git history.
