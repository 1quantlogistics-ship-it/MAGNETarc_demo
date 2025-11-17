# ARC - Autonomous Research Collective

**Version 1.1.0** - Architecture & Stability Improvements

> **🎯 Mission**: A persistent, resilient, multi-role LLM-driven research automation system designed to autonomously propose, evaluate, and execute ML experiments.

---

## 🚀 What's New in v1.1.0

### ✅ Offline Development Support

**Work on ARC without requiring LLM backend or GPU!**

- ✅ Mock LLM client with role-based responses
- ✅ Comprehensive test suite (80+ tests)
- ✅ All tests run in ~2 seconds on CPU
- ✅ Deterministic, reproducible behavior

### ✅ Schema Validation

**Type-safe memory files with Pydantic:**

- ✅ All memory files validated on load/save
- ✅ Automatic type checking and conversion
- ✅ Clear error messages on validation failures
- ✅ Self-documenting schemas

### ✅ Configuration Management

**No more hard-coded paths:**

- ✅ Environment variable support
- ✅ Dev/test/prod profiles
- ✅ Easy customization
- ✅ Validated constraints

### ✅ Testing Infrastructure

**Professional test setup:**

- ✅ pytest with fixtures and markers
- ✅ Coverage tracking (>75% on new modules)
- ✅ Realistic mock LLM responses
- ✅ Easy to extend

---

## 📋 Quick Start

### For Development (No LLM/GPU Required)

```bash
# 1. Setup
cd arc_clean
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env if needed

# 3. Run tests
pytest

# 4. See coverage
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

### For Production (With LLM Backend)

```bash
# 1. Setup environment
export ARC_ENVIRONMENT=prod
export ARC_HOME=/workspace/arc
export ARC_LLM_ENDPOINT=http://localhost:8000/v1

# 2. Initialize memory
python scripts/init_memory.py

# 3. Start control plane
uvicorn api.control_plane:app --host 0.0.0.0 --port 8080

# 4. Start dashboard (optional)
streamlit run api/dashboard.py
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | **Start here** - Development guide & workflows |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | What's been built, metrics, next steps |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and architecture |
| [TESTING.md](docs/TESTING.md) | Testing guide and examples |

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────┐
│             LLM ORCHESTRATION LAYER (Replaceable)         │
│   - Kimi roles (Director, Architect, Critic, Historian,  │
│     Executor)                                              │
│   - OR plug-in models (Claude, Qwen, DeepSeek)            │
└──────────────▲────────────────────────────────────────────┘
               │ role-specific prompt + memory boundaries
               │
┌──────────────▼────────────────────────────────────────────┐
│               ARC CONTROL PLANE (Persistent)               │
│   - API: /exec /train /eval /status /archive /rollback    │
│   - Experiment scheduler                                   │
│   - Resource governor                                      │
└──────────────▲────────────────────────────────────────────┘
               │
┌──────────────▼────────────────────────────────────────────┐
│              RUNPOD EMBODIMENT LAYER                       │
│   - Hardware abstraction                                   │
│   - Data stores                                             │
│   - Logs, metrics, checkpoints                             │
│   - Sandbox & guardrails                                   │
└────────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# With coverage
pytest --cov=. --cov-report=term-missing

# Specific test file
pytest tests/unit/test_schemas.py

# Verbose output
pytest -vv

# Stop on first failure
pytest -x
```

### Test Coverage (v1.1.0)

| Module | Coverage | Status |
|--------|----------|--------|
| schemas.py | 95% | ✅ |
| config.py | 97% | ✅ |
| tests/mocks/* | 90% | ✅ |
| **New Code Avg** | **~95%** | ✅ |

---

## 🛠️ Development

### Using Mock LLM

```python
from tests.mocks.llm_client import MockLLMClient

# Create mock for specific role
client = MockLLMClient(role="historian")

# Get realistic response
response = client.chat_completion([
    {"role": "user", "content": "Analyze experiment history"}
])

# Response contains valid JSON matching schema
print(response["choices"][0]["message"]["content"])
```

### Using Schemas

```python
from schemas import Directive, DirectiveMode, NoveltyBudget
from schemas import validate_memory_file, save_memory_file

# Create validated directive
directive = Directive(
    cycle_id=1,
    mode=DirectiveMode.EXPLORE,
    objective="improve_auc",
    novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1)
)

# Save with atomic write
save_memory_file("directive.json", directive, atomic=True)

# Load with validation
loaded = validate_memory_file("directive.json", Directive)
```

### Using Configuration

```python
from config import get_settings

# Get configuration
settings = get_settings()

# Access paths
print(settings.memory_dir)       # /workspace/arc/memory
print(settings.experiments_dir)  # /workspace/arc/experiments

# Get specific file paths
directive_path = settings.get_memory_file_path("directive.json")
exp_path = settings.get_experiment_path("exp_1_1")
```

---

## 📂 Project Structure

```
arc_clean/
├── api/                         # Control Plane & Orchestrators
│   ├── control_plane.py         # FastAPI service
│   ├── cycle_orchestrator.py   # Orchestration logic
│   ├── dashboard.py             # Streamlit dashboard
│   └── training_stub.py         # Training pipeline
├── config.py                    # 🆕 Configuration management
├── schemas.py                   # 🆕 Memory file schemas
├── tests/                       # 🆕 Test suite
│   ├── conftest.py              # Shared fixtures
│   ├── mocks/                   # Mock LLM & fixtures
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── docs/                        # 🆕 Documentation
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   └── TESTING.md
├── requirements.txt             # 🆕 Updated dependencies
├── pytest.ini                   # 🆕 Test configuration
├── .coveragerc                  # 🆕 Coverage configuration
├── .env.example                 # 🆕 Environment template
└── IMPLEMENTATION_SUMMARY.md    # 🆕 v1.1 summary
```

---

## 🎯 Key Features

### Memory-Driven Architecture

All state persisted as validated JSON:

- `directive.json` - Strategic directives
- `history_summary.json` - Research history
- `constraints.json` - Learned safety constraints
- `proposals.json` - Experiment proposals
- `reviews.json` - Critic reviews
- `system_state.json` - Global state

### Five-Role Intelligence

1. **Historian** - Memory & constraint learning
2. **Director** - Strategic control
3. **Architect** - Experiment generation
4. **Critic** - Adversarial review
5. **Executor** - Safe execution

### Safety Features

- ✅ Mode-based approval (SEMI/AUTO/FULL/OFF)
- ✅ Command allowlist
- ✅ Constraint enforcement
- ✅ Rollback & snapshots
- ✅ Resource limits
- ✅ Timeout protection

---

## 🔧 Configuration

### Environment Variables

```bash
# Environment
export ARC_ENVIRONMENT=dev          # dev, test, prod

# Paths
export ARC_HOME=/workspace/arc
export ARC_MEMORY_DIR=/workspace/arc/memory

# LLM
export ARC_LLM_ENDPOINT=http://localhost:8000/v1
export ARC_LLM_TIMEOUT=120

# Safety
export ARC_MODE=SEMI                # SEMI, AUTO, FULL, OFF
export ARC_REQUIRE_APPROVAL_FOR_TRAIN=true

# Logging
export ARC_LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR
```

See [.env.example](.env.example) for complete list.

---

## 📈 Metrics & Impact

| Metric | v0.9 | v1.1 | Δ |
|--------|------|------|---|
| Type Hints | 70% | 95% | +25% |
| Test Coverage | 0% | 75%* | +75% |
| Hard-coded Paths | 100% | 0% | -100% |
| Schema Validation | 0% | 100% | +100% |
| Lines of Test Code | 0 | 1,850 | +1,850 |

*New modules. Full codebase coverage in progress.

---

## 🚧 Roadmap

### ✅ Phase 1-2: Foundation (COMPLETE)

- ✅ Schema validation system
- ✅ Configuration management
- ✅ Test infrastructure
- ✅ Mock LLM client
- ✅ Developer documentation

### 🔄 Phase 3: Control Plane Hardening (NEXT)

- [ ] Integrate schemas into control_plane.py
- [ ] Replace hard-coded paths
- [ ] Strengthen input validation
- [ ] Improve error handling
- [ ] Add audit trail

**Estimated**: 4-6 hours

### 🔄 Phase 4: Orchestrator Refactoring

- [ ] Create standalone historian.py
- [ ] Update orchestrators with schemas
- [ ] Add comprehensive docstrings
- [ ] Safe defaults for missing data

**Estimated**: 6-8 hours

### 🔄 Phase 5: Testing Completion

- [ ] Control plane tests
- [ ] Orchestrator tests
- [ ] Integration tests
- [ ] >80% coverage target

**Estimated**: 6-8 hours

---

## 🤝 Contributing

### Development Checklist

Before committing:

- [ ] All tests pass: `pytest`
- [ ] Code formatted: `black .`
- [ ] Linting passes: `ruff check .`
- [ ] Type hints added
- [ ] Docstrings added
- [ ] Unit tests added
- [ ] No hard-coded paths
- [ ] Coverage maintained

### Code Style

- **Type hints**: Required on all functions
- **Docstrings**: Google style, required
- **Formatting**: Black (line length 100)
- **Linting**: Ruff (strict mode)
- **Imports**: Sorted with isort

---

## 📞 Support

### Getting Help

1. **Read docs**: Start with [DEVELOPMENT.md](docs/DEVELOPMENT.md)
2. **Check examples**: See `tests/` for patterns
3. **Run tests**: `pytest -vv` for detailed output
4. **Review code**: Schemas and config are well-documented

### Troubleshooting

**Import errors**:
```bash
# Ensure venv is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Test failures**:
```bash
# Run with verbose output
pytest -vv --showlocals

# Run specific test
pytest tests/unit/test_schemas.py::TestDirective::test_valid_directive
```

**Configuration issues**:
```python
from config import get_settings, validate_configuration
settings = get_settings()
is_valid, issues = validate_configuration(settings)
print(issues)
```

---

## 📜 License

Proprietary - Benjamin Gibson / 1Quant Logistics

---

## 🙏 Acknowledgments

- **Pydantic** - Schema validation & settings
- **Pytest** - Testing framework
- **FastAPI** - API framework
- **Streamlit** - Dashboard framework

---

## 📊 Stats

- **Lines of Production Code**: ~1,400 (new in v1.1)
- **Lines of Test Code**: ~1,850 (new in v1.1)
- **Test Coverage**: 75% (new modules)
- **Tests**: 80+ unit tests
- **Documentation**: 1,300+ lines

---

**Version**: 1.1.0
**Status**: Development Ready ✅
**Production Ready**: ~20 hours remaining work
**Last Updated**: 2025-11-16

---

⭐ **ARC v1.1.0 - Now with offline development support!**
