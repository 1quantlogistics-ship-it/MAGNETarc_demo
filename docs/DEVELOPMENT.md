# ARC Development Guide

Guide for local development on the ARC (Autonomous Research Collective) codebase **without requiring a running LLM backend or GPU**.

## Overview

ARC v1.1.0 is designed to support **offline development** with:
- **Mock LLM clients** for testing without vLLM/GPU
- **Configurable environments** (dev/test/prod)
- **Schema validation** for all memory files
- **Comprehensive test suite** with >70% coverage goal

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
cd /Users/bengibson/Desktop/ARC/arc_clean

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env for local development
export ARC_ENVIRONMENT=dev
export ARC_HOME=/Users/bengibson/Desktop/ARC/arc_test
export ARC_LLM_ENDPOINT=http://localhost:8000/v1
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_schemas.py

# Run unit tests only
pytest -m unit
```

## Project Structure

```
arc_clean/
├── api/
│   ├── control_plane.py          # FastAPI control plane
│   ├── cycle_orchestrator.py     # Orchestrator implementations
│   ├── dashboard.py              # Streamlit dashboard
│   └── training_stub.py          # Minimal training for testing
├── config.py                     # Configuration management (NEW)
├── schemas.py                    # Pydantic schemas for memory files (NEW)
├── tests/
│   ├── conftest.py               # Pytest fixtures (NEW)
│   ├── mocks/
│   │   ├── llm_client.py         # Mock LLM client (NEW)
│   │   └── fixtures.py           # Role-based response fixtures (NEW)
│   ├── unit/
│   │   ├── test_schemas.py       # Schema tests (NEW)
│   │   └── test_config.py        # Config tests (NEW)
│   └── integration/
├── docs/
│   ├── ARCHITECTURE.md           # System architecture
│   ├── DEVELOPMENT.md            # This file
│   └── TESTING.md                # Testing guide
├── requirements.txt              # Dependencies (UPDATED)
├── pytest.ini                    # Pytest configuration (NEW)
├── .coveragerc                   # Coverage configuration (NEW)
└── .env.example                  # Environment template (NEW)
```

## Development Workflow

### Working on ARC Without LLM Backend

The key principle: **All LLM interactions are abstracted and mockable**.

#### 1. Use Mock LLM Client

```python
from tests.mocks.llm_client import MockLLMClient, MockMode

# Create mock client for Historian role
client = MockLLMClient(role="historian", mode=MockMode.SUCCESS)

# Simulate LLM call
response = client.chat_completion([
    {"role": "user", "content": "Analyze experiment history"}
])

# Response contains realistic Historian JSON output
print(response["choices"][0]["message"]["content"])
```

#### 2. Use Test Settings

```python
from config import get_test_settings

# Get test configuration (uses temp directories)
settings = get_test_settings()

# All paths are in /tmp or equivalent
print(settings.memory_dir)  # /tmp/arc_test/memory
print(settings.experiments_dir)  # /tmp/arc_test/experiments
```

#### 3. Validate Memory Files

```python
from schemas import Directive, DirectiveMode, Objective, NoveltyBudget
from schemas import validate_memory_file, save_memory_file

# Create validated directive
directive = Directive(
    cycle_id=1,
    mode=DirectiveMode.EXPLORE,
    objective=Objective.IMPROVE_AUC,
    novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1)
)

# Save with schema validation
save_memory_file("/path/to/directive.json", directive, atomic=True)

# Load with schema validation
loaded = validate_memory_file("/path/to/directive.json", Directive)
```

### Code Quality Standards

#### Type Hints
**Required** on all new functions:

```python
from typing import Dict, List, Optional
from pathlib import Path

def load_experiment_config(
    experiment_id: str,
    config_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Load experiment configuration."""
    ...
```

#### Docstrings
**Required** on all new functions (Google style):

```python
def validate_training_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate training configuration against constraints.

    Args:
        config: Training configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_issues)

    Raises:
        ValueError: If config is None or empty

    Example:
        >>> config = {"learning_rate": 0.001}
        >>> is_valid, issues = validate_training_config(config)
        >>> assert is_valid
    """
    ...
```

#### Error Handling
Use **specific exceptions**, not broad `except Exception`:

```python
# Bad
try:
    data = json.load(f)
except Exception as e:
    logger.error(f"Error: {e}")

# Good
try:
    data = json.load(f)
except FileNotFoundError:
    logger.error(f"Config file not found: {path}")
    raise
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in {path}: {e}")
    raise
except PermissionError:
    logger.error(f"Permission denied reading {path}")
    raise
```

### Testing Guidelines

#### Writing Unit Tests

```python
import pytest
from schemas import Directive, DirectiveMode, Objective, NoveltyBudget

@pytest.mark.unit
class TestDirective:
    """Test Directive schema validation."""

    def test_valid_directive(self):
        """Test creating valid directive."""
        directive = Directive(
            cycle_id=1,
            mode=DirectiveMode.EXPLORE,
            objective=Objective.IMPROVE_AUC,
            novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1)
        )
        assert directive.cycle_id == 1

    def test_negative_cycle_fails(self):
        """Test that negative cycle_id is rejected."""
        with pytest.raises(ValidationError):
            Directive(
                cycle_id=-1,
                mode=DirectiveMode.EXPLORE,
                objective=Objective.IMPROVE_AUC,
                novelty_budget=NoveltyBudget(exploit=3, explore=2, wildcat=1)
            )
```

#### Using Fixtures

```python
import pytest

def test_orchestrator_with_mock_llm(test_settings, memory_files):
    """Test orchestrator using fixtures."""
    # test_settings provides temp config
    # memory_files provides pre-created valid memory files

    assert test_settings.memory_dir.exists()
    assert memory_files['directive'].exists()
```

#### Mocking LLM Calls

```python
from tests.mocks.llm_client import mock_llm_request, MockMode

def test_historian_update(monkeypatch):
    """Test historian update with mock LLM."""
    # Patch requests.post to use mock
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: mock_llm_request(
            *args, role="historian", mode=MockMode.SUCCESS, **kwargs
        )
    )

    # Now run historian update - it will use mock
    result = historian_update()
    assert result is not None
```

## Common Development Tasks

### Task 1: Add New Memory File Schema

1. **Define Pydantic model in `schemas.py`:**

```python
class MyNewSchema(BaseModel):
    """Description of this schema."""
    field1: str = Field(description="...")
    field2: int = Field(ge=0, description="...")
```

2. **Add to `create_default_memory_files()` if needed**

3. **Write unit tests in `tests/unit/test_schemas.py`**

4. **Update documentation**

### Task 2: Add New Configuration Option

1. **Add field to `ARCSettings` in `config.py`:**

```python
class ARCSettings(BaseSettings):
    my_new_option: int = Field(
        default=42,
        ge=0,
        description="Description of option"
    )
```

2. **Add to `.env.example`:**

```bash
ARC_MY_NEW_OPTION=42
```

3. **Write unit tests in `tests/unit/test_config.py`**

4. **Update validation if needed**

### Task 3: Add New Orchestrator Logic

1. **Write function with type hints and docstring**

2. **Use `get_settings()` for configuration:**

```python
from config import get_settings

def my_orchestrator_function():
    """Orchestrate a new workflow."""
    settings = get_settings()
    memory_dir = settings.memory_dir
    ...
```

3. **Write unit tests with mocked LLM:**

```python
def test_my_orchestrator(monkeypatch, test_settings, memory_files):
    """Test new orchestrator."""
    # Mock LLM calls
    # Run orchestrator
    # Assert results
```

### Task 4: Improve Error Handling

1. **Identify broad exception handlers:**

```bash
grep -n "except Exception" api/*.py
```

2. **Replace with specific exceptions:**

```python
# Before
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Error: {e}")

# After
try:
    result = risky_operation()
except TimeoutError:
    logger.error("Operation timed out")
    raise
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

3. **Add tests for error paths**

## Environment Variables Reference

See `.env.example` for complete list. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARC_ENVIRONMENT` | `dev` | Environment: dev, test, prod |
| `ARC_HOME` | `/workspace/arc` | Root directory |
| `ARC_MODE` | `SEMI` | Operating mode |
| `ARC_LLM_ENDPOINT` | `http://localhost:8000/v1` | LLM API URL |
| `ARC_LLM_TIMEOUT` | `120` | LLM timeout (seconds) |
| `ARC_LOG_LEVEL` | `INFO` | Logging level |

## Code Quality Tools

### Linting (Ruff)

```bash
# Check code
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Formatting (Black)

```bash
# Format code
black .

# Check formatting
black --check .
```

### Type Checking (MyPy)

```bash
# Run type checker
mypy .
```

### Coverage

```bash
# Generate coverage report
pytest --cov=. --cov-report=html

# View report
open htmlcov/index.html
```

## Debugging Tips

### Debug Test Failures

```bash
# Run with verbose output
pytest -vv tests/unit/test_schemas.py

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest --showlocals
```

### Debug Configuration

```python
from config import get_settings, validate_configuration

settings = get_settings()
print(settings.to_dict())

is_valid, issues = validate_configuration(settings)
if not is_valid:
    for issue in issues:
        print(f"❌ {issue}")
```

### Debug Memory Files

```python
from schemas import validate_memory_file, Directive

try:
    directive = validate_memory_file("/path/to/directive.json", Directive)
    print(directive.model_dump_json(indent=2))
except ValidationError as e:
    print(f"Validation errors:\n{e}")
```

## Performance Considerations

- **Use test settings for fast iteration** (temp dirs, short timeouts)
- **Mock LLM calls** to avoid network overhead
- **Use pytest markers** to run only relevant tests: `pytest -m unit`
- **Parallel test execution**: `pytest -n auto` (requires `pytest-xdist`)

## Contribution Checklist

Before committing:

- [ ] All tests pass: `pytest`
- [ ] Code is formatted: `black .`
- [ ] Linting passes: `ruff check .`
- [ ] Type hints added to new functions
- [ ] Docstrings added to new functions
- [ ] Unit tests added for new features
- [ ] No hard-coded paths (use `config.py`)
- [ ] No broad `except Exception` handlers
- [ ] Coverage maintained or improved

## Troubleshooting

### Issue: Import errors

**Solution**: Ensure you're in the project root and venv is activated:

```bash
cd /Users/bengibson/Desktop/ARC/arc_clean
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Test failures with path errors

**Solution**: Use `test_settings` fixture which provides temp paths:

```python
def test_my_feature(test_settings):
    # Use test_settings.memory_dir instead of hard-coded paths
    assert test_settings.memory_dir.exists()
```

### Issue: Pydantic validation errors

**Solution**: Check schema definition and ensure all required fields are provided:

```python
from pydantic import ValidationError

try:
    obj = MySchema(**data)
except ValidationError as e:
    print(e.json())  # Detailed error info
```

## Next Steps

1. **Read [TESTING.md](TESTING.md)** for detailed testing guide
2. **Read [ARCHITECTURE.md](ARCHITECTURE.md)** for system design
3. **Explore examples** in `tests/` directory
4. **Run the test suite** to verify setup

## Getting Help

- Check existing tests for examples
- Review Pydantic documentation for schema questions
- Review pytest documentation for testing patterns
- Check TODO comments in code for known issues

---

**Happy coding!** Remember: All development can be done offline without GPU or LLM backend.
