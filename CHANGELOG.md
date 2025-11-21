# Changelog

All notable changes to MAGNET (Multi-Agent Naval Architecture Exploration Tool) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added - v0.2.0
- **Synthetic Data Generation System**:
  - Physics-informed training data generation for ML models
  - Multiple sampling strategies (Latin Hypercube, Gaussian, Edge/Corner, Mixed)
  - Data augmentation with controlled noise (2-5x multiplier)
  - Quality metrics (diversity score, parameter coverage, validity ratio)
  - Multi-format export (NumPy, CSV, JSON)
  - Comprehensive 450-line README with usage examples
  - Full test suite with 100% pass rate
- **Multi-Domain Design System** (design_core package):
  - Domain-agnostic design abstractions
  - BaseDesignParameters and BasePhysicsEngine interfaces
  - Universal performance metrics (structural, efficiency, safety)
  - Support for naval, aerial, ground vehicle, and structural domains

### Added - v0.1.0
- Supervisor Naval Agent for meta-learning and strategy adjustment
- Autonomous 5-cycle validation capability
- State persistence for orchestrator (resume capability)
- Experiment history flattening for agent compatibility
- **Enhanced CLI Features** (TASK 2.3):
  - `--visualize` flag for automatic visualization generation
  - `--export-html PATH` for custom dashboard location
  - `--auto-open` flag to automatically open dashboard in browser
  - `--metrics-report` flag for detailed performance summary
  - `--resume STATE_FILE` for continuing previous research sessions
  - `--save-state-every N` for configurable state persistence
  - `--watch` mode for continuous autonomous operation
  - Comprehensive CLI_GUIDE.md user documentation

### Fixed
- **Critical**: Fixed baseline design bug causing NoneType error in first cycle
- **Critical**: Fixed data format mismatch between knowledge base and agents
  - Knowledge base stores nested format: `{cycle, designs: [...], results: [...]}`
  - Agents expect flat format: `[{parameters: {...}, results: {...}}]`
  - Added `_flatten_experiments()` helper to orchestrator
  - Applied fix to all agent calls (explorer, critic, historian, supervisor)
- Configuration module pydantic dependency (migrated to dataclass)
- Logger infinite recursion in orchestrator

### Changed
- Improved orchestrator error handling and logging
- Enhanced knowledge base experiment tracking

## [v0.0.1] - 2025-01-XX

### Added
- Initial MAGNET architecture
- Explorer Naval Agent for hypothesis generation
- Experimental Architect Agent for design generation
- Critic Naval Agent for design validation and analysis
- Historian Naval Agent for pattern recognition and learning
- Knowledge Base for persistent experiment storage
- Mock physics simulation for CPU-only testing
- 7-step autonomous research cycle orchestration
- Integration test suite

### Documentation
- README.md with setup and usage instructions
- Agent architecture documentation
- API documentation for core components
