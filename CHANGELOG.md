# Changelog

All notable changes to MAGNET (Multi-Agent Naval Architecture Exploration Tool) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Supervisor Naval Agent for meta-learning and strategy adjustment
- Autonomous 5-cycle validation capability
- State persistence for orchestrator (resume capability)
- Experiment history flattening for agent compatibility

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
