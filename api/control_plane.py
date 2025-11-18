import os
import json
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
import uvicorn

# Import v1.1.0 infrastructure
from config import get_settings, ARCSettings
from memory_handler import get_memory_handler, MemoryHandler, ValidationFailedError, AtomicWriteError
from schemas import (
    Directive, HistorySummary, Constraints, SystemState,
    OperatingMode, ActiveExperiment
)
from tool_governance import get_tool_governance, ToolGovernance, ToolValidationError, ToolExecutionError

# Initialize settings, memory handler, and tool governance
settings = get_settings()
memory = get_memory_handler(settings)
governance = get_tool_governance(settings, memory)

# Configure logging with config-driven paths
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.logs_dir / 'control_plane.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title='ARC Control Plane', version='1.1.0')

# Models
class ExecRequest(BaseModel):
    command: str
    role: str
    cycle_id: int
    requires_approval: bool = True

class TrainRequest(BaseModel):
    experiment_id: str
    config: Dict[str, Any]
    requires_approval: bool = True

class StatusRequest(BaseModel):
    query: Optional[str] = None

class EvalRequest(BaseModel):
    experiment_id: str
    metrics: List[str]

class ArchiveRequest(BaseModel):
    cycle_id: int
    reason: str

class RollbackRequest(BaseModel):
    snapshot_id: str

# Helper functions
def load_system_state() -> SystemState:
    """Load system state with schema validation."""
    return memory.load_system_state()

def save_system_state(state: SystemState) -> None:
    """Save system state with schema validation."""
    memory.save_system_state(state)

def load_directive() -> Directive:
    """Load directive with schema validation."""
    return memory.load_directive()

def load_constraints() -> Constraints:
    """Load constraints with schema validation."""
    return memory.load_constraints()

def validate_command(command: str) -> bool:
    """Validate command against allowlist from config."""
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return False
    base_cmd = cmd_parts[0]
    return base_cmd in settings.allowed_commands

def check_mode_permission(action: str) -> bool:
    """Check if action is allowed in current mode."""
    state = load_system_state()
    mode = state.mode

    if mode == OperatingMode.SEMI:
        # All actions require approval in SEMI mode
        return True
    elif mode == OperatingMode.AUTO:
        # Auto mode allows most actions except training
        return action != 'train'
    elif mode == OperatingMode.FULL:
        # Full autonomy
        return True
    else:
        return False

# Endpoints
@app.get('/')
async def root():
    return {
        'service': 'ARC Control Plane',
        'version': '1.1.0',
        'status': 'operational'
    }

@app.get('/status')
async def get_status(query: Optional[str] = None):
    """Get system status with validated schema."""
    try:
        state = load_system_state()
        directive = load_directive()

        status = {
            'mode': state.mode.value,
            'arc_version': state.arc_version,
            'status': state.status,
            'last_cycle': state.last_cycle_timestamp,
            'active_experiments': [exp.dict() for exp in state.active_experiments],
            'current_cycle': directive.cycle_id,
            'current_objective': directive.objective.value
        }

        if query:
            # Filter status based on query
            filtered = {k: v for k, v in status.items() if query.lower() in k.lower()}
            return filtered if filtered else status

        return status
    except ValidationFailedError as e:
        logger.error(f'Schema validation failed: {e}')
        raise HTTPException(status_code=500, detail=f'Memory validation error: {str(e)}')
    except Exception as e:
        logger.error(f'Status check failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/exec')
async def execute_command(req: ExecRequest):
    """Execute command with safety validation and schema-validated logging."""
    try:
        logger.info(f'Exec request from {req.role}: {req.command}')

        # Check mode permission
        if not check_mode_permission('exec'):
            raise HTTPException(status_code=403, detail='Action not permitted in current mode')

        # Validate command
        if not validate_command(req.command):
            logger.warning(f'Command blocked: {req.command}')
            raise HTTPException(status_code=400, detail='Command not in allowlist')

        # In SEMI mode, always require approval
        state = load_system_state()
        if state.mode == OperatingMode.SEMI and req.requires_approval:
            return {
                'status': 'pending_approval',
                'command': req.command,
                'role': req.role,
                'cycle_id': req.cycle_id,
                'message': 'Command requires human approval in SEMI mode'
            }

        # Execute command
        result = subprocess.run(
            req.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )

        # Log execution (use config-driven path)
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'role': req.role,
            'cycle_id': req.cycle_id,
            'command': req.command,
            'returncode': result.returncode,
            'stdout': result.stdout[:1000],  # Truncate for logging
            'stderr': result.stderr[:1000]
        }

        log_file = settings.logs_dir / f'exec_cycle_{req.cycle_id}.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        return {
            'status': 'executed',
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail='Command execution timeout')
    except ValidationFailedError as e:
        logger.error(f'Schema validation failed: {e}')
        raise HTTPException(status_code=500, detail=f'Memory validation error: {str(e)}')
    except Exception as e:
        logger.error(f'Exec failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/train')
async def start_training(req: TrainRequest):
    """Start training job with constraint validation and schema-validated state updates."""
    try:
        logger.info(f'Train request for experiment {req.experiment_id}')

        # Check mode permission
        if not check_mode_permission('train'):
            raise HTTPException(status_code=403, detail='Training not permitted in current mode')

        # Load constraints with validation
        constraints = load_constraints()

        # Validate config against constraints
        validation_errors = []
        for param, value in req.config.items():
            for forbidden in constraints.forbidden_ranges:
                if forbidden.param == param:
                    if forbidden.min is not None and value < forbidden.min:
                        validation_errors.append(f'Parameter {param}={value} below safe range (min={forbidden.min})')
                    if forbidden.max is not None and value > forbidden.max:
                        validation_errors.append(f'Parameter {param}={value} above safe range (max={forbidden.max})')

        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={'error': 'validation_failed', 'issues': validation_errors}
            )

        # In SEMI mode, require approval
        state = load_system_state()
        if state.mode == OperatingMode.SEMI and req.requires_approval:
            return {
                'status': 'pending_approval',
                'experiment_id': req.experiment_id,
                'config': req.config,
                'message': 'Training requires human approval in SEMI mode'
            }

        # Add to active experiments using transaction
        with memory.transaction():
            new_experiment = ActiveExperiment(
                experiment_id=req.experiment_id,
                status='queued',
                started_at=datetime.now().isoformat()
            )
            state.active_experiments.append(new_experiment)
            save_system_state(state)

        return {
            'status': 'queued',
            'experiment_id': req.experiment_id,
            'message': 'Training job queued'
        }

    except ValidationFailedError as e:
        logger.error(f'Schema validation failed: {e}')
        raise HTTPException(status_code=500, detail=f'Memory validation error: {str(e)}')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Train request failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/eval')
async def evaluate_experiment(req: EvalRequest):
    """Evaluate experiment metrics using config-driven paths."""
    try:
        exp_dir = settings.experiments_dir / req.experiment_id
        if not exp_dir.exists():
            raise HTTPException(status_code=404, detail='Experiment not found')

        # Load experiment results
        results_path = exp_dir / 'results.json'
        if not results_path.exists():
            raise HTTPException(status_code=404, detail='Results not found')

        with open(results_path, 'r') as f:
            results = json.load(f)

        # Extract requested metrics
        metrics = {m: results.get(m) for m in req.metrics if m in results}

        return {
            'experiment_id': req.experiment_id,
            'metrics': metrics
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Eval failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/archive')
async def archive_cycle(req: ArchiveRequest):
    """Archive cycle data using memory handler's backup system."""
    try:
        # Create backup using memory handler
        backup_dir = memory.backup_memory()

        # Save metadata
        metadata = {
            'snapshot_id': backup_dir.name,
            'cycle_id': req.cycle_id,
            'reason': req.reason,
            'timestamp': datetime.now().isoformat()
        }

        with open(backup_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f'Archived cycle {req.cycle_id} to {backup_dir.name}')

        return {
            'status': 'archived',
            'snapshot_id': backup_dir.name
        }

    except Exception as e:
        logger.error(f'Archive failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/rollback')
async def rollback_to_snapshot(req: RollbackRequest):
    """Rollback to previous snapshot using memory handler's restore system."""
    try:
        snapshot_dir = settings.snapshots_dir / req.snapshot_id
        if not snapshot_dir.exists():
            raise HTTPException(status_code=404, detail='Snapshot not found')

        # Restore memory using memory handler
        memory.restore_memory(snapshot_dir)

        # Validate restored memory
        is_valid, errors = memory.validate_all_memory()
        if not is_valid:
            logger.error(f'Restored memory validation failed: {errors}')
            raise HTTPException(
                status_code=500,
                detail={'error': 'restore_validation_failed', 'issues': errors}
            )

        logger.info(f'Rolled back to snapshot {req.snapshot_id}')

        return {
            'status': 'rolled_back',
            'snapshot_id': req.snapshot_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Rollback failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/mode')
async def set_mode(mode: str):
    """Change ARC operating mode with schema validation."""
    try:
        # Validate mode using schema enum
        try:
            new_mode = OperatingMode(mode)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid mode. Must be one of: {[m.value for m in OperatingMode]}'
            )

        # Update state atomically
        with memory.transaction():
            state = load_system_state()
            old_mode = state.mode
            state.mode = new_mode
            save_system_state(state)

        logger.info(f'Mode changed: {old_mode.value} -> {new_mode.value}')

        return {
            'status': 'mode_changed',
            'old_mode': old_mode.value,
            'new_mode': new_mode.value
        }

    except ValidationFailedError as e:
        logger.error(f'Schema validation failed: {e}')
        raise HTTPException(status_code=500, detail=f'Memory validation error: {str(e)}')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Mode change failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    # Ensure directories exist using config
    settings.ensure_directories()

    # Start server
    uvicorn.run(app, host='0.0.0.0', port=8002, log_level='info')
