import os
import json
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/arc/logs/control_plane.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
MEMORY_DIR = '/workspace/arc/memory'
EXPERIMENTS_DIR = '/workspace/arc/experiments'
LOGS_DIR = '/workspace/arc/logs'
SNAPSHOTS_DIR = '/workspace/arc/snapshots'

# Command allowlist for safety
ALLOWED_COMMANDS = [
    'python', 'pip', 'conda', 'git', 'nvidia-smi',
    'ls', 'cat', 'head', 'tail', 'grep', 'find',
    'mkdir', 'cp', 'mv', 'rm', 'chmod'
]

# FastAPI app
app = FastAPI(title='ARC Control Plane', version='0.8.0')

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
def load_system_state() -> Dict[str, Any]:
    path = os.path.join(MEMORY_DIR, 'system_state.json')
    with open(path, 'r') as f:
        return json.load(f)

def save_system_state(state: Dict[str, Any]):
    path = os.path.join(MEMORY_DIR, 'system_state.json')
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)

def load_directive() -> Dict[str, Any]:
    path = os.path.join(MEMORY_DIR, 'directive.json')
    with open(path, 'r') as f:
        return json.load(f)

def load_constraints() -> Dict[str, Any]:
    path = os.path.join(MEMORY_DIR, 'constraints.json')
    with open(path, 'r') as f:
        return json.load(f)

def validate_command(command: str) -> bool:
    '''Validate command against allowlist'''
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return False
    base_cmd = cmd_parts[0]
    return base_cmd in ALLOWED_COMMANDS

def check_mode_permission(action: str) -> bool:
    '''Check if action is allowed in current mode'''
    state = load_system_state()
    mode = state.get('mode', 'SEMI')
    
    if mode == 'SEMI':
        # All actions require approval in SEMI mode
        return True
    elif mode == 'AUTO':
        # Auto mode allows most actions except training
        return action != 'train'
    elif mode == 'FULL':
        # Full autonomy
        return True
    else:
        return False

# Endpoints
@app.get('/')
async def root():
    return {
        'service': 'ARC Control Plane',
        'version': '0.8.0',
        'status': 'operational'
    }

@app.get('/status')
async def get_status(query: Optional[str] = None):
    '''Get system status'''
    try:
        state = load_system_state()
        directive = load_directive()
        
        status = {
            'mode': state.get('mode'),
            'arc_version': state.get('arc_version'),
            'status': state.get('status'),
            'last_cycle': state.get('last_cycle_timestamp'),
            'active_experiments': state.get('active_experiments', []),
            'current_cycle': directive.get('cycle_id'),
            'current_objective': directive.get('objective')
        }
        
        if query:
            # Filter status based on query
            filtered = {k: v for k, v in status.items() if query.lower() in k.lower()}
            return filtered if filtered else status
        
        return status
    except Exception as e:
        logger.error(f'Status check failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/exec')
async def execute_command(req: ExecRequest):
    '''Execute command with safety validation'''
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
        if state.get('mode') == 'SEMI' and req.requires_approval:
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
        
        # Log execution
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'role': req.role,
            'cycle_id': req.cycle_id,
            'command': req.command,
            'returncode': result.returncode,
            'stdout': result.stdout[:1000],  # Truncate for logging
            'stderr': result.stderr[:1000]
        }
        
        with open(os.path.join(LOGS_DIR, f'exec_cycle_{req.cycle_id}.jsonl'), 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return {
            'status': 'executed',
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail='Command execution timeout')
    except Exception as e:
        logger.error(f'Exec failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/train')
async def start_training(req: TrainRequest):
    '''Start training job with approval check'''
    try:
        logger.info(f'Train request for experiment {req.experiment_id}')
        
        # Check mode permission
        if not check_mode_permission('train'):
            raise HTTPException(status_code=403, detail='Training not permitted in current mode')
        
        # Load constraints
        constraints = load_constraints()
        
        # Validate config against constraints
        for param, value in req.config.items():
            for forbidden in constraints.get('forbidden_ranges', []):
                if forbidden.get('param') == param:
                    min_val = forbidden.get('min')
                    max_val = forbidden.get('max')
                    if min_val is not None and value < min_val:
                        raise HTTPException(status_code=400, detail=f'Parameter {param} below safe range')
                    if max_val is not None and value > max_val:
                        raise HTTPException(status_code=400, detail=f'Parameter {param} above safe range')
        
        # In SEMI mode, require approval
        state = load_system_state()
        if state.get('mode') == 'SEMI' and req.requires_approval:
            return {
                'status': 'pending_approval',
                'experiment_id': req.experiment_id,
                'config': req.config,
                'message': 'Training requires human approval in SEMI mode'
            }
        
        # Add to active experiments
        state['active_experiments'] = state.get('active_experiments', [])
        state['active_experiments'].append({
            'experiment_id': req.experiment_id,
            'config': req.config,
            'status': 'queued',
            'submitted_at': datetime.now().isoformat()
        })
        save_system_state(state)
        
        return {
            'status': 'queued',
            'experiment_id': req.experiment_id,
            'message': 'Training job queued'
        }
        
    except Exception as e:
        logger.error(f'Train request failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/eval')
async def evaluate_experiment(req: EvalRequest):
    '''Evaluate experiment metrics'''
    try:
        exp_dir = os.path.join(EXPERIMENTS_DIR, req.experiment_id)
        if not os.path.exists(exp_dir):
            raise HTTPException(status_code=404, detail='Experiment not found')
        
        # Load experiment results
        results_path = os.path.join(exp_dir, 'results.json')
        if not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail='Results not found')
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Extract requested metrics
        metrics = {m: results.get(m) for m in req.metrics if m in results}
        
        return {
            'experiment_id': req.experiment_id,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f'Eval failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/archive')
async def archive_cycle(req: ArchiveRequest):
    '''Archive cycle data to snapshots'''
    try:
        snapshot_id = f'cycle_{req.cycle_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        snapshot_dir = os.path.join(SNAPSHOTS_DIR, snapshot_id)
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Copy memory files
        for fname in ['directive.json', 'history_summary.json', 'constraints.json', 'system_state.json']:
            src = os.path.join(MEMORY_DIR, fname)
            dst = os.path.join(snapshot_dir, fname)
            if os.path.exists(src):
                subprocess.run(['cp', src, dst], check=True)
        
        # Save metadata
        metadata = {
            'snapshot_id': snapshot_id,
            'cycle_id': req.cycle_id,
            'reason': req.reason,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(snapshot_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f'Archived cycle {req.cycle_id} to {snapshot_id}')
        
        return {
            'status': 'archived',
            'snapshot_id': snapshot_id
        }
        
    except Exception as e:
        logger.error(f'Archive failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/rollback')
async def rollback_to_snapshot(req: RollbackRequest):
    '''Rollback to previous snapshot'''
    try:
        snapshot_dir = os.path.join(SNAPSHOTS_DIR, req.snapshot_id)
        if not os.path.exists(snapshot_dir):
            raise HTTPException(status_code=404, detail='Snapshot not found')
        
        # Restore memory files
        for fname in ['directive.json', 'history_summary.json', 'constraints.json', 'system_state.json']:
            src = os.path.join(snapshot_dir, fname)
            dst = os.path.join(MEMORY_DIR, fname)
            if os.path.exists(src):
                subprocess.run(['cp', src, dst], check=True)
        
        logger.info(f'Rolled back to snapshot {req.snapshot_id}')
        
        return {
            'status': 'rolled_back',
            'snapshot_id': req.snapshot_id
        }
        
    except Exception as e:
        logger.error(f'Rollback failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/mode')
async def set_mode(mode: str):
    '''Change ARC operating mode'''
    try:
        if mode not in ['SEMI', 'AUTO', 'FULL']:
            raise HTTPException(status_code=400, detail='Invalid mode. Must be SEMI, AUTO, or FULL')
        
        state = load_system_state()
        old_mode = state.get('mode')
        state['mode'] = mode
        save_system_state(state)
        
        logger.info(f'Mode changed: {old_mode} -> {mode}')
        
        return {
            'status': 'mode_changed',
            'old_mode': old_mode,
            'new_mode': mode
        }
        
    except Exception as e:
        logger.error(f'Mode change failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Start server
    uvicorn.run(app, host='0.0.0.0', port=8002, log_level='info')
