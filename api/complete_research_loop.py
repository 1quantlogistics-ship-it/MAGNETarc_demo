#!/usr/bin/env python3
"""
ARC Complete Research Loop
Reasoning (Director/Architect/Critic) → Training (Executor) → Learning (Historian)
"""

import os
import sys
import json
import subprocess
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_complete_cycle(cycle_id: int):
    """Execute complete ARC research cycle"""
    logger.info(f'========== COMPLETE RESEARCH LOOP: CYCLE {cycle_id} ==========')
    
    # Phase 1: Reasoning (Director/Architect/Critic)
    logger.info('[PHASE 1] Running reasoning loop (Director/Architect/Critic)...')
    result = subprocess.run(
        ['python', '/workspace/arc/api/full_cycle_orchestrator.py', str(cycle_id)],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode != 0:
        logger.error(f'Reasoning loop failed: {result.stderr}')
        return None
    
    reasoning_output = json.loads(result.stdout)
    logger.info(f'[PHASE 1] Complete - {reasoning_output["approved"]} experiments approved')
    
    # Phase 2: Training (Executor)
    logger.info('[PHASE 2] Running training execution...')
    result = subprocess.run(
        ['python', '/workspace/arc/api/training_cycle_orchestrator.py', str(cycle_id)],
        capture_output=True,
        text=True,
        timeout=600
    )
    
    if result.returncode != 0:
        logger.error(f'Training execution failed: {result.stderr}')
        return None
    
    training_output = json.loads(result.stdout)
    logger.info(f'[PHASE 2] Complete - {training_output["experiments_run"]} experiments trained')
    
    # Phase 3: Results summary
    best_auc = training_output.get('best_auc', 0)
    logger.info(f'[PHASE 3] Learning complete - Best AUC: {best_auc:.4f}')
    
    summary = {
        'cycle_id': cycle_id,
        'reasoning': {
            'proposals': reasoning_output['proposals'],
            'approved': reasoning_output['approved'],
            'rejected': reasoning_output['rejected']
        },
        'training': {
            'experiments_run': training_output['experiments_run'],
            'results': training_output['results']
        },
        'learning': {
            'best_auc': best_auc
        },
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f'========== CYCLE {cycle_id} COMPLETE ==========')
    return summary

if __name__ == '__main__':
    cycle_id = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    result = run_complete_cycle(cycle_id)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print('{"status": "failed"}')
        sys.exit(1)
