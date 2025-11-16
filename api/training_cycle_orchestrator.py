#!/usr/bin/env python3
"""
ARC Training Cycle Orchestrator - Full cycle with real training execution
"""

import os
import json
import requests
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/arc/logs/training_cycle.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MEMORY_DIR = '/workspace/arc/memory'
EXPERIMENTS_DIR = '/workspace/arc/experiments'
LLM_ENDPOINT = 'http://localhost:8000/generate'

class TrainingCycleOrchestrator:
    def __init__(self):
        self.memory_dir = MEMORY_DIR
        self.experiments_dir = EXPERIMENTS_DIR
        self.llm_endpoint = LLM_ENDPOINT
        
    def load_memory(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.memory_dir, filename)
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_memory(self, filename: str, data: Dict[str, Any]):
        path = os.path.join(self.memory_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        try:
            response = requests.post(
                self.llm_endpoint,
                json={'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0.7},
                timeout=120
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('text', [''])[0]
            return ''
        except Exception as e:
            logger.error(f'LLM call error: {e}')
            return ''
    
    def extract_json(self, response: str) -> Dict[str, Any]:
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            return {}
        except Exception as e:
            logger.error(f'JSON extraction error: {e}')
            return {}
    
    def historian_update(self, cycle_id: int, new_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f'[HISTORIAN] Updating history for cycle {cycle_id}')
        history = self.load_memory('history_summary.json')
        
        if new_results:
            # Update with real results from training
            history['total_experiments'] = history.get('total_experiments', 0) + len(new_results)
            
            # Update best metrics
            for result in new_results:
                auc = result.get('auc')
                sensitivity = result.get('sensitivity')
                specificity = result.get('specificity')
                
                if auc and (history['best_metrics']['auc'] is None or auc > history['best_metrics']['auc']):
                    history['best_metrics']['auc'] = auc
                if sensitivity and (history['best_metrics']['sensitivity'] is None or sensitivity > history['best_metrics']['sensitivity']):
                    history['best_metrics']['sensitivity'] = sensitivity
                if specificity and (history['best_metrics']['specificity'] is None or specificity > history['best_metrics']['specificity']):
                    history['best_metrics']['specificity'] = specificity
            
            # Add to recent experiments
            history['recent_experiments'] = history.get('recent_experiments', [])
            for result in new_results:
                history['recent_experiments'].append({
                    'experiment_id': result.get('experiment_id'),
                    'auc': result.get('auc'),
                    'training_time': result.get('training_time'),
                    'timestamp': result.get('timestamp')
                })
            
            # Keep only last 10
            history['recent_experiments'] = history['recent_experiments'][-10:]
        
        history['total_cycles'] = cycle_id + 1
        
        self.save_memory('history_summary.json', history)
        logger.info(f'[HISTORIAN] Updated with {len(new_results) if new_results else 0} new results')
        return history
    
    def executor_train(self, cycle_id: int, approved: List[str], proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f'[EXECUTOR] Training {len(approved)} approved experiments')
        
        results = []
        for exp_id in approved:
            proposal = next((p for p in proposals if p['experiment_id'] == exp_id), None)
            if not proposal:
                continue
            
            exp_dir = os.path.join(self.experiments_dir, exp_id)
            
            # Execute training
            logger.info(f'[EXECUTOR] Launching training for {exp_id}')
            try:
                cmd = [
                    'python', '/workspace/arc/api/training_stub.py', exp_id
                ]
                result = subprocess.run(
                    cmd,
                    cwd='/workspace/arc/api',
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
                )
                
                if result.returncode == 0:
                    # Load results
                    results_path = os.path.join(exp_dir, 'results.json')
                    if os.path.exists(results_path):
                        with open(results_path, 'r') as f:
                            metrics = json.load(f)
                        results.append(metrics)
                        logger.info(f'[EXECUTOR] Training complete for {exp_id}: AUC={metrics.get("auc", 0):.4f}')
                else:
                    logger.error(f'[EXECUTOR] Training failed for {exp_id}: {result.stderr}')
            except Exception as e:
                logger.error(f'[EXECUTOR] Training error for {exp_id}: {e}')
        
        return results
    
    def run_training_cycle(self, cycle_id: int = 0):
        logger.info(f'===== TRAINING CYCLE {cycle_id} START =====')
        
        # Load proposals and reviews from previous cycle
        proposals_data = self.load_memory('proposals.json')
        reviews_data = self.load_memory('reviews.json')
        
        proposals = proposals_data.get('proposals', [])
        approved = reviews_data.get('approved', [])
        
        if not approved:
            logger.warning('No approved experiments to train')
            return {
                'cycle_id': cycle_id,
                'experiments_run': 0,
                'results': [],
                'message': 'No approved experiments'
            }
        
        # Execute training
        training_results = self.executor_train(cycle_id, approved, proposals)
        
        # Update Historian with real results
        history = self.historian_update(cycle_id, training_results)
        
        # Update system state
        system_state = self.load_memory('system_state.json')
        system_state['last_cycle_timestamp'] = datetime.now().isoformat()
        system_state['status'] = 'training_complete'
        self.save_memory('system_state.json', system_state)
        
        logger.info(f'===== TRAINING CYCLE {cycle_id} COMPLETE =====')
        
        return {
            'cycle_id': cycle_id,
            'experiments_run': len(training_results),
            'results': training_results,
            'best_auc': history['best_metrics']['auc'],
            'timestamp': datetime.now().isoformat()
        }

if __name__ == '__main__':
    import sys
    orchestrator = TrainingCycleOrchestrator()
    cycle_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    result = orchestrator.run_training_cycle(cycle_id)
    print(json.dumps(result, indent=2))
