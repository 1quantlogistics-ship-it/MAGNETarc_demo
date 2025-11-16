#!/usr/bin/env python3
"""
ARC Full Cycle Orchestrator - Includes all roles
Historian -> Director -> Architect -> Critic -> Executor (dry-run)
"""

import os
import json
import requests
import logging
from datetime import datetime
from typing import Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/arc/logs/full_cycle.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MEMORY_DIR = '/workspace/arc/memory'
EXPERIMENTS_DIR = '/workspace/arc/experiments'
LLM_ENDPOINT = 'http://localhost:8000/generate'

class FullCycleOrchestrator:
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
            else:
                logger.error(f'LLM call failed: {response.status_code}')
                return ''
        except Exception as e:
            logger.error(f'LLM call error: {e}')
            return ''
    
    def extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling think tags"""
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
    
    def historian_update(self, cycle_id: int) -> Dict[str, Any]:
        logger.info(f'[HISTORIAN] Running for cycle {cycle_id}')
        history = self.load_memory('history_summary.json')
        
        prompt = f"""You are the ARC Historian. Update the research history for cycle {cycle_id}.

Current history: {json.dumps(history, indent=2)}

Output ONLY valid JSON:
{{
  "total_cycles": {cycle_id},
  "total_experiments": 0,
  "best_metrics": {{"auc": null}},
  "recent_experiments": [],
  "failed_configs": [],
  "successful_patterns": []
}}"""
        
        response = self.call_llm(prompt, 500)
        updated = self.extract_json(response)
        
        if updated:
            self.save_memory('history_summary.json', updated)
            logger.info('[HISTORIAN] Update complete')
            return updated
        return history
    
    def director_directive(self, cycle_id: int) -> Dict[str, Any]:
        logger.info(f'[DIRECTOR] Creating directive for cycle {cycle_id+1}')
        history = self.load_memory('history_summary.json')
        
        prompt = f"""You are the ARC Director. Create directive for cycle {cycle_id+1}.

History: {json.dumps(history, indent=2)}

Output ONLY valid JSON:
{{
  "cycle_id": {cycle_id+1},
  "mode": "explore",
  "objective": "improve_model_performance",
  "novelty_budget": {{"exploit": 3, "explore": 2, "wildcat": 0}},
  "focus_areas": ["hyperparameters", "architecture"],
  "forbidden_axes": [],
  "encouraged_axes": ["learning_rate", "batch_size"],
  "notes": "Focus on safe improvements"
}}"""
        
        response = self.call_llm(prompt, 500)
        directive = self.extract_json(response)
        
        if directive:
            directive['cycle_id'] = cycle_id + 1
            self.save_memory('directive.json', directive)
            logger.info('[DIRECTOR] Directive complete')
            return directive
        return self.load_memory('directive.json')
    
    def architect_propose(self, cycle_id: int) -> List[Dict[str, Any]]:
        logger.info(f'[ARCHITECT] Generating proposals for cycle {cycle_id}')
        directive = self.load_memory('directive.json')
        history = self.load_memory('history_summary.json')
        
        prompt = f"""You are the ARC Architect. Generate 3 experiment proposals.

Directive: {json.dumps(directive, indent=2)}
History: {json.dumps(history, indent=2)}

Output ONLY valid JSON array:
[
  {{
    "experiment_id": "exp_{cycle_id}_001",
    "description": "Baseline configuration",
    "config": {{"learning_rate": 0.001, "batch_size": 32}},
    "novelty_type": "exploit",
    "rationale": "Establish baseline metrics"
  }},
  {{
    "experiment_id": "exp_{cycle_id}_002",
    "description": "Increased batch size",
    "config": {{"learning_rate": 0.001, "batch_size": 64}},
    "novelty_type": "explore",
    "rationale": "Test batch size impact"
  }},
  {{
    "experiment_id": "exp_{cycle_id}_003",
    "description": "Higher learning rate",
    "config": {{"learning_rate": 0.01, "batch_size": 32}},
    "novelty_type": "explore",
    "rationale": "Faster convergence"
  }}
]"""
        
        response = self.call_llm(prompt, 1000)
        proposals = self.extract_json(response)
        
        if isinstance(proposals, dict):
            proposals = []
        
        if not proposals:
            proposals = [{
                'experiment_id': f'exp_{cycle_id}_001',
                'description': 'Default baseline',
                'config': {'learning_rate': 0.001, 'batch_size': 32},
                'novelty_type': 'exploit',
                'rationale': 'Fallback baseline configuration'
            }]
        
        proposal_data = {
            'cycle_id': cycle_id,
            'proposals': proposals,
            'timestamp': datetime.now().isoformat()
        }
        self.save_memory('proposals.json', proposal_data)
        logger.info(f'[ARCHITECT] Generated {len(proposals)} proposals')
        return proposals
    
    def critic_review(self, cycle_id: int, proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f'[CRITIC] Reviewing {len(proposals)} proposals')
        constraints = self.load_memory('constraints.json')
        
        prompt = f"""You are the ARC Critic. Review these proposals.

Proposals: {json.dumps(proposals, indent=2)}
Constraints: {json.dumps(constraints, indent=2)}

Output ONLY valid JSON:
{{
  "cycle_id": {cycle_id},
  "reviews": [
    {{"experiment_id": "exp_{cycle_id}_001", "decision": "approve", "reasoning": "Safe baseline"}},
    {{"experiment_id": "exp_{cycle_id}_002", "decision": "approve", "reasoning": "Moderate risk"}},
    {{"experiment_id": "exp_{cycle_id}_003", "decision": "reject", "reasoning": "Learning rate too high"}}
  ],
  "approved": ["exp_{cycle_id}_001", "exp_{cycle_id}_002"],
  "rejected": ["exp_{cycle_id}_003"],
  "timestamp": "{datetime.now().isoformat()}"
}}"""
        
        response = self.call_llm(prompt, 1000)
        reviews = self.extract_json(response)
        
        if not reviews:
            reviews = {
                'cycle_id': cycle_id,
                'reviews': [{'experiment_id': p['experiment_id'], 'decision': 'approve', 'reasoning': 'Default approval'} for p in proposals],
                'approved': [p['experiment_id'] for p in proposals],
                'rejected': [],
                'timestamp': datetime.now().isoformat()
            }
        
        self.save_memory('reviews.json', reviews)
        logger.info(f"[CRITIC] Approved: {len(reviews.get('approved', []))}, Rejected: {len(reviews.get('rejected', []))}")
        return reviews
    
    def executor_dryrun(self, cycle_id: int, approved: List[str], proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f'[EXECUTOR] Dry-run for {len(approved)} approved experiments')
        
        results = []
        for exp_id in approved:
            proposal = next((p for p in proposals if p['experiment_id'] == exp_id), None)
            if not proposal:
                continue
            
            exp_dir = os.path.join(self.experiments_dir, exp_id)
            os.makedirs(exp_dir, exist_ok=True)
            
            metadata = {
                'experiment_id': exp_id,
                'cycle_id': cycle_id,
                'description': proposal['description'],
                'config': proposal['config'],
                'novelty_type': proposal['novelty_type'],
                'status': 'dry_run',
                'timestamp': datetime.now().isoformat()
            }
            
            with open(os.path.join(exp_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            results.append({
                'experiment_id': exp_id,
                'status': 'queued',
                'directory': exp_dir
            })
            
            logger.info(f'[EXECUTOR] Created dry-run for {exp_id}')
        
        return {
            'cycle_id': cycle_id,
            'executed': results,
            'mode': 'dry_run',
            'timestamp': datetime.now().isoformat()
        }
    
    def run_full_cycle(self, cycle_id: int = 0):
        logger.info(f'===== FULL CYCLE {cycle_id} START =====')
        
        # Phase 1: Historian
        history = self.historian_update(cycle_id)
        
        # Phase 2: Director
        directive = self.director_directive(cycle_id)
        
        # Phase 3: Architect
        proposals = self.architect_propose(cycle_id)
        
        # Phase 4: Critic
        reviews = self.critic_review(cycle_id, proposals)
        
        # Phase 5: Executor (dry-run)
        approved = reviews.get('approved', [])
        execution = self.executor_dryrun(cycle_id, approved, proposals)
        
        # Phase 6: Update system state
        system_state = self.load_memory('system_state.json')
        system_state['last_cycle_timestamp'] = datetime.now().isoformat()
        system_state['status'] = 'cycle_complete'
        self.save_memory('system_state.json', system_state)
        
        logger.info(f'===== FULL CYCLE {cycle_id} COMPLETE =====')
        
        return {
            'cycle_id': cycle_id,
            'history': history,
            'directive': directive,
            'proposals': len(proposals),
            'approved': len(approved),
            'rejected': len(reviews.get('rejected', [])),
            'execution': execution,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == '__main__':
    import sys
    orchestrator = FullCycleOrchestrator()
    cycle_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    result = orchestrator.run_full_cycle(cycle_id)
    print(json.dumps(result, indent=2))
