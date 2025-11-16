#!/usr/bin/env python3
"""
ARC Cycle Orchestrator
Executes the full research cycle: Historian -> Director -> Architect -> Critic -> Executor
"""

import os
import json
import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/arc/logs/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MEMORY_DIR = '/workspace/arc/memory'
EXPERIMENTS_DIR = '/workspace/arc/experiments'
LLM_ENDPOINT = 'http://localhost:8000/generate'
CONTROL_PLANE_URL = 'http://localhost:8002'

class ARCOrchestrator:
    def __init__(self):
        self.memory_dir = MEMORY_DIR
        self.experiments_dir = EXPERIMENTS_DIR
        self.llm_endpoint = LLM_ENDPOINT
        
    def load_memory(self, filename: str) -> Dict[str, Any]:
        """Load memory file"""
        path = os.path.join(self.memory_dir, filename)
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_memory(self, filename: str, data: Dict[str, Any]):
        """Save memory file"""
        path = os.path.join(self.memory_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call the LLM endpoint"""
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
    
    def historian_update(self, cycle_id: int) -> Dict[str, Any]:
        """Historian role: Update history summary"""
        logger.info(f'Running Historian for cycle {cycle_id}')
        
        history = self.load_memory('history_summary.json')
        
        prompt = f"""You are the Historian role in the ARC (Autonomous Research Collective) system.

Your task: Update the research history summary for cycle {cycle_id}.

Current history:
{json.dumps(history, indent=2)}

Instructions:
1. Review the current history
2. If this is cycle 0 (initialization), note that the baseline is being established
3. Summarize any patterns or insights
4. Output ONLY valid JSON matching this structure:
{{
  "total_cycles": <number>,
  "total_experiments": <number>,
  "best_metrics": {{"auc": <float or null>, "sensitivity": <float or null>, "specificity": <float or null>}},
  "recent_experiments": [],
  "failed_configs": [],
  "successful_patterns": []
}}

Output (JSON only):"""
        
        response = self.call_llm(prompt, max_tokens=1000)
        
        # Parse JSON from response
        try:
            # Extract JSON from response (may have think tags)
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                updated_history = json.loads(json_str)
                self.save_memory('history_summary.json', updated_history)
                logger.info('Historian update complete')
                return updated_history
            else:
                logger.warning('No JSON found in Historian response, keeping existing history')
                return history
        except Exception as e:
            logger.error(f'Historian JSON parsing error: {e}')
            return history
    
    def director_directive(self, cycle_id: int) -> Dict[str, Any]:
        """Director role: Create strategic directive"""
        logger.info(f'Running Director for cycle {cycle_id}')
        
        history = self.load_memory('history_summary.json')
        constraints = self.load_memory('constraints.json')
        
        prompt = f"""You are the Director role in the ARC (Autonomous Research Collective) system.

Your task: Create a strategic directive for cycle {cycle_id+1}.

Research history:
{json.dumps(history, indent=2)}

Current constraints:
{json.dumps(constraints, indent=2)}

Instructions:
1. Analyze the research history
2. For cycle 0, set mode to "explore" and objective to "establish_baseline"
3. Allocate novelty budget: exploit (safe variations), explore (moderate risk), wildcat (high risk)
4. Identify focus areas and any forbidden/encouraged axes
5. Output ONLY valid JSON matching this structure:
{{
  "cycle_id": <number>,
  "mode": "<explore|exploit|wildcat>",
  "objective": "<objective_description>",
  "novelty_budget": {{"exploit": <int>, "explore": <int>, "wildcat": <int>}},
  "focus_areas": ["<area1>", "<area2>"],
  "forbidden_axes": [],
  "encouraged_axes": ["<axis1>"],
  "notes": "<strategic notes>"
}}

Output (JSON only):"""
        
        response = self.call_llm(prompt, max_tokens=1000)
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                directive = json.loads(json_str)
                directive['cycle_id'] = cycle_id + 1
                self.save_memory('directive.json', directive)
                logger.info('Director directive complete')
                return directive
            else:
                logger.warning('No JSON found in Director response')
                return self.load_memory('directive.json')
        except Exception as e:
            logger.error(f'Director JSON parsing error: {e}')
            return self.load_memory('directive.json')
    
    def run_cycle(self, cycle_id: int = 0):
        """Execute full research cycle"""
        logger.info(f'===== Starting ARC Cycle {cycle_id} =====')
        
        # Phase 1: Historian
        history = self.historian_update(cycle_id)
        
        # Phase 2: Director
        directive = self.director_directive(cycle_id)
        
        # Phase 3: Update system state
        system_state = self.load_memory('system_state.json')
        system_state['last_cycle_timestamp'] = datetime.now().isoformat()
        system_state['status'] = 'cycle_complete'
        self.save_memory('system_state.json', system_state)
        
        logger.info(f'===== Cycle {cycle_id} Complete =====')
        
        return {
            'cycle_id': cycle_id,
            'history': history,
            'directive': directive,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == '__main__':
    import sys
    
    orchestrator = ARCOrchestrator()
    
    # Get cycle ID from command line or default to 0
    cycle_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    result = orchestrator.run_cycle(cycle_id)
    
    print(json.dumps(result, indent=2))
