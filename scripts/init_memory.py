#!/usr/bin/env python3
"""
Initialize ARC memory structure with baseline files
"""

import os
import json
from datetime import datetime

MEMORY_DIR = '../memory'

def init_memory():
    """Create initial memory files"""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    
    # directive.json
    directive = {
        "cycle_id": 0,
        "mode": "explore",
        "objective": "establish_baseline",
        "novelty_budget": {"exploit": 3, "explore": 2, "wildcat": 0},
        "focus_areas": ["initialization", "baseline_metrics"],
        "forbidden_axes": [],
        "encouraged_axes": ["stable_configurations"],
        "notes": "Initial cycle - establishing baseline state for ARC system"
    }
    
    # history_summary.json
    history = {
        "total_cycles": 0,
        "total_experiments": 0,
        "best_metrics": {"auc": None, "sensitivity": None, "specificity": None},
        "recent_experiments": [],
        "failed_configs": [],
        "successful_patterns": []
    }
    
    # constraints.json
    constraints = {
        "forbidden_ranges": [],
        "unstable_configs": [],
        "safe_baselines": []
    }
    
    # system_state.json
    system_state = {
        "mode": "SEMI",
        "arc_version": "0.9.0",
        "llm_endpoint": "http://localhost:8000/generate",
        "last_cycle_timestamp": None,
        "status": "initialized",
        "active_experiments": []
    }
    
    # proposals.json
    proposals = {
        "cycle_id": 0,
        "proposals": [],
        "timestamp": None
    }
    
    # reviews.json
    reviews = {
        "cycle_id": 0,
        "reviews": [],
        "approved": [],
        "rejected": [],
        "timestamp": None
    }
    
    files = {
        'directive.json': directive,
        'history_summary.json': history,
        'constraints.json': constraints,
        'system_state.json': system_state,
        'proposals.json': proposals,
        'reviews.json': reviews
    }
    
    for filename, data in files.items():
        path = os.path.join(MEMORY_DIR, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Created {filename}')
    
    print('\nARC memory initialized successfully')
    print(f'Memory directory: {os.path.abspath(MEMORY_DIR)}')

if __name__ == '__main__':
    init_memory()
