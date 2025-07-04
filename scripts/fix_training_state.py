#!/usr/bin/env python3
"""
Utility script to fix training state files that may be missing required keys.
This can happen when upgrading from older versions.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fix_state_file(state_file_path):
    """Fix a training state file by adding missing keys."""
    # Default state structure
    default_state = {
        "preprocessing": {"completed": False, "timestamp": None},
        "syncnet_training": {"completed": False, "epochs": 0, "checkpoint": None, "timestamp": None},
        "main_training": {"completed": False, "epochs": 0, "checkpoints": [], "timestamp": None}
    }
    
    if not os.path.exists(state_file_path):
        print(f"State file not found: {state_file_path}")
        return False
    
    try:
        # Load existing state
        with open(state_file_path, 'r') as f:
            state = json.load(f)
        
        # Check if it needs fixing
        needs_update = False
        
        # Ensure all top-level keys exist
        for key, default_value in default_state.items():
            if key not in state:
                print(f"  Adding missing key: {key}")
                state[key] = default_value
                needs_update = True
            elif isinstance(default_value, dict):
                # Ensure all nested keys exist
                for sub_key, sub_default in default_value.items():
                    if sub_key not in state[key]:
                        print(f"  Adding missing sub-key: {key}.{sub_key}")
                        state[key][sub_key] = sub_default
                        needs_update = True
        
        if needs_update:
            # Backup original file
            backup_path = state_file_path + '.backup'
            os.rename(state_file_path, backup_path)
            print(f"  Backed up original to: {backup_path}")
            
            # Save fixed state
            with open(state_file_path, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"  Fixed state file: {state_file_path}")
            return True
        else:
            print(f"  State file is already valid: {state_file_path}")
            return False
            
    except Exception as e:
        print(f"  Error fixing state file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Fix training state files that may be missing required keys'
    )
    parser.add_argument('--dataset', type=str, help='Specific dataset name to fix')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Base directory for datasets (default: dataset)')
    parser.add_argument('--all', action='store_true',
                        help='Fix all datasets in the dataset directory')
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all:
        parser.error('Either --dataset or --all must be specified')
    
    datasets_to_fix = []
    
    if args.all:
        # Find all datasets with state files
        dataset_base = Path(args.dataset_dir)
        if dataset_base.exists():
            for dataset_dir in dataset_base.iterdir():
                if dataset_dir.is_dir():
                    state_file = dataset_dir / '.training_state.json'
                    if state_file.exists():
                        datasets_to_fix.append(dataset_dir.name)
    else:
        datasets_to_fix.append(args.dataset)
    
    if not datasets_to_fix:
        print("No datasets found to fix")
        return
    
    print(f"Found {len(datasets_to_fix)} dataset(s) to check")
    print()
    
    fixed_count = 0
    for dataset_name in datasets_to_fix:
        dataset_path = os.path.join(args.dataset_dir, dataset_name)
        state_file_path = os.path.join(dataset_path, '.training_state.json')
        
        print(f"Checking dataset: {dataset_name}")
        if fix_state_file(state_file_path):
            fixed_count += 1
        print()
    
    print(f"Summary: Fixed {fixed_count} state file(s)")


if __name__ == '__main__':
    main()