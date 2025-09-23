#!/usr/bin/env python3
"""
Script to convert Qwen3 JSON files from separate shift files to unified format.

The input files are in format: {subset}_{shift}.json (e.g., r1s7T5_shift_1.json)
The output files should be in format: handsup_{subset}_qwen3.json

Usage: python convert_qwen3_json.py
"""

import json
import os
import re
from collections import defaultdict

def convert_qwen3_json():
    """Convert Qwen3 JSON files from separate shift format to unified format"""
    
    # Define the input directory
    input_dir = "/mnt/data/users/ivan.rodkin/lab/qwen3_handsup"
    
    # Find all JSON files and group by subset
    files_by_subset = defaultdict(list)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            # Extract subset and shift from filename
            # Format: qwen3_r1s7T5_shift_1.json or qwen3_no_reasoning_r1s7T5_shift_1.json
            match = re.match(r'qwen3(?:_no_reasoning)?_([^_]+)_shift_(\d+)\.json', filename)
            if match:
                subset = match.group(1)
                shift = int(match.group(2))
                # Group by subset and model variant (with/without reasoning)
                model_variant = "no_reasoning" if "_no_reasoning_" in filename else "reasoning"
                key = f"{subset}_{model_variant}"
                files_by_subset[key].append((shift, filename))
    
    # Process each subset
    for subset, shift_files in files_by_subset.items():
        print(f"Processing subset: {subset}")
        
        # Sort by shift number
        shift_files.sort(key=lambda x: x[0])
        
        # Initialize the output structure
        output_data = {
            'shift_1': [],
            'shift_2': [],
            'shift_3': [],
            'shift_4': []
        }
        
        # Process each shift file
        for shift_num, filename in shift_files:
            shift_key = f'shift_{shift_num}'
            file_path = os.path.join(input_dir, filename)
            
            print(f"  Processing {filename}...")
            
            # Load the JSON file (JSONL format - one JSON object per line)
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            
            # Convert each sample to the expected format
            for sample in data:
                # Extract the required fields - adjust field names as needed for Qwen3
                generation = sample.get('llm_response', sample.get('response', sample.get('generation', '')))
                target_names = sample.get('answer_next_names', sample.get('target_names', []))
                
                # Convert target names to comma-separated string
                if isinstance(target_names, list):
                    target_str = ', '.join(target_names)
                else:
                    target_str = str(target_names)
                
                # Create the sample in the expected format
                converted_sample = {
                    'generation': generation,
                    'target': target_str
                }
                
                output_data[shift_key].append(converted_sample)
        
        # Create output filename
        if "no_reasoning" in subset:
            model_name = "qwen3-no-reasoning"
            clean_subset = subset.replace("_no_reasoning", "")
        else:
            model_name = "qwen3"
            clean_subset = subset.replace("_reasoning", "")
        
        output_filename = f"handsup_{clean_subset}_{model_name}.json"
        output_path = os.path.join(input_dir, output_filename)
        
        # Save the converted data
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  Saved {output_filename}")
        print(f"  Total samples: {sum(len(samples) for samples in output_data.values())}")
        for shift_key, samples in output_data.items():
            print(f"    {shift_key}: {len(samples)} samples")
        print()

if __name__ == "__main__":
    convert_qwen3_json()
