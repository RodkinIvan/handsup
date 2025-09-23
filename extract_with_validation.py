#!/usr/bin/env python3
"""
Script to extract names from Gemini generations using gemma-3-12b with validation.
Applies the rule: extract -> validate -> output only valid names.
"""

import json
import argparse
import ollama
import os
from typing import List, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set environment variables for optimal GPU usage
os.environ['OLLAMA_GPU_LAYERS'] = '-1'  # Use all layers on GPU
os.environ['OLLAMA_GPU_MEMORY_FRACTION'] = '0.9'  # Use 90% of GPU memory

# Known friend names for validation
ALL_FRIENDS = ['Alice', 'Bob', 'Carol', 'Dave', 'Erin', 'Frank', 'Grace', 'Heidi', 'Ivan', 'Judy', 'Mallory', 'Niaj', 'Olivia', 'Peggy', 'Quentin', 'Rupert', 'Sybil', 'Trent', 'Uma', 'Victor']

def create_completion_prompt(generation: str) -> str:
    """Create a text completion prompt for LLM to extract the final answer."""
    
    prompt = f"""{generation}
If the final answer is nobody, the final list is "No one". If the final answer is everyone, the final list is "Everyone".
Therefore, the final list of comma-separated friends' names with raised hands are: """
    return prompt

def extract_names_with_gemma(generation: str) -> str:
    """Use gemma-3-12b for text completion to extract names."""
    
    prompt = create_completion_prompt(generation)

    try:
        response = ollama.generate(
            model="gemma3:12b-it-q8_0",
            prompt=prompt,
            raw=True,
            options={
                "temperature": 0.0,  # Deterministic greedy generation
                "top_p": 1.0,        # Use all tokens for greedy selection
                "num_predict": 100,  # Increased to capture longer name lists
                "gpu_layers": -1,    # Use all layers on GPU
                "stop": ["\n\n", "Therefore", "In summary", "Step", "So,", "Since", "Based on", "Following", "The pattern"]  # Stop at reasoning words
            }
        )
        
        completion = response['response'].strip()
        return completion
            
    except Exception as e:
        return f"Error: {str(e)}"

def parse_extracted_names(extracted_text: str, num_friends: int = None) -> List[str]:
    """Parse the extracted text to get individual names."""
    if not extracted_text or extracted_text.startswith("Error:"):
        return []
    
    # Handle special cases
    text_lower = extracted_text.lower()
    if "no one" in text_lower or "nobody" in text_lower:
        return []
    if "everyone" in text_lower:
        if num_friends is None:
            num_friends = 7  # Default fallback
        return ALL_FRIENDS[:num_friends]
    
    # Clean up the text
    text = extracted_text.strip().strip('.,')
    
    # Handle "and" separators more carefully
    names = []
    
    # First, split by commas
    comma_parts = text.split(',')
    
    for part in comma_parts:
        part = part.strip()
        if not part:
            continue
            
        # Check if this part contains "and"
        if ' and ' in part:
            # Split by "and" and process each part
            and_parts = part.split(' and ')
            for and_part in and_parts:
                name = and_part.strip().strip('.,')
                if name and name not in ['and', 'the']:
                    names.append(name)
        else:
            # No "and" in this part, just add it
            if part and part not in ['and', 'the']:
                names.append(part)
    
    # If no commas found, try splitting by "and" directly
    if not names and ' and ' in text:
        and_parts = text.split(' and ')
        for part in and_parts:
            name = part.strip().strip('.,')
            if name and name not in ['and', 'the']:
                names.append(name)
    
    # If still no names found, try the whole text as a single name
    if not names:
        if text and text not in ['and', 'the']:
            names = [text]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    
    return unique_names

def validate_names(names: List[str], valid_friends: Set[str]) -> List[str]:
    """Validate names against the known friend list and return only valid ones."""
    valid_names = []
    for name in names:
        if name in valid_friends:
            valid_names.append(name)
    return valid_names

def process_json_file(input_file: str, output_file: str, num_friends: int = 7):
    """Process a JSON file and extract validated names."""
    
    print(f"Processing: {input_file}")
    
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Determine valid friends based on num_friends
    valid_friends = set(ALL_FRIENDS[:num_friends])
    print(f"Valid friends for validation: {sorted(valid_friends)}")
    
    # First, count total samples to process
    total_samples = 0
    samples_to_process = []
    
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        if shift in data:
            for i, sample in enumerate(data[shift]):
                if sample['generation'] is not None and sample['target'] is not None:
                    samples_to_process.append((shift, i, sample))
                    total_samples += 1
    
    print(f"Found {total_samples} samples to process")
    
    # Process each sample with progress bar
    processed_samples = 0
    
    for shift, i, sample in tqdm(samples_to_process, desc="Processing samples"):
        # Extract names using gemma-3-12b
        extracted_text = extract_names_with_gemma(sample['generation'])
        
        # Parse extracted names
        extracted_names = parse_extracted_names(extracted_text, num_friends)
        
        # Validate names
        valid_names = validate_names(extracted_names, valid_friends)
        
        # Add the extracted and validated names to the sample
        sample['extracted_names'] = valid_names
        sample['extraction_method'] = 'gemma3:12b-it-q8_0_with_validation'
        
        processed_samples += 1
    
    # Save the updated data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Completed! Processed {processed_samples} out of {total_samples} samples.")
    print(f"Results saved to: {output_file}")
    
    return processed_samples, total_samples

def show_sample_results(input_file: str, num_samples: int = 5):
    """Show sample results after processing."""
    
    print(f"\nLoading results from: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Collect samples with extracted names
    samples_with_extractions = []
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        if shift in data:
            for i, sample in enumerate(data[shift]):
                if sample.get('extracted_names') is not None:
                    samples_with_extractions.append({
                        'shift': shift,
                        'index': i,
                        'generation': sample['generation'],
                        'target': sample['target'],
                        'extracted_names': sample['extracted_names']
                    })
    
    # Show sample results
    print(f"\nShowing {min(num_samples, len(samples_with_extractions))} sample results:")
    print("=" * 80)
    
    for i, sample in enumerate(samples_with_extractions[:num_samples]):
        print(f"\nSample {i+1} ({sample['shift']}, index {sample['index']}):")
        print("-" * 40)
        print(f"Target: {sample['target']}")
        print(f"Extracted names: {sample['extracted_names']}")
        print(f"Generation (last 200 chars): ...{sample['generation'][-200:]}")
        
        # Show if extraction matches target
        target_names = [name.strip() for name in sample['target'].split(',')]
        if set(sample['extracted_names']) == set(target_names):
            print("✓ EXACT MATCH!")
        else:
            print("✗ No match")
            print(f"  Missing from extraction: {set(target_names) - set(sample['extracted_names'])}")
            print(f"  Extra in extraction: {set(sample['extracted_names']) - set(target_names)}")

def find_files_to_process(data_dir="handsup_evals"):
    """Find all JSON files that don't have extracted versions yet."""
    import os
    import re
    
    files_to_process = []
    
    # Get all JSON files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and filename.startswith('handsup_'):
            # Skip already extracted files
            if '_extracted.json' in filename:
                continue
                
            # Skip evaluated files
            if '_evaluated.json' in filename:
                print(f"Skipping {filename} (evaluated file)")
                continue
                
            # Skip files that are already extracted (have _extracted version)
            base_name = filename.replace('.json', '')
            extracted_filename = f"{base_name}_extracted.json"
            extracted_path = os.path.join(data_dir, extracted_filename)
            
            if os.path.exists(extracted_path):
                print(f"Skipping {filename} (already has extracted version)")
                continue
            
            # Determine number of friends from filename
            match = re.search(r's(\d+)', filename)
            if match:
                num_friends = int(match.group(1))
            else:
                num_friends = 7  # default fallback
                
            input_path = os.path.join(data_dir, filename)
            output_path = os.path.join(data_dir, extracted_filename)
            
            files_to_process.append({
                'input': input_path,
                'output': output_path,
                'num_friends': num_friends,
                'filename': filename
            })
    
    return files_to_process

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract names from JSON files using gemma-3-12b with validation')
    parser.add_argument('--input', type=str,
                       help='Input JSON file (e.g., handsup_r1s7T5_gemini-2.5-pro.json) - if not provided, process all files in handsup_evals')
    parser.add_argument('--output', type=str,
                       help='Output JSON file (default: input file with _extracted suffix)')
    parser.add_argument('--num_friends', type=int, default=7,
                       help='Number of friends to validate against (default: 7, auto-detected from filename if processing all files)')
    parser.add_argument('--show_samples', type=int, default=5,
                       help='Number of sample results to show (default: 5)')
    parser.add_argument('--data_dir', type=str, default='handsup_evals',
                       help='Directory containing JSON files (default: handsup_evals)')
    parser.add_argument('--all', action='store_true',
                       help='Process all files in handsup_evals directory that don\'t have extracted versions')
    
    args = parser.parse_args()
    
    if args.all or not args.input:
        # Process all files in handsup_evals directory
        print("Finding files to process...")
        files_to_process = find_files_to_process(args.data_dir)
        
        if not files_to_process:
            print("No files found to process!")
            exit(0)
        
        print(f"\nFound {len(files_to_process)} files to process:")
        print("=" * 60)
        for i, file_info in enumerate(files_to_process, 1):
            print(f"{i:2d}. {file_info['filename']} → {file_info['output'].split('/')[-1]} (s{file_info['num_friends']})")
        
        print("=" * 60)
        
        # Ask for confirmation before proceeding
        try:
            response = input("\nProceed with extraction? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Extraction cancelled.")
                exit(0)
        except KeyboardInterrupt:
            print("\nExtraction cancelled.")
            exit(0)
        
        print("\nStarting batch processing...")
        print("=" * 80)
        
        total_processed = 0
        total_samples = 0
        
        for i, file_info in enumerate(files_to_process):
            print(f"\n[{i+1}/{len(files_to_process)}] Processing: {file_info['filename']}")
            print("-" * 60)
            
            try:
                processed, total = process_json_file(
                    file_info['input'], 
                    file_info['output'], 
                    file_info['num_friends']
                )
                total_processed += processed
                total_samples += total
                print(f"✓ Successfully processed {processed}/{total} samples")
                
            except Exception as e:
                print(f"✗ Error processing {file_info['filename']}: {e}")
                continue
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"Total files processed: {len(files_to_process)}")
        print(f"Total samples processed: {total_processed}/{total_samples}")
        print(f"{'='*80}")
        
    else:
        # Process single file (original behavior)
        # Set default output file if not provided
        if not args.output:
            base_name = args.input.replace('.json', '')
            args.output = f"{base_name}_extracted.json"
        
        # Process the file
        processed, total = process_json_file(args.input, args.output, args.num_friends)
        
        # Show sample results
        show_sample_results(args.output, args.show_samples)
