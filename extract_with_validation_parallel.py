#!/usr/bin/env python3
"""
Script to extract names from Gemini generations using gemma-3-12b with validation.
Applies the rule: extract -> validate -> output only valid names.
Now with parallel processing for maximum throughput.
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
    """Create a prompt for text completion to extract names."""
    # Truncate generation to last 500 characters to avoid context length issues
    if len(generation) > 500:
        truncated_text = "..." + generation[-500:]
    else:
        truncated_text = generation
    
    prompt = f"""{truncated_text}
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
    elif "everyone" in text_lower:
        return ALL_FRIENDS[:num_friends] if num_friends else ALL_FRIENDS
    
    # Try to parse comma-separated names
    names = []
    
    # Split by comma and clean up
    for name in extracted_text.split(','):
        name = name.strip()
        if name and name not in ['and', 'the', '']:
            names.append(name)
    
    # Also try splitting by "and" if we have few names
    if len(names) <= 2 and ' and ' in extracted_text:
        names = []
        for part in extracted_text.split(' and '):
            part = part.strip()
            if part and part not in ['and', 'the', '']:
                names.append(part)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    
    return unique_names

def validate_names(extracted_names: List[str], valid_friends: Set[str]) -> List[str]:
    """Validate that all extracted names are in the valid friends list."""
    return [name for name in extracted_names if name in valid_friends]

def process_single_sample(sample_data: dict, valid_friends: set) -> dict:
    """Process a single sample with extraction and validation."""
    shift, i, sample = sample_data['shift'], sample_data['index'], sample_data['sample']
    
    if sample['generation'] is None or sample['target'] is None:
        return None
    
    # Extract names using gemma-3-12b
    extracted_text = extract_names_with_gemma(sample['generation'])
    
    # Parse extracted names
    extracted_names = parse_extracted_names(extracted_text, len(valid_friends))
    
    # Validate names
    valid_names = validate_names(extracted_names, valid_friends)
    
    # Print extracted names for monitoring
    thread_id = threading.current_thread().name
    print(f"[{thread_id}] {shift}[{i}]: {valid_names}")
    
    # Return updated sample
    sample['extracted_names'] = valid_names
    sample['extraction_method'] = 'gemma3:12b-it-q8_0_with_validation'
    
    return {
        'shift': shift,
        'index': i,
        'sample': sample
    }

def process_json_file(input_file: str, output_file: str, num_friends: int = 7, max_workers: int = 4):
    """Process a JSON file and extract validated names in parallel."""
    
    print(f"Processing: {input_file}")
    
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Determine valid friends based on num_friends
    valid_friends = set(ALL_FRIENDS[:num_friends])
    print(f"Valid friends for validation: {sorted(valid_friends)}")
    
    # Collect all samples to process
    samples_to_process = []
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        if shift in data:
            for i, sample in enumerate(data[shift]):
                if sample['generation'] is not None and sample['target'] is not None:
                    samples_to_process.append({
                        'shift': shift,
                        'index': i,
                        'sample': sample
                    })
    
    print(f"Found {len(samples_to_process)} samples to process with {max_workers} parallel workers")
    
    # Process samples in parallel
    processed_samples = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(process_single_sample, sample_data, valid_friends): sample_data 
            for sample_data in samples_to_process
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(samples_to_process), desc="Processing samples", unit="samples") as pbar:
            for future in as_completed(future_to_sample):
                result = future.result()
                if result is not None:
                    # Update the original data structure
                    shift = result['shift']
                    index = result['index']
                    data[shift][index] = result['sample']
                    processed_samples += 1
                pbar.update(1)
    
    # Save the updated data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Completed! Processed {processed_samples} out of {len(samples_to_process)} samples.")
    print(f"Results saved to: {output_file}")
    
    return processed_samples, len(samples_to_process)

def show_sample_results(output_file: str, num_samples: int = 5):
    """Show sample results from the processed file."""
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # Collect samples with extractions
    samples_with_extractions = []
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        if shift in data:
            for i, sample in enumerate(data[shift]):
                if 'extracted_names' in sample:
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
    parser = argparse.ArgumentParser(description='Extract names from JSON files using gemma-3-12b with validation (parallel version)')
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
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of parallel workers (default: 4)')
    
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
        
        print(f"\nStarting batch processing with {args.max_workers} parallel workers...")
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
                    file_info['num_friends'],
                    args.max_workers
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
        processed, total = process_json_file(args.input, args.output, args.num_friends, args.max_workers)
        
        # Show sample results
        show_sample_results(args.output, args.show_samples)
