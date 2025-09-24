#!/usr/bin/env python3
"""
Script to estimate Gemma-3-12B extraction quality by manually annotating
30 random samples and comparing with automatic extractions.
"""

import json
import random
import argparse
import ollama
import os
from typing import List, Set, Dict
from model_extractors import ALL_FRIENDS

# Set environment variables for optimal GPU usage
os.environ['OLLAMA_GPU_LAYERS'] = '-1'
os.environ['OLLAMA_GPU_MEMORY_FRACTION'] = '0.9'

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
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": 100,
                "gpu_layers": -1,
                "stop": ["\n\n", "Therefore", "In summary", "Step", "So,", "Since", "Based on", "Following", "The pattern"]
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
            num_friends = 7
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

def load_first_samples(file_path: str, num_samples: int = 30) -> List[Dict]:
    """Load the first N samples from a JSON file in order."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Collect all samples from all shifts in order
    all_samples = []
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        if shift in data:
            for i, sample in enumerate(data[shift]):
                if sample['generation'] is not None and sample['target'] is not None:
                    all_samples.append({
                        'shift': shift,
                        'index': i,
                        'generation': sample['generation'],
                        'target': sample['target']
                    })
    
    # Take first N samples (no shuffling)
    return all_samples[:num_samples]


def get_manual_annotations() -> Dict[int, List[str]]:
    """Return hardcoded manual annotations for 30 random samples."""
    
    # Manual annotations based on what the model actually predicted in its generation (first 30 samples)
    manual_extractions = {
        0: ['Carol'],  # "Carol raises her hand." (model's actual prediction)
        1: ['Alice', 'Bob', 'Carol', 'Dave', 'Frank', 'Grace'],  # "Alice, Bob, Carol, Dave, Frank, and Grace raise their hands."
        2: [],  # "they will all keep their hands on the table in Round 6."
        3: ['Alice', 'Bob', 'Carol', 'Dave', 'Erin', 'Frank', 'Grace'],  # "Alice, Bob, Carol, Dave, Erin, Frank, and Grace raise their hands."
        4: ['Bob'],  # "Bob raises his hand."
        5: ['Alice', 'Carol', 'Erin', 'Frank'],  # "Alice, Carol, Erin, and Frank raise their hands."
        6: ['Alice', 'Bob', 'Carol', 'Dave', 'Erin', 'Frank', 'Grace'],  # "Alice, Bob, Carol, Dave, Erin, Frank, and Grace raise their hands."
        7: ['Grace'],  # "Grace raises her hand."
        8: ['Frank'],  # "Frank raises his hand."
        9: ['Alice', 'Bob', 'Carol', 'Erin', 'Frank', 'Grace'],  # "Alice, Bob, Carol, Erin, Frank, and Grace raise their hands."
        10: ['Alice', 'Carol', 'Erin', 'Frank'],  # "Alice, Carol, Erin, and Frank raise their hands."
        11: ['Bob', 'Dave', 'Erin', 'Grace'],  # "Bob, Dave, Erin, and Grace raise their hands."
        12: ['Bob', 'Carol', 'Erin', 'Frank', 'Grace'],  # "Bob, Carol, Erin, Frank, and Grace raise their hands."
        13: ['Alice', 'Bob', 'Carol', 'Dave', 'Frank', 'Grace'],  # "Alice, Bob, Carol, Dave, Frank, and Grace raise their hands."
        14: ['Carol', 'Dave', 'Erin'],  # "Carol, Dave, and Erin raise their hands." (model's actual prediction)
        15: ['Carol', 'Grace'],  # "Carol and Grace raise their hands."
        16: ['Bob', 'Carol', 'Dave', 'Frank', 'Grace'],  # "Bob, Carol, Dave, Frank, and Grace raise their hands."
        17: ['Alice', 'Carol', 'Erin'],  # "Alice, Carol, and Erin raise their hands."
        18: ['Alice', 'Dave', 'Frank'],  # "Alice, Dave, and Frank raise their hands."
        19: ['Alice', 'Bob', 'Carol'],  # "Alice, Bob, and Carol raise their hands."
        20: ['Alice', 'Bob', 'Carol', 'Dave', 'Erin', 'Frank', 'Grace'],  # "Alice, Bob, Carol, Dave, Erin, Frank, and Grace raise their hands."
        21: ['Alice', 'Bob', 'Carol', 'Dave', 'Erin', 'Frank', 'Grace'],  # "Alice, Bob, Carol, Dave, Erin, Frank, and Grace."
        22: ['Erin', 'Frank', 'Grace'],  # "Erin, Frank, and Grace raise their hands."
        23: ['Alice', 'Carol', 'Dave', 'Grace'],  # "Alice, Carol, Dave, and Grace raise their hands."
        24: ['Alice', 'Bob', 'Carol', 'Dave', 'Erin', 'Grace'],  # "Alice, Bob, Carol, Dave, Erin, and Grace raise their hands."
        25: ['Alice', 'Erin'],  # "Alice and Erin raise their hands."
        26: ['Alice', 'Bob', 'Carol', 'Dave', 'Erin', 'Frank', 'Grace'],  # "Alice, Bob, Carol, Dave, Erin, Frank, and Grace raise their hands."
        27: ['Alice', 'Bob', 'Carol', 'Dave', 'Erin', 'Frank', 'Grace'],  # "Alice, Bob, Carol, Dave, Erin, Frank, and Grace raise their hands."
        28: ['Bob', 'Carol', 'Frank'],  # "Bob, Carol, and Frank raise their hands."
        29: ['Dave', 'Erin', 'Frank', 'Grace']  # "Dave, Erin, Frank, and Grace raise their hands."
    }
    
    return manual_extractions

def run_gemma_extraction(samples: List[Dict], num_friends: int) -> Dict[int, List[str]]:
    """Run Gemma extraction on all samples."""
    gemma_extractions = {}
    valid_friends = set(ALL_FRIENDS[:num_friends])
    
    print(f"\nRunning Gemma-3-12B extraction on {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}...")
        
        # Extract names using Gemma
        extracted_text = extract_names_with_gemma(sample['generation'])
        
        # Parse extracted names
        extracted_names = parse_extracted_names(extracted_text, num_friends)
        
        # Validate names
        valid_names = validate_names(extracted_names, valid_friends)
        
        gemma_extractions[i] = valid_names
        
        print(f"  Extracted: {valid_names}")
    
    return gemma_extractions

def calculate_metrics(manual_extractions: Dict[int, List[str]], 
                     gemma_extractions: Dict[int, List[str]]) -> Dict:
    """Calculate extraction quality metrics."""
    
    exact_matches = 0
    total_samples = len(manual_extractions)
    
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    
    detailed_results = []
    
    for i in range(total_samples):
        manual = set(manual_extractions.get(i, []))
        gemma = set(gemma_extractions.get(i, []))
        
        # Exact match
        is_exact_match = manual == gemma
        if is_exact_match:
            exact_matches += 1
        
        # Precision, Recall, F1
        if len(gemma) == 0 and len(manual) == 0:
            precision = recall = f1 = 1.0
        elif len(gemma) == 0:
            precision = recall = f1 = 0.0
        else:
            tp = len(manual & gemma)  # True positives
            fp = len(gemma - manual)  # False positives
            fn = len(manual - gemma)  # False negatives
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        
        detailed_results.append({
            'sample_idx': i,
            'manual': sorted(list(manual)),
            'gemma': sorted(list(gemma)),
            'exact_match': is_exact_match,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return {
        'exact_match_accuracy': exact_matches / total_samples,
        'average_precision': precision_sum / total_samples,
        'average_recall': recall_sum / total_samples,
        'average_f1': f1_sum / total_samples,
        'total_samples': total_samples,
        'exact_matches': exact_matches,
        'detailed_results': detailed_results
    }

def print_results(metrics: Dict):
    """Print the evaluation results."""
    print(f"\n{'='*80}")
    print("GEMMA-3-12B EXTRACTION QUALITY EVALUATION")
    print(f"{'='*80}")
    
    print(f"Total samples evaluated: {metrics['total_samples']}")
    print(f"Exact matches: {metrics['exact_matches']}")
    print(f"Exact match accuracy: {metrics['exact_match_accuracy']:.3f}")
    print(f"Average precision: {metrics['average_precision']:.3f}")
    print(f"Average recall: {metrics['average_recall']:.3f}")
    print(f"Average F1 score: {metrics['average_f1']:.3f}")
    
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    
    for result in metrics['detailed_results']:
        status = "✓" if result['exact_match'] else "✗"
        print(f"\nSample {result['sample_idx']+1}: {status}")
        print(f"  Manual: {result['manual']}")
        print(f"  Gemma:  {result['gemma']}")
        print(f"  Precision: {result['precision']:.3f}, Recall: {result['recall']:.3f}, F1: {result['f1']:.3f}")

def save_results(metrics: Dict, output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate Gemma-3-12B extraction quality')
    parser.add_argument('--file', type=str, 
                       default="handsup_evals/handsup_r1s7T5_gemini-2.5-pro.json",
                       help='Path to the JSON file with generations')
    parser.add_argument('--num_samples', type=int, default=30,
                       help='Number of random samples to evaluate (default: 30)')
    parser.add_argument('--output', type=str, default="gemma_extraction_evaluation.json",
                       help='Output file for results (default: gemma_extraction_evaluation.json)')
    
    args = parser.parse_args()
    
    # Determine number of friends from filename
    import re
    match = re.search(r's(\d+)', args.file)
    num_friends = int(match.group(1)) if match else 7
    
    print(f"Loading random samples from: {args.file}")
    print(f"Number of friends: {num_friends}")
    
    # Load first samples in order
    samples = load_first_samples(args.file, args.num_samples)
    print(f"Loaded {len(samples)} samples")
    
    # Get hardcoded manual annotations
    manual_extractions = get_manual_annotations()
    print(f"Using hardcoded manual annotations for {len(manual_extractions)} samples")
    
    # Run Gemma extraction
    gemma_extractions = run_gemma_extraction(samples, num_friends)
    
    # Calculate metrics
    metrics = calculate_metrics(manual_extractions, gemma_extractions)
    
    # Print results
    print_results(metrics)
    
    # Save results
    save_results(metrics, args.output)
