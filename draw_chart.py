import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from model_extractors import parse_target_string, ALL_FRIENDS



def names_to_binary(names_list, num_friends=None):
    """Convert list of names to binary string representation"""
    if num_friends is None:
        num_friends = len(ALL_FRIENDS)
    
    binary = ['0'] * num_friends
    
    for name in names_list:
        if name in ALL_FRIENDS:
            idx = ALL_FRIENDS.index(name)
            if idx < num_friends:  # Only include if within the expected range
                binary[idx] = '1'
    
    return ''.join(binary)

def binary_to_names(binary_str, num_friends=None):
    """Convert binary string back to list of names"""
    if num_friends is None:
        num_friends = len(ALL_FRIENDS)
    
    names = []
    for i, bit in enumerate(binary_str):
        if bit == '1' and i < num_friends and i < len(ALL_FRIENDS):
            names.append(ALL_FRIENDS[i])
    return names

def compute_metrics(predictions, targets, num_friends=None):
    """Compute accuracy, precision, recall, and F1 score"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    total_samples = len(predictions)
    correct_predictions = 0
    
    # Per-sample metrics
    precisions = []
    recalls = []
    f1_scores = []
    
    for pred, target in zip(predictions, targets):
        # Convert binary strings to sets of names
        pred_set = set(binary_to_names(pred, num_friends))
        target_set = set(binary_to_names(target, num_friends))
        
        # Check if prediction is exactly correct
        if pred_set == target_set:
            correct_predictions += 1
        
        # Compute precision, recall, F1
        if len(pred_set) == 0 and len(target_set) == 0:
            precision = recall = f1 = 1.0
        elif len(pred_set) == 0:
            precision = recall = f1 = 0.0
        else:
            tp = len(pred_set & target_set)  # True positives
            fp = len(pred_set - target_set)  # False positives
            fn = len(target_set - pred_set)  # False negatives
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Overall metrics
    accuracy = correct_predictions / total_samples
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    return {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'exact_matches': correct_predictions,
        'total_samples': total_samples
    }

def analyze_results(file_path):
    """Analyze results from JSON file with pre-computed extractions"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Parse filename to get the number of friends and model name
    import os
    filename = os.path.basename(file_path)
    # Extract s<number> from filename like handsup_r2s20T10_gemini-2.5-pro.json
    import re
    match = re.search(r's(\d+)', filename)
    if match:
        num_friends = int(match.group(1))
        print(f"Parsed {num_friends} friends from filename: {filename}")
    else:
        num_friends = 20  # fallback
        print(f"Could not parse number of friends from filename, using default: {num_friends}")
    
    # Extract model name from filename
    model_match = re.search(r'handsup_r\d+s\d+T\d+_(.+)\.json$', filename)
    model_name = model_match.group(1) if model_match else 'gemini-2.5-pro'  # default to gemini-2.5-pro
    print(f"Using model: {model_name}")
    
    # Check if this file has pre-computed extractions
    has_extractions = False
    extraction_method = None
    if data.get('shift_1') and len(data['shift_1']) > 0:
        sample = data['shift_1'][0]
        if 'extracted_names' in sample:
            has_extractions = True
            extraction_method = sample.get('extraction_method', 'unknown')
            print(f"Found pre-computed extractions using method: {extraction_method}")
    
    all_predictions = []
    all_targets = []
    shift_metrics = {}
    
    print("Analyzing results...")
    print("=" * 80)
    
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        print(f"\n{shift.upper()}:")
        print("-" * 40)
        
        shift_predictions = []
        shift_targets = []
        
        for i, sample in enumerate(data[shift]):
            if sample['generation'] is None or sample['target'] is None:
                continue
            
            # Get target names
            target_names = parse_target_string(sample['target'])
            
            # Get predicted names - use pre-computed extractions if available
            if has_extractions and 'extracted_names' in sample:
                pred_names = sample['extracted_names']
            else:
                # Fallback to rule-based extraction if no pre-computed extractions
                from model_extractors import extract_names_from_text
                pred_names = extract_names_from_text(sample['generation'], num_friends, model_name)
            
            # Convert to binary
            pred_binary = names_to_binary(pred_names, num_friends)
            target_binary = names_to_binary(target_names, num_friends)
            
            shift_predictions.append(pred_binary)
            shift_targets.append(target_binary)
            
            # Print first few examples
            if i < 3:
                print(f"\nSample {i+1}:")
                print(f"Target names: {target_names}")
                print(f"Target binary: {target_binary}")
                print(f"Predicted names: {pred_names}")
                print(f"Predicted binary: {pred_binary}")
                print(f"Generation (last 300 chars): ...{sample['generation'][-300:]}")
                
                # Show if prediction matches target
                if pred_names == target_names:
                    print("✓ EXACT MATCH!")
                else:
                    print("✗ No match")
                    print(f"  Missing from prediction: {set(target_names) - set(pred_names)}")
                    print(f"  Extra in prediction: {set(pred_names) - set(target_names)}")
        
        # Compute metrics for this shift
        if shift_predictions:
            metrics = compute_metrics(shift_predictions, shift_targets, num_friends)
            shift_metrics[shift] = metrics
            print(f"\n{shift} Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1']:.3f}")
            print(f"  Exact Matches: {metrics['exact_matches']}/{metrics['total_samples']}")
            
            all_predictions.extend(shift_predictions)
            all_targets.extend(shift_targets)
    
    # Overall metrics
    if all_predictions:
        overall_metrics = compute_metrics(all_predictions, all_targets, num_friends)
        print(f"\n{'='*80}")
        print("OVERALL METRICS:")
        print(f"  Accuracy: {overall_metrics['accuracy']:.3f}")
        print(f"  Precision: {overall_metrics['precision']:.3f}")
        print(f"  Recall: {overall_metrics['recall']:.3f}")
        print(f"  F1 Score: {overall_metrics['f1']:.3f}")
        print(f"  Exact Matches: {overall_metrics['exact_matches']}/{overall_metrics['total_samples']}")
        
        if has_extractions:
            print(f"  Extraction Method: {extraction_method}")
    
    return all_predictions, all_targets, num_friends, shift_metrics

def create_accuracy_chart(shift_metrics, file_path):
    """Create a line chart showing exact match accuracy by shift"""
    shifts = []
    accuracies = []
    
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        if shift in shift_metrics:
            shifts.append(shift.replace('_', ' ').title())
            accuracies.append(shift_metrics[shift]['accuracy'])
    
    # Extract subset and model name from filename
    import os
    filename = os.path.basename(file_path)
    # Extract subset and model name from filename like handsup_r1s7T5_gemini-2.5-pro.json or handsup_r2s20T10_gemini-2.5-pro.json
    import re
    subset_match = re.search(r'handsup_(r\d+s\d+T\d+)_', filename)
    subset_name = subset_match.group(1) if subset_match else 'Unknown Subset'
    
    model_match = re.search(r'handsup_r\d+s\d+T\d+_(.+)_extracted\.json$', filename)
    model_name = model_match.group(1) if model_match else 'Unknown Model'
    
    # Create the chart
    plt.figure(figsize=(12, 8))
    
    # Create line chart with markers
    line = plt.plot(shifts, accuracies, marker='o', linewidth=3, markersize=8, 
                   color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='#F18F01', 
                   markeredgewidth=2, label=model_name)
    
    # Customize the chart
    plt.title(f'{subset_name}', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Shift', fontsize=14, fontweight='bold')
    plt.ylabel('Exact Match Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, max(accuracies) * 1.15 if accuracies else 1)
    
    # Add value labels on data points
    for i, accuracy in enumerate(accuracies):
        plt.annotate(f'{accuracy:.3f}', 
                    (i, accuracy), 
                    textcoords="offset points", 
                    xytext=(0,15), 
                    ha='center', 
                    fontweight='bold',
                    fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Customize axes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the chart with subset name
    chart_filename = f"{subset_name}_{model_name}.pdf"
    plt.savefig(chart_filename, bbox_inches='tight')
    print(f"\nLine chart saved as: {chart_filename}")
    
    # Show the chart
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze handsup evaluation results with pre-computed extractions')
    parser.add_argument('--file', type=str, 
                       default="handsup_evals/handsup_r2s20T10_gemini-2.5-pro_extracted.json",
                       help='Path to the extracted JSON file to analyze')
    parser.add_argument('--dataset', type=str, choices=['r1s7T5', 'r2s20T10'], 
                       help='Dataset to analyze (r1s7T5 or r2s20T10)')
    parser.add_argument('--model', type=str, help='Model to analyze')
    
    args = parser.parse_args()
    
    # Set default file based on dataset if specified
    if args.dataset and args.model:
        file_path = f"handsup_evals/handsup_{args.dataset}_{args.model}_extracted.json"
    else:
        file_path = args.file
    
    print(f"Analyzing file: {file_path}")
    predictions, targets, num_friends, shift_metrics = analyze_results(file_path)
    
    print(f"\nBinary string length: {num_friends} (one bit per friend)")
    print(f"Total samples analyzed: {len(predictions)}")
    
    # Additional analysis
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS:")
    print(f"{'='*80}")
    
    # Count how many predictions have different numbers of friends
    pred_counts = defaultdict(int)
    target_counts = defaultdict(int)
    
    for pred, target in zip(predictions, targets):
        pred_count = pred.count('1')
        target_count = target.count('1')
        pred_counts[pred_count] += 1
        target_counts[target_count] += 1
    
    print(f"\nDistribution of number of friends predicted to raise hands:")
    for count in sorted(pred_counts.keys()):
        print(f"  {count:2d} friends: {pred_counts[count]:4d} samples ({pred_counts[count]/len(predictions)*100:.1f}%)")
    
    print(f"\nDistribution of number of friends who should raise hands (target):")
    for count in sorted(target_counts.keys()):
        print(f"  {count:2d} friends: {target_counts[count]:4d} samples ({target_counts[count]/len(predictions)*100:.1f}%)")
    
    # Show some examples of exact matches
    exact_matches = []
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        if pred == target:
            exact_matches.append(i)
    
    print(f"\nFirst 5 exact matches (out of {len(exact_matches)}):")
    for i, match_idx in enumerate(exact_matches[:5]):
        pred_names = binary_to_names(predictions[match_idx], num_friends)
        target_names = binary_to_names(targets[match_idx], num_friends)
        print(f"  Match {i+1}: {pred_names}")
    
    # Create chart showing exact match accuracy by shift
    create_accuracy_chart(shift_metrics, file_path)