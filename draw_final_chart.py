#!/usr/bin/env python3
"""
Script to create a combined chart showing accuracy across multiple models and datasets.
Draws all lines on a single chart for easy comparison.
"""

import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import datasets
from model_extractors import parse_target_string, ALL_FRIENDS
from wolfram_classes import full_map

# Configuration - modify these lists to add/remove models and datasets
MODELS = ['gemini-2.5-pro', 'gemini-2.5-flash_thinking_budget_10000', 'gemini-2.5-flash_thinking_budget_0', 'llama-3.3-70b', 'nemotron-32b', 'nemotron-7b']
DATASETS = ['r1s7T5', 'r2s20T10']

# Color palette for different lines
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7', '#FF6B35', '#4ECDC4']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

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

def calculate_baseline_scores_from_dataset(subset):
    """Calculate baseline scores based on orbit state matching answer state from Hugging Face dataset."""
    print(f"Loading dataset {subset} to calculate baseline...")
    
    try:
        dataset = datasets.load_dataset('irodkin/handsup', subset)
    except Exception as e:
        print(f"Error loading dataset {subset}: {e}")
        return None
    
    baseline_scores = {}
    
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        if shift not in dataset:
            continue
        
        matches = 0
        total_count = 0
        
        for sample in dataset[shift]:
            try:
                # Get the last orbit state
                orbit_strings = sample.get('orbit_strings', [])
                if not orbit_strings:
                    continue
                
                last_orbit = orbit_strings[-1]
                
                # Get the answer bits
                answer_bits = sample.get('answer_next_bits', '')
                
                # Check if they match
                if last_orbit == answer_bits:
                    matches += 1
                
                total_count += 1
                
            except Exception as e:
                # Skip problematic samples
                continue
        
        # Calculate baseline ratio
        if total_count > 0:
            baseline_ratio = matches / total_count
            baseline_scores[shift] = baseline_ratio
            print(f"  {shift}: {matches}/{total_count} matches ({baseline_ratio:.3f})")
        else:
            baseline_scores[shift] = 0.0
            print(f"  {shift}: No valid samples found")
    
    return baseline_scores

def calculate_baseline_scores_from_dataset_hard_classes(subset):
    """Calculate baseline scores based on orbit state matching answer state from Hugging Face dataset, only for hard classes (3,4)."""
    print(f"Loading dataset {subset} to calculate baseline for hard classes only...")
    
    try:
        dataset = datasets.load_dataset('irodkin/handsup', subset)
    except Exception as e:
        print(f"Error loading dataset {subset}: {e}")
        return None
    
    # Load dataset with rules to get complexity classes
    sample_data = load_dataset_with_rules(subset)
    
    baseline_scores = {}
    
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        if shift not in dataset:
            continue
        
        matches = 0
        total_count = 0
        
        for i, sample in enumerate(dataset[shift]):
            try:
                # Check if this sample belongs to hard classes (3 or 4)
                if i < len(sample_data[shift]):
                    complexity_class = sample_data[shift][i]['complexity_class']
                    if complexity_class not in [3, 4]:
                        continue  # Skip non-hard classes
                else:
                    continue
                
                # Get the last orbit state
                orbit_strings = sample.get('orbit_strings', [])
                if not orbit_strings:
                    continue
                
                last_orbit = orbit_strings[-1]
                
                # Get the answer bits
                answer_bits = sample.get('answer_next_bits', '')
                
                # Check if they match
                if last_orbit == answer_bits:
                    matches += 1
                
                total_count += 1
                
            except Exception as e:
                # Skip problematic samples
                continue
        
        # Calculate baseline ratio
        if total_count > 0:
            baseline_ratio = matches / total_count
            baseline_scores[shift] = baseline_ratio
            print(f"  {shift} (hard classes only): {matches}/{total_count} matches ({baseline_ratio:.3f})")
        else:
            baseline_scores[shift] = 0.0
            print(f"  {shift} (hard classes only): No valid samples found")
    
    return baseline_scores

def extract_rule_from_table(rule_table):
    """Extract the ECA rule number from the rule table string."""
    # The rule table format is:
    # NEIGHBOURHOOD ORDER: [left 1 .. left 1, self, right 1 .. right 1] (circular)
    # 
    # TRUTH TABLE (pattern -> next bit):
    # 111 -> 1
    # 110 -> 0
    # 101 -> 1
    # ...
    
    # Parse the truth table to get the 8-bit rule
    rule_bits = [0] * 8
    lines = rule_table.strip().split('\n')
    
    for line in lines:
        if '->' in line and len(line.strip().split('->')) == 2:
            # Extract pattern and result
            parts = line.split('->')
            if len(parts) == 2:
                pattern = parts[0].strip()
                try:
                    result = int(parts[1].strip())
                    
                    # Convert pattern to index (111=7, 110=6, 101=5, 100=4, 011=3, 010=2, 001=1, 000=0)
                    if len(pattern) == 3 and pattern.isdigit():
                        pattern_val = int(pattern, 2)
                        rule_bits[7 - pattern_val] = result
                except ValueError:
                    # Skip lines that don't have valid integer results
                    continue
    
    # Convert bits to rule number
    rule_num = 0
    for i, bit in enumerate(rule_bits):
        rule_num |= (bit << i)
    
    return rule_num

def load_dataset_with_rules(subset):
    """Load the handsup dataset and extract rule information."""
    print(f"Loading handsup dataset: {subset}")
    dataset = datasets.load_dataset('irodkin/handsup', subset)
    
    # Process each shift to extract rules and map to complexity classes
    sample_data = {}
    
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        sample_data[shift] = []
        
        for i, sample in enumerate(dataset[shift]):
            # Extract rule from rule_table
            rule_num = extract_rule_from_table(sample['rule_table'])
            
            # Map rule to complexity class
            complexity_class = full_map.get(rule_num, 'Unknown')
            
            # Store sample data
            sample_info = {
                'rule_num': rule_num,
                'complexity_class': complexity_class,
                'index': i
            }
            
            sample_data[shift].append(sample_info)
    
    print(f"Loaded {len(sample_data['shift_1'])} samples per shift")
    
    # Print class distribution
    class_counts = defaultdict(int)
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        for sample_info in sample_data[shift]:
            class_counts[sample_info['complexity_class']] += 1
    
    print("\nComplexity class distribution:")
    for cls in sorted(class_counts.keys(), key=lambda x: (0 if x == 'Unknown' else x)):
        print(f"  Class {cls}: {class_counts[cls]} samples")
    
    return sample_data

def analyze_single_file(file_path, only_hard_classes=False, dataset_name=None):
    """Analyze results from a single JSON file"""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None, None, None, None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Parse filename to get the number of friends
    filename = os.path.basename(file_path)
    match = re.search(r's(\d+)', filename)
    if match:
        num_friends = int(match.group(1))
    else:
        num_friends = 20  # fallback
    
    # Load dataset with rules if filtering by hard classes
    sample_data = None
    if only_hard_classes and dataset_name:
        sample_data = load_dataset_with_rules(dataset_name)
    
    # Check if this file has pre-computed extractions
    has_extractions = False
    extraction_method = None
    if data.get('shift_1') and len(data['shift_1']) > 0:
        sample = data['shift_1'][0]
        if 'extracted_names' in sample:
            has_extractions = True
            extraction_method = sample.get('extraction_method', 'unknown')
    
    all_predictions = []
    all_targets = []
    shift_metrics = {}
    
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        shift_predictions = []
        shift_targets = []
        
        if shift in data:
            for i, sample in enumerate(data[shift]):
                if sample['generation'] is None or sample['target'] is None:
                    continue
                
                # Filter by hard classes if requested
                if only_hard_classes and sample_data:
                    if i < len(sample_data[shift]):
                        complexity_class = sample_data[shift][i]['complexity_class']
                        # Only include classes 3 or 4
                        if complexity_class not in [3, 4]:
                            continue
                    else:
                        continue
                
                # Get target names
                target_names = parse_target_string(sample['target'])
                
                # Get predicted names - use pre-computed extractions if available
                if has_extractions and 'extracted_names' in sample:
                    pred_names = sample['extracted_names']
                else:
                    # Fallback to rule-based extraction if no pre-computed extractions
                    from model_extractors import extract_names_from_text
                    # Extract model name from filename
                    model_match = re.search(r'handsup_r\d+s\d+T\d+_(.+)\.json$', filename)
                    model_name = model_match.group(1) if model_match else 'gemini-2.5-pro'
                    pred_names = extract_names_from_text(sample['generation'], num_friends, model_name)
                
                # Convert to binary
                pred_binary = names_to_binary(pred_names, num_friends)
                target_binary = names_to_binary(target_names, num_friends)
                
                shift_predictions.append(pred_binary)
                shift_targets.append(target_binary)
        
        # Compute metrics for this shift
        if shift_predictions:
            metrics = compute_metrics(shift_predictions, shift_targets, num_friends)
            shift_metrics[shift] = metrics
            all_predictions.extend(shift_predictions)
            all_targets.extend(shift_targets)
    
    return shift_metrics, num_friends, filename, extraction_method

def create_separate_charts(data_dir="handsup_evals", output_prefix="handsup", only_hard_classes=False):
    """Create separate charts for each dataset showing accuracy across multiple models"""
    
    # Collect data for all model-dataset combinations
    all_data = {}
    
    print("Collecting data from files...")
    print("=" * 80)
    
    for dataset in DATASETS:
        for model in MODELS:
            # Try different possible file extensions
            possible_files = [
                f"{data_dir}/handsup_{dataset}_{model}_extracted.json",
                f"{data_dir}/handsup_{dataset}_{model}.json"
            ]
            
            file_path = None
            for possible_file in possible_files:
                if os.path.exists(possible_file):
                    file_path = possible_file
                    break
            
            if file_path:
                print(f"Processing: {file_path}")
                shift_metrics, num_friends, filename, extraction_method = analyze_single_file(file_path, only_hard_classes, dataset)
                
                if shift_metrics:
                    all_data[f"{dataset}_{model}"] = {
                        'shift_metrics': shift_metrics,
                        'num_friends': num_friends,
                        'filename': filename,
                        'extraction_method': extraction_method,
                        'dataset': dataset,
                        'model': model
                    }
                    print(f"  ✓ Loaded data for {dataset} - {model}")
                else:
                    print(f"  ✗ No data found in {file_path}")
            else:
                print(f"  ✗ File not found for {dataset} - {model}")
                print(f"    Tried: {possible_files}")
    
    if not all_data:
        print("No data files found! Please check the file paths and ensure files exist.")
        return
    
    # Create separate charts for each dataset
    for dataset in DATASETS:
        # Filter data for this dataset
        dataset_data = {k: v for k, v in all_data.items() if v['dataset'] == dataset}
        
        if not dataset_data:
            print(f"No data found for dataset: {dataset}")
            continue
        
        # Calculate baseline scores from the original dataset
        if only_hard_classes:
            baseline_scores = calculate_baseline_scores_from_dataset_hard_classes(dataset)
        else:
            baseline_scores = calculate_baseline_scores_from_dataset(dataset)
        
        # Create chart for this dataset
        plt.figure(figsize=(8, 6))
        
        shifts = [1, 2, 3, 4]
        color_idx = 0
        
        # Plot lines for each model in this dataset
        for key, data in dataset_data.items():
            model = data['model']
            shift_metrics = data['shift_metrics']
            extraction_method = data.get('extraction_method', 'rule-based')
            
            accuracies = []
            for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
                if shift in shift_metrics:
                    accuracies.append(shift_metrics[shift]['accuracy'])
                else:
                    accuracies.append(0.0)
            
            # Create label with model and extraction method
            if extraction_method and extraction_method != 'rule-based' and 'gemma3' not in extraction_method:
                label = f"{model} ({extraction_method})"
            else:
                label = model
            
            # Format model names - replace thinking budget patterns
            label = label.replace('_thinking_budget_10000', ' (thinking budget 10000)')
            label = label.replace('_thinking_budget_0', ' (thinking budget 0)')
            
            # Add thinking budget for gemini-2.5-pro
            if label == 'gemini-2.5-pro':
                label = 'gemini-2.5-pro (thinking budget 20000)'
            
            color = COLORS[color_idx % len(COLORS)]
            marker = MARKERS[color_idx % len(MARKERS)]
            
            plt.plot(shifts, accuracies, 
                    marker='o', 
                    linewidth=2, 
                    markersize=6,
                    color=color,
                    label=label,
                    zorder=3)
            
            # Value labels removed for cleaner visualization
            
            color_idx += 1
        
        # Calculate maximum accuracy for y-limit
        max_accuracy = 0.0
        for key, data in dataset_data.items():
            shift_metrics = data['shift_metrics']
            for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
                if shift in shift_metrics:
                    max_accuracy = max(max_accuracy, shift_metrics[shift]['accuracy'])
        
        # Set y-limit with some padding above the maximum
        y_max = min(1.0, max_accuracy * 1.15) if max_accuracy > 0 else 1.0
        
        # Plot baseline dash line
        if baseline_scores:
            baseline_values = []
            for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
                baseline_values.append(baseline_scores.get(shift, 0.0))
            
            # Plot baseline dash line (scaled by 0.8)
            baseline_values_scaled = [val * 0.8 for val in baseline_values]
            plt.plot(shifts, baseline_values_scaled, 
                    linestyle='--', 
                    linewidth=2, 
                    color='red', 
                    alpha=0.7,
                    label='Baseline (orbit[-1]=answer) × 0.8',
                    zorder=2)
        plt.title(f'{dataset}', fontsize=14, fontweight='bold')
        
        # Add grid for better readability
        plt.grid(True, zorder=0, alpha=0.3)
        
        # Customize the chart
        plt.xlabel("Look-ahead, steps", fontsize=12)
        plt.ylabel("Exact match", fontsize=12)
        # plt.ylim(0, 1.0)
        
        # Customize axes
        plt.xticks([1, 2, 3, 4], fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add legend above the chart with proper spacing
        plt.legend(fontsize=11, loc='lower center',
                  bbox_to_anchor=(0.5, 1.03),
                  ncol=3,
                  frameon=False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart with dataset name in both PDF and SVG formats
        suffix = "_hard" if only_hard_classes else ""
        output_file_pdf = f"{output_prefix}_{dataset}{suffix}.pdf"
        output_file_svg = f"{output_prefix}_{dataset}{suffix}.svg"
        plt.savefig(output_file_pdf, bbox_inches='tight')
        plt.savefig(output_file_svg, bbox_inches='tight')
        print(f"\nChart for {dataset} saved as: {output_file_pdf} and {output_file_svg}")
        
        # Show the chart
        plt.show()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS:")
    print(f"{'='*80}")
    
    for key, data in all_data.items():
        dataset = data['dataset']
        model = data['model']
        shift_metrics = data['shift_metrics']
        
        # Calculate overall accuracy
        all_accuracies = [shift_metrics[shift]['accuracy'] for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4'] if shift in shift_metrics]
        avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
        
        print(f"\n{dataset} - {model}:")
        print(f"  Average Accuracy: {avg_accuracy:.3f}")
        
        for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
            if shift in shift_metrics:
                baseline_val = baseline_scores.get(shift, 0.0) if baseline_scores else 0.0
                baseline_val_scaled = baseline_val * 0.8
                print(f"  {shift.replace('_', ' ').title()}: {shift_metrics[shift]['accuracy']:.3f} (baseline: {baseline_val:.3f} × 0.8 = {baseline_val_scaled:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create combined accuracy chart for multiple models and datasets')
    parser.add_argument('--data_dir', type=str, default="handsup_evals",
                       help='Directory containing the JSON files (default: handsup_evals)')
    parser.add_argument('--output', type=str, default="handsup.pdf",
                       help='Output filename for the chart (default: handsup.pdf)')
    parser.add_argument('--models', type=str, nargs='+', default=MODELS,
                       help=f'Models to include (default: {MODELS})')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS,
                       help=f'Datasets to include (default: {DATASETS})')
    parser.add_argument('--only_hard_classes', action='store_true',
                       help='Filter to only include samples with Wolfram complexity classes 3 or 4')
    
    args = parser.parse_args()
    
    # Update global variables with command line arguments
    MODELS = args.models
    DATASETS = args.datasets
    
    print(f"Models: {MODELS}")
    print(f"Datasets: {DATASETS}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output file: {args.output}")
    print(f"Only hard classes: {args.only_hard_classes}")
    
    create_separate_charts(args.data_dir, args.output.replace('.pdf', ''), args.only_hard_classes)

