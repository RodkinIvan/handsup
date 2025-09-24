#!/usr/bin/env python3
"""
Script to analyze handsup dataset performance by Wolfram complexity classes.
Maps each sample to its ECA complexity class and creates bar charts showing
accuracy by complexity class for each model.
"""

import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import datasets
from wolfram_classes import full_map
from model_extractors import parse_target_string, ALL_FRIENDS
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import bootstrap

def calculate_confidence_interval_wilson(predictions, targets, confidence_level=0.95):
    """Calculate confidence interval for exact match accuracy using Wilson method."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    # Count exact matches (binary: 1 for match, 0 for no match)
    matches = []
    for pred, target in zip(predictions, targets):
        if pred == target:
            matches.append(1)
        else:
            matches.append(0)
    
    if not matches:
        return 0.0, 0.0, 0.0  # accuracy, lower_bound, upper_bound
    
    matches_array = np.array(matches)
    count = matches_array.sum()
    nobs = len(matches_array)
    
    # Calculate accuracy
    accuracy = count / nobs
    
    # Calculate Wilson confidence interval
    alpha = 1 - confidence_level
    lo, hi = proportion_confint(count=count, nobs=nobs, alpha=alpha, method="wilson")
    
    return accuracy, lo, hi

def calculate_confidence_interval_bootstrap(predictions, targets, confidence_level=0.95):
    """Calculate confidence interval for exact match accuracy using bootstrap method."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    # Count exact matches (binary: 1 for match, 0 for no match)
    matches = []
    for pred, target in zip(predictions, targets):
        if pred == target:
            matches.append(1)
        else:
            matches.append(0)
    
    if not matches:
        return 0.0, 0.0, 0.0  # accuracy, lower_bound, upper_bound
    
    matches_array = np.array(matches)
    
    # Calculate accuracy
    accuracy = matches_array.mean()
    
    # Bootstrap confidence interval
    res = bootstrap((matches_array,), np.mean, confidence_level=confidence_level, 
                   n_resamples=10000, method="BCa", random_state=0)
    lo, hi = res.confidence_interval.low, res.confidence_interval.high
    
    return accuracy, lo, hi

def calculate_confidence_interval(predictions, targets, confidence_level=0.95, method="wilson"):
    """Calculate confidence interval for exact match accuracy.
    
    Args:
        predictions: List of predicted binary strings
        targets: List of target binary strings  
        confidence_level: Confidence level (default 0.95 for 95%)
        method: Either "wilson" or "bootstrap"
    
    Returns:
        tuple: (accuracy, lower_bound, upper_bound)
    """
    if method == "wilson":
        return calculate_confidence_interval_wilson(predictions, targets, confidence_level)
    elif method == "bootstrap":
        return calculate_confidence_interval_bootstrap(predictions, targets, confidence_level)
    else:
        raise ValueError("Method must be either 'wilson' or 'bootstrap'")

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

def compute_accuracy(predictions, targets, num_friends=None):
    """Compute exact match accuracy"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    correct_predictions = 0
    
    for pred, target in zip(predictions, targets):
        # Convert binary strings to sets of names
        pred_set = set(binary_to_names(pred, num_friends))
        target_set = set(binary_to_names(target, num_friends))
        
        # Check if prediction is exactly correct
        if pred_set == target_set:
            correct_predictions += 1
    
    return correct_predictions / len(predictions)

def load_dataset_with_rules(subset):
    """Load the handsup dataset and extract rule information."""
    print(f"Loading handsup dataset: {subset}")
    dataset = datasets.load_dataset('irodkin/handsup', subset)
    
    # Process each shift to extract rules and map to complexity classes
    rule_to_class_mapping = {}
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
                'target_names': sample['answer_next_names'],
                'target_bits': sample['answer_next_bits'],
                'names': sample['names'],
                'width': sample['width']
            }
            
            sample_data[shift].append(sample_info)
            
            # Store rule to class mapping
            rule_to_class_mapping[rule_num] = complexity_class
    
    print(f"Loaded {len(sample_data['shift_1'])} samples per shift")
    print(f"Found {len(set(rule_to_class_mapping.values()))} unique complexity classes")
    
    # Print class distribution
    class_counts = defaultdict(int)
    for rule, cls in rule_to_class_mapping.items():
        class_counts[cls] += 1
    
    print("\nComplexity class distribution:")
    for cls in sorted(class_counts.keys(), key=lambda x: (0 if x == 'Unknown' else x)):
        print(f"  Class {cls}: {class_counts[cls]} rules")
    
    return sample_data, rule_to_class_mapping

def load_extracted_results(file_path, num_friends):
    """Load extracted model results from JSON file."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check if this file has pre-computed extractions
    has_extractions = False
    extraction_method = None
    if data.get('shift_1') and len(data['shift_1']) > 0:
        sample = data['shift_1'][0]
        if 'extracted_names' in sample:
            has_extractions = True
            extraction_method = sample.get('extraction_method', 'unknown')
    
    results = {}
    
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        results[shift] = []
        
        if shift in data:
            for sample in data[shift]:
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
                    # Extract model name from filename
                    filename = os.path.basename(file_path)
                    model_match = re.search(r'handsup_r\d+s\d+T\d+_(.+)\.json$', filename)
                    model_name = model_match.group(1) if model_match else 'gemini-2.5-pro'
                    pred_names = extract_names_from_text(sample['generation'], num_friends, model_name)
                
                # Convert to binary
                pred_binary = names_to_binary(pred_names, num_friends)
                target_binary = names_to_binary(target_names, num_friends)
                
                results[shift].append({
                    'pred_names': pred_names,
                    'target_names': target_names,
                    'pred_binary': pred_binary,
                    'target_binary': target_binary
                })
    
    return results, extraction_method

def calculate_class_baseline_scores_from_dataset(subset):
    """Calculate baseline scores (orbit=answer) for each complexity class from Hugging Face dataset."""
    print(f"Loading dataset {subset} to calculate Wolfram class baselines...")
    
    try:
        dataset = datasets.load_dataset('irodkin/handsup', subset)
    except Exception as e:
        print(f"Error loading dataset {subset}: {e}")
        return None
    
    # Load rule mapping to get complexity classes
    sample_data, rule_to_class_mapping = load_dataset_with_rules(subset)
    
    class_baselines = defaultdict(list)
    
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        if shift not in dataset:
            continue
        
        # Get corresponding samples from sample_data for complexity classes
        sample_data_shift = sample_data.get(shift, [])
        
        for i, sample in enumerate(dataset[shift]):
            try:
                # Get the last orbit state
                orbit_strings = sample.get('orbit_strings', [])
                if not orbit_strings:
                    continue
                
                last_orbit = orbit_strings[-1]
                
                # Get the answer bits
                answer_bits = sample.get('answer_next_bits', '')
                
                # Check if they match
                is_match = last_orbit == answer_bits
                
                # Get complexity class from sample_data
                if i < len(sample_data_shift):
                    complexity_class = sample_data_shift[i]['complexity_class']
                    class_baselines[complexity_class].append(is_match)
                
            except Exception as e:
                # Skip problematic samples
                continue
    
    # Calculate baseline ratio for each class
    class_baseline_ratios = {}
    for cls, matches in class_baselines.items():
        if len(matches) > 0:
            class_baseline_ratios[cls] = sum(matches) / len(matches)
            print(f"  Class {cls}: {sum(matches)}/{len(matches)} matches ({class_baseline_ratios[cls]:.3f})")
        else:
            class_baseline_ratios[cls] = 0.0
            print(f"  Class {cls}: No samples found")
    
    return class_baseline_ratios

def analyze_by_complexity_class(subset, model, data_dir="handsup_evals"):
    """Analyze model performance by Wolfram complexity class."""
    
    # Load dataset with rule information
    sample_data, rule_to_class_mapping = load_dataset_with_rules(subset)
    
    # Determine number of friends from subset name
    match = re.search(r's(\d+)', subset)
    num_friends = int(match.group(1)) if match else 7
    
    # Load model results
    possible_files = [
        f"{data_dir}/handsup_{subset}_{model}_extracted.json",
        f"{data_dir}/handsup_{subset}_{model}.json"
    ]
    
    file_path = None
    for possible_file in possible_files:
        if os.path.exists(possible_file):
            file_path = possible_file
            break
    
    if not file_path:
        print(f"No results file found for {subset} - {model}")
        return None
    
    print(f"Loading model results from: {file_path}")
    results, extraction_method = load_extracted_results(file_path, num_friends)
    
    if not results:
        print("No results loaded")
        return None
    
    # Combine dataset and results
    combined_data = {}
    
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        combined_data[shift] = []
        
        for i, (sample_info, result) in enumerate(zip(sample_data[shift], results[shift])):
            combined_sample = {
                'rule_num': sample_info['rule_num'],
                'complexity_class': sample_info['complexity_class'],
                'pred_binary': result['pred_binary'],
                'target_binary': result['target_binary'],
                'pred_names': result['pred_names'],
                'target_names': result['target_names']
            }
            combined_data[shift].append(combined_sample)
    
    # Calculate baseline scores for each complexity class from dataset
    class_baseline_ratios = calculate_class_baseline_scores_from_dataset(subset)
    
    # Calculate accuracy by complexity class
    class_accuracies = defaultdict(list)
    class_sample_counts = defaultdict(int)
    class_binary_data = defaultdict(lambda: {'predictions': [], 'targets': []})  # Store binary data for CI calculation
    
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        for sample in combined_data[shift]:
            complexity_class = sample['complexity_class']
            
            # Check if prediction matches target
            pred_set = set(sample['pred_names'])
            target_set = set(sample['target_names'])
            is_correct = pred_set == target_set
            
            class_accuracies[complexity_class].append(is_correct)
            class_sample_counts[complexity_class] += 1
            
            # Store binary data for confidence interval calculation
            class_binary_data[complexity_class]['predictions'].append(sample['pred_binary'])
            class_binary_data[complexity_class]['targets'].append(sample['target_binary'])
    
    # Calculate average accuracy per class
    class_avg_accuracies = {}
    for cls, accuracies in class_accuracies.items():
        class_avg_accuracies[cls] = sum(accuracies) / len(accuracies)
    
    print(f"\nAccuracy by complexity class for {subset} - {model}:")
    print("=" * 60)
    for cls in sorted(class_avg_accuracies.keys(), key=lambda x: (0 if x == 'Unknown' else x)):
        avg_acc = class_avg_accuracies[cls]
        baseline_ratio = class_baseline_ratios.get(cls, 0.0)
        sample_count = class_sample_counts[cls]
        print(f"Class {cls}: {avg_acc:.3f} (baseline: {baseline_ratio:.3f}) ({sample_count} samples)")
    
    return {
        'subset': subset,
        'model': model,
        'extraction_method': extraction_method,
        'class_accuracies': class_avg_accuracies,
        'class_baseline_ratios': class_baseline_ratios,
        'class_sample_counts': class_sample_counts,
        'class_binary_data': class_binary_data
    }

def create_complexity_class_chart(analysis_results, output_file="handsup_complexity.pdf", ci_method="wilson"):
    """Create bar charts showing accuracy by complexity class for each model."""
    
    if not analysis_results:
        print("No analysis results to plot")
        return
    
    # Get all unique complexity classes
    all_classes = set()
    for result in analysis_results:
        all_classes.update(result['class_accuracies'].keys())
    all_classes = sorted([cls for cls in all_classes if cls != 'Unknown'], key=lambda x: (0 if x == 'Unknown' else x))
    
    # Create subplots for each subset
    subsets = list(set(result['subset'] for result in analysis_results))
    # Sort subsets to ensure r1s7T5 comes first (left), then r2s20T10 (right)
    subsets = sorted(subsets, key=lambda x: (0 if x == 'r1s7T5' else 1))
    models = list(set(result['model'] for result in analysis_results))
    
    fig, axes = plt.subplots(1, len(subsets), figsize=(9, 8))
    if len(subsets) == 1:
        axes = [axes]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7', '#FF6B35', '#4ECDC4']
    
    for subplot_idx, subset in enumerate(subsets):
        ax = axes[subplot_idx]
        
        # Filter results for this subset
        subset_results = [r for r in analysis_results if r['subset'] == subset]
        
        # Prepare data for plotting
        x = np.arange(len(all_classes))
        n_cats = len(subset_results)
        bar_width = 0.7 / n_cats  # Adjusted bar width for spacing between bars
        group_spacing = 0.2  # Space between groups of bars
        
        for model_idx, result in enumerate(subset_results):
            model = result['model']
            class_binary_data = result['class_binary_data']
            
            # Calculate accuracies and confidence intervals for each class
            accuracies = []
            errors_lower = []
            errors_upper = []
            
            for cls in all_classes:
                if cls in class_binary_data:
                    predictions = class_binary_data[cls]['predictions']
                    targets = class_binary_data[cls]['targets']
                    
                    if predictions and targets:
                        accuracy, lo, hi = calculate_confidence_interval(predictions, targets, method=ci_method)
                        accuracies.append(accuracy)
                        errors_lower.append(accuracy - lo)
                        errors_upper.append(hi - accuracy)
                    else:
                        accuracies.append(0.0)
                        errors_lower.append(0.0)
                        errors_upper.append(0.0)
                else:
                    accuracies.append(0.0)
                    errors_lower.append(0.0)
                    errors_upper.append(0.0)
            
            baseline_ratios = [result['class_baseline_ratios'].get(cls, 0) for cls in all_classes]
            
            # Format model name for legend
            model_label = model.replace('_thinking_budget_10000', ' (thinking budget 10000)')
            model_label = model_label.replace('_thinking_budget_0', ' (thinking budget 0)')
            
            # Add thinking budget for gemini-2.5-pro
            if model_label == 'gemini-2.5-pro':
                model_label = 'gemini-2.5-pro (thinking budget 20000)'
            
            # Calculate bar positions with gaps
            positions = x + model_idx * (bar_width + group_spacing / n_cats) - (n_cats - 1) * (bar_width + group_spacing / n_cats) / 2
            
            # Create bars with error bars
            bars = ax.bar(positions, accuracies, bar_width, 
                         yerr=[errors_lower, errors_upper],
                         label=model_label, 
                         color=colors[model_idx % len(colors)],
                         alpha=0.8,
                         capsize=0,
                         error_kw={'elinewidth': 2},
                         zorder=3)
            
            # Value labels removed for cleaner visualization
        
        # Add baseline line for the first model (they should be the same for all models in same dataset)
        if subset_results:
            first_result = subset_results[0]
            baseline_ratios = [first_result['class_baseline_ratios'].get(cls, 0) for cls in all_classes]
            
            # Plot baseline stairs (scaled by 0.8) - extend from left to right edge
            baseline_ratios_scaled = [val * 0.8 for val in baseline_ratios]
            # Create extended x-coordinates to cover full width
            # Calculate the full width of bar groups
            total_bar_width = (bar_width + group_spacing / n_cats) * n_cats
            bar_centers = x
            x_stairs = np.concatenate([[bar_centers[0] - total_bar_width/2], bar_centers, [bar_centers[-1] + total_bar_width/2]])
            baseline_stairs = np.concatenate([[baseline_ratios_scaled[0]], baseline_ratios_scaled, [baseline_ratios_scaled[-1]]])
            ax.step(x_stairs, baseline_stairs, 
                   'r--', linewidth=2, alpha=0.7, label='Baseline (orbit[-1]=answer) × 0.8', where='mid')
        
        # Customize subplot
        ax.set_xlabel("Wolfram Complexity Class", fontsize=16)
        ax.set_ylabel("Exact match", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {cls}' for cls in all_classes], fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.grid(zorder=0)
        ax.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=2, fontsize=14, framealpha=0.9)
    
    # No title for cleaner appearance
    
    plt.tight_layout()
    
    # Save in both PDF and SVG formats
    output_file_pdf = output_file.replace('.pdf', '.pdf')
    output_file_svg = output_file.replace('.pdf', '.svg')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    plt.savefig(output_file_svg, bbox_inches='tight')
    print(f"\nComplexity class analysis chart saved as: {output_file_pdf} and {output_file_svg}")
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS:")
    print(f"{'='*80}")
    
    for result in analysis_results:
        subset = result['subset']
        model = result['model']
        class_accuracies = result['class_accuracies']
        
        print(f"\n{subset} - {model}:")
        class_baseline_ratios = result.get('class_baseline_ratios', {})
        for cls in sorted(class_accuracies.keys(), key=lambda x: (0 if x == 'Unknown' else x)):
            if cls != 'Unknown':
                baseline_ratio = class_baseline_ratios.get(cls, 0.0)
                baseline_ratio_scaled = baseline_ratio * 0.8
                print(f"  Class {cls}: {class_accuracies[cls]:.3f} (baseline: {baseline_ratio:.3f} × 0.8 = {baseline_ratio_scaled:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze handsup dataset performance by Wolfram complexity classes')
    parser.add_argument('--subset', type=str, default='r1s7T5',
                       help='Dataset subset to analyze (must be radius 1, e.g., r1s7T5)')
    parser.add_argument('--model', type=str, default='gemini-2.5-pro',
                       help='Model to analyze (default: gemini-2.5-pro)')
    parser.add_argument('--data_dir', type=str, default="handsup_evals",
                       help='Directory containing the JSON files (default: handsup_evals)')
    parser.add_argument('--output', type=str, default="handsup_complexity.pdf",
                       help='Output filename for the chart (default: handsup_complexity.pdf)')
    parser.add_argument('--all_models', action='store_true',
                       help='Analyze all available models')
    parser.add_argument('--ci_method', type=str, choices=['wilson', 'bootstrap'], default='bootstrap',
                       help='Confidence interval method: wilson (default) or bootstrap')
    
    args = parser.parse_args()
    
    # Determine which models and subsets to analyze
    if args.all_models:
        # Analyze all combinations (only radius 1 subsets for Wolfram classification)
        models = ['gemini-2.5-pro',  'gemini-2.5-flash_thinking_budget_10000', 'gemini-2.5-flash_thinking_budget_0', 'llama-3.3-70b', 'nemotron-32b', 'nemotron-7b',]
    else:
        models = [args.model]
    
    subsets = ['r1s7T5']  # Only ECA (radius 1) subsets
    # Run analysis for each combination
    analysis_results = []
    
    for subset in subsets:
        # Validate that subset is compatible with Wolfram classification
        if not subset.startswith('r1'):
            print(f"\nWarning: Wolfram complexity classes are only defined for elementary cellular automata (radius 1).")
            print(f"Skipping {subset} as it uses radius 2.")
            continue
            
        for model in models:
            print(f"\n{'='*80}")
            print(f"Analyzing {subset} - {model}")
            print(f"{'='*80}")
            
            result = analyze_by_complexity_class(subset, model, args.data_dir)
            if result:
                analysis_results.append(result)
    
    # Create combined chart
    if analysis_results:
        create_complexity_class_chart(analysis_results, args.output, args.ci_method)
    else:
        print("No analysis results to plot")
