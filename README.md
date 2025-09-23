# Handsup Game Analysis

This repository contains Python scripts for analyzing the performance of Large Language Models (LLMs) on the "Handsup Game" - a cellular automata-based reasoning task.

## Overview

The Handsup Game involves predicting which friends will raise their hands based on cellular automata rules. This project evaluates different LLMs' ability to extract the correct answer from their reasoning process.

## Key Features

- **LLM-based extraction**: Uses Gemma-3-12B for robust name extraction from model generations
- **Wolfram complexity analysis**: Analyzes performance by Elementary Cellular Automata complexity classes
- **Multi-model evaluation**: Supports Gemini, Llama, Nemotron, and Qwen models
- **Visualization**: Creates comprehensive charts showing accuracy across different shifts and complexity classes
- **Parallel processing**: Optimized extraction with parallel Ollama requests

## Core Scripts

### Analysis Scripts
- `extract_with_validation_parallel.py` - Main extraction script with parallel processing
- `draw_final_chart.py` - Creates comprehensive accuracy charts for multiple models
- `analyze_by_wolfram_class.py` - Analyzes performance by Wolfram complexity classes
- `draw_chart.py` - Individual model accuracy visualization

### Data Processing
- `convert_llama_json.py` - Converts Llama model results to unified format
- `convert_qwen3_json.py` - Converts Qwen3 model results to unified format

### Utilities
- `model_extractors.py` - Model-specific extraction strategies
- `wolfram_classes.py` - Wolfram complexity class mappings
- `restart_ollama_gpu.sh` - GPU-optimized Ollama startup script

## Usage

### Basic Extraction
```bash
# Extract names from all files in handsup_evals directory
python3 extract_with_validation_parallel.py --all

# Extract from specific file
python3 extract_with_validation.py --input handsup_evals/handsup_r1s7T5_gemini-2.5-pro.json
```

### Visualization
```bash
# Create comprehensive charts
python3 draw_final_chart.py --all

# Analyze by complexity classes
python3 analyze_by_wolfram_class.py --all_models

# Filter to hard classes only
python3 draw_final_chart.py --all --only_hard_classes
```

### Data Conversion
```bash
# Convert Llama results
python3 convert_llama_json.py

# Convert Qwen3 results
python3 convert_qwen3_json.py
```

## Requirements

- Python 3.8+
- ollama (for LLM extraction)
- matplotlib (for visualization)
- datasets (Hugging Face)
- numpy, tqdm, concurrent.futures

## Models Supported

- **Gemini 2.5 Pro/Flash** - Google's latest models
- **Llama 3.3 70B** - Meta's large language model
- **Nemotron 32B/7B** - NVIDIA's reasoning models
- **Qwen3 235B** - Alibaba's large model (with/without reasoning)

## Datasets

- **r1s7T5**: 7 friends, radius 1, 5 time steps
- **r2s20T10**: 20 friends, radius 2, 10 time steps

## Key Features

### LLM-based Extraction
Uses Gemma-3-12B with deterministic generation (temperature=0.0) for consistent name extraction from model reasoning.

### Complexity Analysis
Maps each sample to Wolfram complexity classes (1-4) for Elementary Cellular Automata rules, enabling analysis of how rule complexity affects model performance.

### Baseline Comparison
Compares model performance against a baseline where the last orbit state matches the answer state, providing a meaningful reference point.

### GPU Optimization
Optimized for GPU usage with Ollama, including environment variable configuration and parallel processing for maximum throughput.

## Output

The scripts generate:
- **PDF charts** showing accuracy across shifts and complexity classes
- **Extracted JSON files** with validated name extractions
- **Summary statistics** comparing model performance
- **Complexity class analysis** showing performance by rule difficulty

## License

This project is for research purposes. Please cite appropriately if used in academic work.
