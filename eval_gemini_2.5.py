import datasets
from openai import OpenAI
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
import argparse
from google import generativeai as genai
from google.generativeai.types.generation_types import BlockedPromptException, StopCandidateException
from google.genai import types
from google import genai
import time
import threading
import os

THINKING_BUDGET = 0
MODEL_NAME = None

def get_gemini_response(prompt_data, client):
    global MODEL_NAME
    shift, index, request, target = prompt_data
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=request,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
            ),
        )
    except BlockedPromptException:
        return shift, index, "Refused to answer", target
    except StopCandidateException:
        return shift, index, "Prohibited answer", target
    except Exception as e:
        print("Error occurred, sleeping. Error:", e)
        time.sleep(10)
        return get_gemini_response(prompt_data, client)
    try:
        return shift, index, response.text, target
    except Exception as e:
        print("Error occurred while returning the answer. Error:", e)
        print("response:", response)
        print("Sending \"Not enough reasoning tokens.\" as an output")
        return shift, index, "Not enough reasoning tokens.", target


def main(args):
    subset = 'r2s20T10'
    # subset = 'r1s7T5'
    dataset = datasets.load_dataset('irodkin/handsup', subset)

    # model = genai.GenerativeModel(
    #         model_name=args.model_name,
    #         generation_config=generation_config,    
    # )
    client = genai.Client()

    process_prompt = lambda prompt_data: get_gemini_response(prompt_data, client)

    MAX_WORKERS = 8
    prompts_to_process = []
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        for i in range(len(dataset[shift])):
            # Get target answers and convert to comma-separated string
            target_answers = dataset[shift][i]['answer_next_names']
            prompt = dataset[shift][i]['prompt']
            if isinstance(target_answers, list):
                target_str = ', '.join(target_answers)
            else:
                target_str = str(target_answers)
            prompts_to_process.append((shift, i, prompt, target_str))

    answers = {'shift_1': [{'generation': None, 'target': None}] * len(dataset['shift_1']),
               'shift_2': [{'generation': None, 'target': None}] * len(dataset['shift_2']),
               'shift_3': [{'generation': None, 'target': None}] * len(dataset['shift_3']),
               'shift_4': [{'generation': None, 'target': None}] * len(dataset['shift_4'])}

    # Create results folder if it doesn't exist
    os.makedirs(args.results_folder, exist_ok=True)
    
    output_file = f"{args.results_folder}/handsup_{subset}_{args.model_name}_thinking_budget_{THINKING_BUDGET}.json"
    
    # Create file lock for thread-safe writing
    file_lock = threading.Lock()
    
    def write_result_to_file(shift, index, generation, target):
        with file_lock:
            answers[shift][index] = {
                'generation': generation,
                'target': target
            }
            with open(output_file, 'w') as f:
                json.dump(answers, f, indent=2)

    print(f"Processing a total of {len(prompts_to_process)} prompts in parallel...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks and get futures
        futures = [executor.submit(process_prompt, prompt_data) for prompt_data in prompts_to_process]
        
        # Process results as they complete
        with tqdm(total=len(futures), desc="Parallel API Calls") as pbar:
            for future in futures:
                shift, index, generation, target = future.result()
                write_result_to_file(shift, index, generation, target)
                pbar.update(1)

    print(f"All requests completed. Results saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--results_folder', type=str, default='./handsup_evals',
                        help='Folder to store results')
    parser.add_argument('--dataset_name', type=str, default='irodkin/handsup',
                        help='dataset name from huggingface')
    args = parser.parse_args()
    args.model_name = "gemini-2.5-flash"
    MODEL_NAME = args.model_name
    main(args)