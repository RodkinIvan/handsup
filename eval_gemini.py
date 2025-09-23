import datasets
from openai import OpenAI
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
import argparse
from google import generativeai as genai
from google.generativeai.types.generation_types import BlockedPromptException, StopCandidateException
import time
import threading
import os

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 20000,
    "response_mime_type": "text/plain",
}
def get_gemini_response(prompt_data, model):
    shift, index, request = prompt_data
    chat_session = model.start_chat(
        history=[
        ]
    )
    try:
        response = chat_session.send_message(request, safety_settings="BLOCK_NONE")
    except BlockedPromptException:
        return shift, index,"Refused to answer"
    except StopCandidateException:
        return shift, index, "Prohibited answer"
    except Exception as e:
        print("Error occurred, sleeping. Error:", e)
        time.sleep(10)
        return get_gemini_response(prompt_data, model)
    try:
        return shift, index, response.text
    except Exception as e:
        print("Error occurred while returning the answer. Error:", e)
        print("response:", response)
        print("Sending \"Not enough reasoning tokens.\" as an output")
        return shift, index, "Not enough reasoning tokens."


def main(args):
    # subset = 'r2s20T10'
    subset = 'r1s7T5'
    dataset = datasets.load_dataset('irodkin/handsup', subset)

    model = genai.GenerativeModel(
            model_name=args.model_name,
            generation_config=generation_config,    
    )

    process_prompt = lambda prompt_data: get_gemini_response(prompt_data, model)

    MAX_WORKERS = 8
    prompts_to_process = []
    for shift in ['shift_1', 'shift_2', 'shift_3', 'shift_4']:
        for i in range(len(dataset[shift])):
            prompts_to_process.append((shift, i, dataset[shift][i]['prompt']))

    answers = {'shift_1': [None] * len(dataset['shift_1']),
               'shift_2': [None] * len(dataset['shift_2']),
               'shift_3': [None] * len(dataset['shift_3']),
               'shift_4': [None] * len(dataset['shift_4'])}

    # Create results folder if it doesn't exist
    os.makedirs(args.results_folder, exist_ok=True)
    
    output_file = f"{args.results_folder}/handsup_{subset}_{args.model_name}.json"
    
    # Create file lock for thread-safe writing
    file_lock = threading.Lock()
    
    def write_result_to_file(shift, index, content):
        with file_lock:
            answers[shift][index] = content
            with open(output_file, 'w') as f:
                json.dump(answers, f, indent=2)

    print(f"Processing a total of {len(prompts_to_process)} prompts in parallel...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks and get futures
        futures = [executor.submit(process_prompt, prompt_data) for prompt_data in prompts_to_process]
        
        # Process results as they complete
        with tqdm(total=len(futures), desc="Parallel API Calls") as pbar:
            for future in futures:
                shift, index, content = future.result()
                write_result_to_file(shift, index, content)
                pbar.update(1)

    print(f"All requests completed. Results saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--results_folder', type=str, default='./handsup_evals',
                        help='Folder to store results')
    parser.add_argument('--dataset_name', type=str, default='irodkin/handsup',
                        help='dataset name from huggingface')
    parser.add_argument('--model_name', type=str, default="gemini-1.5-pro-002", help='Name of the model to use')
    args = parser.parse_args()
    main(args)