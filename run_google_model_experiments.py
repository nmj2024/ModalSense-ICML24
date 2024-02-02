import argparse
import pandas as pd
import time
import json
from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from adapters import mbpp_adapter, humaneval_adapter
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup the argument parser for command-line options
def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Script to generate outputs using Google's Generative AI models")
    parser.add_argument("-m", "--model", default='Google_GemP', help="LLM model name")
    parser.add_argument("-d", "--dataset", default='humaneval', choices=['humaneval', 'mbpp'], help="Dataset to use")
    parser.add_argument("-n", "--num_runs", type=int, default=1, help="Number of runs for each prompt")
    parser.add_argument("-c", "--chunk", default='all', help="Specific chunk of prompts to run")
    parser.add_argument("-t", "--type", choices=['base', 'transform'], default='base', help="Type of deltas to generate")
    parser.add_argument("-k", "--key", help="API key for Google")
    return parser.parse_args()

# Function to interact with Google's Generative AI
def generate_google_output(deltas, api_key):
    """
    Generates output from Google's Generative AI model for a batch of deltas.
    :param deltas: A list of prompt deltas.
    :param api_key: API key for authentication with Google's service.
    :return: Batch responses from the model.
    """
    client = ChatGoogleGenerativeAI(model="gemini-pro", max_tokens=2048, temperature=0, google_api_key=api_key)
    return client.batch(deltas)

# Function to extract Python code from LLM output
def extract_python_code(llm_output):
    """
    Extracts Python code from LLM output block.
    :param llm_output: String, LLM output containing a Python code block.
    :return: String, extracted Python code.
    """
    code_block = []
    in_code_block = False
    for line in llm_output.split('\n'):
        if (line.strip() == '```python' or line.strip() == '```') and not in_code_block:
            in_code_block = True
        elif line.strip() == '```' and in_code_block:
            break
        elif in_code_block:
            code_block.append(line)
    return '\n'.join(code_block)

# Main function to orchestrate the flow of execution
def main():
    args = setup_arg_parser()
    problems, dataset_hash = load_dataset(args.dataset)
    expected_output = get_groundtruth(problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS if args.dataset == 'mbpp' else [])
    enhance_problems_with_expected_output(problems, expected_output)
    df = prepare_dataframe(problems)
    chunk_start, chunk_end = determine_chunk(df, args.chunk)
    df_results, failed_prompts = run_experiments(df, chunk_start, chunk_end, args, generate_deltas_for_prompt)
    save_results(df_results, failed_prompts, args)

# Load dataset based on command-line argument
def load_dataset(dataset):
    if dataset == 'mbpp':
        return get_mbpp_plus(), get_mbpp_plus_hash()
    elif dataset == 'humaneval':
        return get_human_eval_plus(), get_human_eval_plus_hash()

# Enhance problems data with expected output
def enhance_problems_with_expected_output(problems, expected_output):
    for problem in problems:
        for key in expected_output[problem]:
            problems[problem][key] = expected_output[problem][key]

# Prepare the problems DataFrame
def prepare_dataframe(problems):
    return pd.DataFrame.from_dict(problems, orient='index').reset_index(drop=True)

# Determine the chunk of prompts to run
def determine_chunk(df, chunk):
    if chunk == 'all':
        return 0, len(df) - 1
    else:
        return map(int, chunk.split(','))

# Run experiments on a specific chunk of data
def run_experiments(df, chunk_start, chunk_end, args, generate_deltas_for_prompt_func):
    df_results = pd.DataFrame(columns=['task_id', 'run', 'completion', 'all_code'])
    failed_prompts = []
    for prompt_index in tqdm(range(chunk_start, chunk_end + 1), desc="Prompts completed"):
        task_id = df.iloc[prompt_index]['task_id']
        deltas = generate_deltas_for_prompt_func(df, prompt_index, args)
        process_prompts(deltas, task_id, df_results, args.num_runs, args.key, failed_prompts)
    return df_results, failed_prompts

# Process each prompt, generate output and handle exceptions
def process_prompts(deltas, task_id, df_results, num_runs, api_key, failed_prompts):
    for run_index in range(num_runs):
        try:
            all_code = generate_google_output(deltas, api_key)
            for response in all_code:
                content = response.content
                completion = extract_python_code(content)
                df_results.loc[len(df_results)] = [task_id, run_index, completion, content]
        except Exception as e:
            failed_prompts.append(task_id)
            for i in range(len(deltas)):
               df_results.loc[len(df_results)+i] = [task_id, run_index, '', '']

# Generate deltas based on dataset and type
def generate_deltas_for_prompt(df, prompt_index, args):
    if args.dataset == 'mbpp':
        return mbpp_adapter.generate_deltas(df, prompt_index, None, False, args.type == 'transform')
    elif args.dataset == 'humaneval':
        return humaneval_adapter.generate_deltas(df, prompt_index, None, False, args.type == 'transform')

# Save results and failed prompts to files
def save_results(df_results, failed_prompts, args):
    df_results.to_json(f"{args.model}_{args.dataset}_chunk_{args.chunk}_{args.num_runs}_runs.jsonl", orient='records', lines=True)
    with open(f"failed_prompts_{args.model}_{args.dataset}_chunk_{args.chunk}_{args.num_runs}_runs.json", 'w') as f:
        json.dump(failed_prompts, f)

if __name__ == "__main__":
    main()