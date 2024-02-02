import argparse
import pandas as pd
from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from adapters import general_adapter, mbpp_adapter, humaneval_adapter
from openai import OpenAI

# Setup the argument parser for command-line options
parser = argparse.ArgumentParser(description="Script to generate outputs using OpenAI's models")
parser.add_argument("-m", "--model", default='OpenAI_4T', help="LLM model name")
parser.add_argument("-d", "--dataset", default='humaneval', choices=['humaneval', 'mbpp'], help="Dataset to use")
parser.add_argument("-n", "--num_runs", type=int, default=1, help="Number of runs for each prompt")
parser.add_argument("-c", "--chunk", default='all', help="Specific chunk of prompts to run")
parser.add_argument("-t", "--type", choices=['base', 'transform'], default='base', help="Type of deltas to generate")
parser.add_argument("-k", "--key", help="API key for OpenAI")
args = parser.parse_args()

# Model mapping for OpenAI models
model_dict = {
    'OpenAI_3.5T': 'gpt-3.5-turbo',
    'OpenAI_4': 'gpt-4',
    'OpenAI_4T': 'gpt-4-0125-preview'
}

# Function to generate output from OpenAI model
def generate_openai_output(delta):
    """
    Generates output from OpenAI model for a given delta.

    :param delta: A string representing the delta for which to generate output.
    :return: The model's response to the delta.
    """
    question = delta
    client = OpenAI(api_key=args.key)
    response = client.chat.completions.create(
        model=model_dict[args.model],
        messages=[{'role': 'user', 'content': question}],
        max_tokens=2048,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Main process to fetch dataset, generate outputs, and save results
def main():
    # Load dataset based on command-line argument
    problems, dataset_hash = load_dataset()
    expected_output = get_groundtruth(problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS if args.dataset == 'mbpp' else [])

    # Enhance problems with expected output
    enhance_problems_with_expected_output(problems, expected_output)

    # Convert problems to DataFrame
    df = pd.DataFrame.from_dict(problems, orient='index').reset_index(drop=True)

    # Determine chunk of prompts to run
    chunk_start, chunk_end = determine_chunk(df)

    # Running experiments on selected chunk
    df_results = run_experiments(df, chunk_start, chunk_end)

    # Save experiment results
    save_results(df_results)

def load_dataset():
    if args.dataset == 'mbpp':
        return get_mbpp_plus(), get_mbpp_plus_hash()
    elif args.dataset == 'humaneval':
        return get_human_eval_plus(), get_human_eval_plus_hash()

def enhance_problems_with_expected_output(problems, expected_output):
    for problem in problems:
        for key in expected_output[problem]:
            problems[problem][key] = expected_output[problem][key]

def determine_chunk(df):
    if args.chunk == 'all':
        return 0, len(df) - 1
    else:
        return map(int, args.chunk.split(','))

def run_experiments(df, chunk_start, chunk_end):
    df_results = pd.DataFrame(columns=['task_id', 'run', 'completion', 'all_code'])
    print(f"Running prompts {chunk_start} to {chunk_end} for {args.dataset} dataset")

    for prompt_index in tqdm(range(chunk_start, chunk_end + 1), desc="Prompts completed"):
        task_id = df.iloc[prompt_index]['task_id']
        deltas = generate_deltas(df, prompt_index)

        for run_index in range(args.num_runs):
            print(f"Run {run_index + 1} of {args.num_runs} for Prompt {prompt_index}")
            df_results = process_deltas(df_results, deltas, task_id, run_index)

    return df_results

def generate_deltas(df, prompt_index):
    if args.dataset == 'mbpp':
        return mbpp_adapter.generate_deltas(df, prompt_index, None, False, args.type == 'transform')
    elif args.dataset == 'humaneval':
        return humaneval_adapter.generate_deltas(df, prompt_index, None, False, args.type == 'transform')

def process_deltas(df_results, deltas, task_id, run_index):
    for delta_index, delta in enumerate(deltas):
        print(f"Generating Output for Delta {delta_index + 1} of {len(deltas)}")
        all_code = generate_openai_output(delta)
        completion = general_adapter.extract_python_code(all_code)
        row = {'task_id': task_id, 'run': run_index, 'completion': completion, 'all_code': all_code}
        df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
    return df_results  # Return the updated DataFrame

def save_results(df_results):
    print(df_results.head(2))
    df_results.to_json(f"{args.model}_{args.dataset}_chunk_{args.chunk}_{args.num_runs}_runs.jsonl", orient='records', lines=True)

if __name__ == "__main__":
    main()