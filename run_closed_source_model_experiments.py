import argparse
import json
import os
import pandas as pd
import pickle
import random
import shutil
from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash, write_jsonl
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
import warnings
from tools import llm, models

# Suppress all warnings
warnings.filterwarnings("ignore")

# Function Definitions

def parse_num_prompts(arg):
    """Parse the num_prompts argument to either a list of integers or a single integer."""
    if arg.startswith('[') and arg.endswith(']'):
        return json.loads(arg)
    return int(arg)

# Setup Argument Parser

parser = argparse.ArgumentParser(description="Script to evaluate LLM models on specific datasets.")
parser.add_argument("-m", "--model", default='hf_codellama_13B', help="LLM model name.")
parser.add_argument("-d", "--dataset", default='humaneval', choices=['humaneval', 'mbpp'], help="Dataset to use.")
parser.add_argument("-p", "--num_prompts", default=-1, help="Number of prompts to test or list of prompt numbers.")
parser.add_argument("-n", "--num_runs", default=1, help="Number of runs for each prompt.")
parser.add_argument("-g", "--delta_grouping", help="Grouping for generating delta: permutations or combinations.")
parser.add_argument("-exp", "--experiment", default='humaneval_codellama_13B')
parser.add_argument("-t", "--temperature", type=float, default=0.01)
parser.add_argument("--max_len", type=int, default=2048)
parser.add_argument("--greedy_decode", type=bool, default=True)
parser.add_argument("--decoding_style", type=str, default='sampling')
parser.add_argument("--save_embds", default=True, type=bool)
parser.add_argument("--save_modal_components", default=False, type=bool)
parser.add_argument("--modal_transformations", default=True, type=bool)

args = parser.parse_args()

# Main Logic

# Clean up the workspace
print('Deleting generated_code_files folder...')
if os.path.exists('generated_code_files'):
    shutil.rmtree('generated_code_files')

# Load dataset
if args.dataset == 'mbpp':
    problems, dataset_hash = get_mbpp_plus(), get_mbpp_plus_hash()
elif args.dataset == 'humaneval':
    problems, dataset_hash = get_human_eval_plus(), get_human_eval_plus_hash()
expected_output = get_groundtruth(problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS if args.dataset == 'mbpp' else [])

# Enhance problems with expected output
for problem in problems:
    problem['expected_output'] = expected_output.get(problem, {})

# Convert problems to DataFrame for easier manipulation
df = pd.DataFrame.from_dict(problems, orient='index').reset_index(drop=True)

# Determine which prompts to test
prompt_numbers = parse_num_prompts(args.num_prompts) if isinstance(args.num_prompts, str) else list(range(len(problems))) if args.num_prompts == -1 else random.sample(range(len(df)), args.num_prompts)

# Initialize model and tokenizer based on arguments
if 'hf' in args.model:
    model, tokenizer, generation_config = models.get_hf_model(args.model, args.temperature, args.max_len, args.greedy_decode, args.decoding_style) if args.save_embds or 'llama' not in args.model else models.get_hf_pipeline(args.model, args.temperature, args.max_len, args.greedy_decode, args.decoding_style)

# Execute LLM tests or embeddings generation
print("Running llm tests...")
results, embeds = [], {}
for prompt_number in tqdm(prompt_numbers, desc="Prompts completed"):
    try:
        # Conditionally save embeddings or model output
        if args.save_embds:
            embeds[prompt_number] = llm.gen_hf_model_embeds(model, tokenizer, args.dataset, prompt_number, args.delta_grouping, df)
        else:
            results += llm.gen_hf_model_output(model, tokenizer, generation_config, args.dataset, prompt_number, args.num_runs, args.delta_grouping, df, args.max_len, args.save_modal_components, args.model, args.modal_transformations)
    except Exception as e:
        print(f"Error in Prompt {prompt_number}: {e}")

# Save results
if args.save_embds:
    with open(f'{args.experiment}_embeds.pkl', 'wb') as file:
        pickle.dump(embeds, file)
else:
    write_jsonl(f'{args.experiment}_generated_code.jsonl', results)
