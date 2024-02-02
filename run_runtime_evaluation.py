import argparse
import pandas as pd
import multiprocessing
import queue  # For communication between processes
from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from adapters import mbpp_adapter, humaneval_adapter
from tools import llm

def run_in_process(queue, file_name, test_list):
    """
    Function to execute test cases in a separate process.
    
    :param queue: Multiprocessing queue for communication between processes.
    :param file_name: The name of the file containing the code to test.
    :param test_list: List of test cases to run against the code.
    """
    result, error = llm.run_test_cases_for_file(file_name, test_list)
    queue.put((result, error))

def run_test_cases_with_timeout(file_name, test_list, timeout=180):
    """
    Wrapper function to execute test cases with a specified timeout.
    
    :param file_name: The name of the file containing the code to test.
    :param test_list: List of test cases to run against the code.
    :param timeout: Maximum time allowed for the tests to run, in seconds.
    :return: The result of test execution and any error that occurred.
    """
    q = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_in_process, args=(q, file_name, test_list))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        print("Timeout exceeded!")
        return "Fail", "Timeout Error (Exceeded 3 Minutes)"
    else:
        return q.get()

def parse_arguments():
    """
    Parses command-line arguments.
    
    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="JSONL file")
    parser.add_argument("-d", "--dataset", default='humaneval', help="Dataset", choices=['humaneval', 'mbpp'])
    return parser.parse_args()

def get_dataset_and_expected_output(dataset):
    """
    Retrieves dataset problems and their expected output based on the specified dataset.
    
    :param dataset: The name of the dataset ('mbpp' or 'humaneval').
    :return: Tuple containing problems, dataset hash, and expected output.
    """
    if dataset == 'mbpp':
        problems = get_mbpp_plus()
        dataset_hash = get_mbpp_plus_hash()
        expected_output = get_groundtruth(problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS)
    elif dataset == 'humaneval':
        problems = get_human_eval_plus()
        dataset_hash = get_human_eval_plus_hash()
        expected_output = get_groundtruth(problems, dataset_hash, [])
    return problems, dataset_hash, expected_output

def merge_test_lists_with_dataset(df_dataset, dataset, problems, expected_output):
    """
    Merges the dataset dataframe with test lists extracted for each problem.
    
    :param df_dataset: DataFrame of the dataset.
    :param dataset: The name of the dataset ('mbpp' or 'humaneval').
    :param problems: The problems extracted from the dataset.
    :param expected_output: The expected output for each problem.
    :return: DataFrame of the dataset merged with test lists.
    """
    test_list_dict = {}
    for i in range(len(df_dataset)):
        task_id = df_dataset.iloc[i]['task_id']
        entry_point = df_dataset.iloc[i]['entry_point']
        plus_input = df_dataset.iloc[i]['plus_input']
        plus = df_dataset.iloc[i]['plus']
        try:
            if dataset == 'mbpp':
                test_list = mbpp_adapter.extract_mbpp_test_list(entry_point, plus_input, plus)
            elif dataset == 'humaneval':
                test_list = humaneval_adapter.extract_humaneval_test_list(entry_point, plus_input, plus)
        except Exception as e:
            print(f"Error generating test list for task_id {task_id}: {e}")
            continue
        test_list_dict[task_id] = test_list
    test_list_df = pd.DataFrame(list(test_list_dict.items()), columns=['task_id', 'test_list'])
    return pd.merge(df_dataset, test_list_df, on='task_id', how='left')

def main():
    args = parse_arguments()
    
    # Load the JSONL file into a DataFrame
    df_llm_response = pd.read_json(args.file, lines=True)

    # Get the dataset problems and expected output
    problems, dataset_hash, expected_output = get_dataset_and_expected_output(args.dataset)

    # Add expected output to problems for further processing
    for problem_id, problem_data in problems.items():
        for key in expected_output[problem_id]:
            problem_data[key] = expected_output[problem_id][key]

    # Create a DataFrame from the problems dataset
    df_dataset = pd.DataFrame.from_dict(problems, orient='index').reset_index(drop=True)
    df_dataset = df_dataset[['task_id', 'entry_point', 'plus_input', 'plus']]
    
    # Merge test lists with the dataset
    df_dataset = merge_test_lists_with_dataset(df_dataset, args.dataset, problems, expected_output)

    # Run test cases for each LLM response and update the DataFrame with results
    for i in tqdm(range(len(df_llm_response)), desc="Processing deltas"):
        delta = f"delta_{i % 7}"
        with open('test.py', 'w') as file:
            file.write(df_llm_response.iloc[i]['completion'])
        task_id = df_llm_response.iloc[i]['task_id']
        test_list = df_dataset[df_dataset['task_id'] == task_id]['test_list'].values[0]
        result, error = run_test_cases_with_timeout('test.py', test_list)
        df_llm_response.at[i, 'delta'] = delta
        df_llm_response.at[i, 'result'] = result
        df_llm_response.at[i, 'error'] = error

    # Save the modified DataFrame back to a JSONL file
    df_llm_response.to_json(f"{args.file}_RUNTIME.jsonl", orient='records', lines=True)

if __name__ == '__main__':
    main()