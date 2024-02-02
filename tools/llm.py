import pandas as pd
import os
import importlib.util
import timeout_decorator
import models
from adapters import general_adapter, mbpp_adapter, humaneval_adapter

# Define a timeout duration for test cases
TEST_CASE_TIMEOUT = 30  # Example: 30 seconds

def run_single_test_case(module, test):
    """
    Run a single test case.

    :param module: The Python module in which the test will be executed.
    :param test: A string representing the test case to be executed.
    """
    exec(test, globals(), module.__dict__)

@timeout_decorator.timeout(TEST_CASE_TIMEOUT)
def run_test_case_with_timeout(module, test):
    """
    Execute a test case with a specified timeout.

    :param module: The Python module in which the test will be executed.
    :param test: A string representing the test case to be executed.
    """
    run_single_test_case(module, test)

def run_test_cases(module, test_list):
    """
    Run a list of test cases for a given module, handling timeouts and assertions.

    :param module: The Python module to test.
    :param test_list: A list of test case strings to be executed.
    :return: True if all tests pass, False otherwise.
    """
    for test in test_list:
        try:
            run_test_case_with_timeout(module, test)
        except AssertionError:
            return False
        except timeout_decorator.TimeoutError:
            print("Test case timed out. Moving to next test case.")
    return True

def run_test_cases_for_file(file_path, test_list):
    """
    Run test cases for a Python file.

    :param file_path: Path to the Python file to be tested.
    :param test_list: A list of test case strings to be executed.
    :return: A tuple with the test result ('Pass' or 'Fail') and an error type if applicable.
    """
    try:
        # Check if the file is empty
        if os.path.getsize(file_path) == 0 or not open(file_path).read().strip():
            return ('Fail', 'No Code Generated')

        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if run_test_cases(module, test_list):
            return ('Pass', None)
        else:
            # If the test cases run but fail, it's a semantic error
            return ('Fail', 'Semantic Error (Test Case)')
    except SyntaxError:
        return ('Fail', 'Syntax Error')
    except AssertionError:
        # If an assertion fails, it's considered a semantic error
        return ('Fail', 'Semantic Error (Assertion)')
    except timeout_decorator.TimeoutError:
        return ('Fail', 'Timeout')
    except MemoryError:
        return ('Fail', 'Resource Error')
    except ImportError:
        return ('Fail', 'Dependency Error')
    except EnvironmentError:
        return ('Fail', 'Environment Error')
    except Exception as e:
        return ('Fail', f'Runtime Error - {e.__class__.__name__}')


def gen_hf_model_output(model, tokenizer, generation_config, dataset, prompt_index, 
                        num_runs, delta_method, df, max_len, save_modal_components, model_name, modal_transformations):
    """
    Generate model output for a given prompt index across multiple runs.

    :param model: The model object used for generating output.
    :param tokenizer: The tokenizer object corresponding to the model.
    :param generation_config: Configuration parameters for the generation process.
    :param dataset: The name of the dataset ('mbpp' or 'humaneval').
    :param prompt_index: Index of the prompt in the dataset for which output is generated.
    :param num_runs: Number of times to generate output for the prompt.
    :param delta_method: Method used for generating variations (deltas) of the prompt.
    :param df: DataFrame containing the dataset.
    :param max_len: Maximum length of the generated output.
    :param save_modal_components: Flag indicating whether modal components should be saved.
    :param model_name: Name of the model being used.
    :param modal_transformations: Flag indicating whether modal transformations are applied.
    :return: A list of dictionaries with the results of the generation.
    """
    all_results = []

    if dataset == 'mbpp':
        deltas= mbpp_adapter.generate_deltas(df, prompt_index, delta_method, save_modal_components, modal_transformations)
    elif dataset == 'humaneval':
        deltas = humaneval_adapter.generate_deltas(df, prompt_index, delta_method, save_modal_components, modal_transformations)    
    
    if save_modal_components:
        return [{f'delta_{i}':j for i,j in enumerate(deltas)}]
    
    for run_index in range(num_runs):
        for i, delta in enumerate(deltas):
            if 'wizardcoder' in model_name:
                trucated_seq, raw_seq = models.generate_wizardcode_output(delta, model, tokenizer, generation_config, max_len)
            elif 'codellama' in model_name:
                trucated_seq, raw_seq = models.generate_llama_output(delta, model, tokenizer, generation_config, max_len, model_name)
            all_results.append(dict(task_id = f'HumanEval/{prompt_index}', completion = trucated_seq, all_code = raw_seq))

    return all_results

def gen_hf_model_embeds(model, tokenizer, dataset, prompt_index, delta_method, df):
    """
    Generate embeddings for model prompts based on the dataset, prompt index, and delta method.

    :param model: The model object used for generating embeddings.
    :param tokenizer: The tokenizer object corresponding to the model.
    :param dataset: The name of the dataset ('mbpp' or 'humaneval').
    :param prompt_index: Index of the prompt in the dataset for which embeddings are generated.
    :param delta_method: Method used for generating variations (deltas) of the prompt.
    :param df: DataFrame containing the dataset.
    :return: A list of embeddings for each delta of the prompt.
    """
    all_results = []

    if dataset == 'mbpp':
        deltas= mbpp_adapter.generate_deltas(df, prompt_index, delta_method, True)
    elif dataset == 'humaneval':
        deltas = humaneval_adapter.generate_deltas(df, prompt_index, delta_method, True)    
    
    
    for i, delta in enumerate(deltas):
        delta_embedding = models.get_hf_model_embedding(delta, model, tokenizer)
        all_results.append(delta_embedding)

    return all_results