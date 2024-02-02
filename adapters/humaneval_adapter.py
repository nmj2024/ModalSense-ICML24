from . import general_adapter
import itertools
from typing import Iterable, Dict
import gzip
import json
from adapters.manual_prompts_comp import humaneval_manual_prompt_dict
import sys
sys.set_int_max_str_digits(0)

def read_problems(evalset_file: str) -> Dict[str, Dict]:
    """
    Reads problems from a given evalset file and returns a dictionary
    with task IDs as keys and task details as values.
    
    :param evalset_file: Path to the evaluation set file.
    :return: Dictionary of task IDs to task details.
    """
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Streams content from a JSONL (JSON Lines) file, parsing each line
    as a JSON object and yielding it as a dictionary.
    
    :param filename: Path to the JSONL file.
    :return: Iterable of dictionaries representing each line in the file.
    """
    if filename.endswith(".gz"):
        with gzip.open(filename, 'rt') as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def extract_humaneval_examples(code, function_header, start_words):
    """
    Extracts human evaluation examples from a code snippet based on the function header
    and start words indicating where examples begin.
    
    :param code: The complete code snippet.
    :param function_header: The header of the function to extract examples for.
    :param start_words: List of words that indicate the start of an example.
    :return: Extracted examples as a string.
    """
    text = code.split(function_header)[1].strip()
    examples_text = ""
    recording = False

    for line in text.split('\n'):
        if any(start_word in line for start_word in start_words):
            recording = True  
        elif recording and (line.strip() == '' or line.strip().startswith('"""')):
            break
        if recording:
            examples_text += line + '\n'
    return examples_text.strip()

def extract_humaneval_docstring(code, function_header, stop_words):
    """
    Extracts the docstring from a code snippet based on the function header
    and stop words indicating where the docstring ends.
    
    :param code: The complete code snippet.
    :param function_header: The header of the function to extract docstring for.
    :param stop_words: List of words that indicate the end of a docstring.
    :return: Extracted docstring as a string.
    """
    text = code.split(function_header)[1].strip()
    for stop_word in stop_words:
        if stop_word in text:
            text = text.split(stop_word)[0]
    return text

def extract_humaneval_test_list(entry_point, plus_input, expected_output):
    """
    Prepares a list of test assertions for a given entry point, input, and expected output.
    
    :param entry_point: The name of the function to test.
    :param plus_input: List of inputs for each test case.
    :param expected_output: Expected output for each test case.
    :return: List of test assertions as strings.
    """
    def prepare_input(inp):
        return ', '.join([str(i) for i in inp])
    return [f'assert {entry_point}({prepare_input(i)}) == {str(j)}' for i, j in zip(plus_input, expected_output)]

def transform_func_name(entry_point):
    """
    Transforms a snake_case function name to CamelCase.
    
    :param entry_point: The snake_case name of the function.
    :return: The CamelCase version of the function name.
    """
    if '_' in entry_point:
        func_elements = entry_point.split('_')
        return ''.join(i.capitalize() for i in func_elements)
    return entry_point.capitalize()


def generate_deltas(df, prompt_index, delta_method, return_modal_components, modal_transformations):
    """
    Generate deltas based on the provided DataFrame, prompt index, and delta method.
    This function is designed to facilitate testing and evaluation by generating
    variations of the problem descriptions and solutions.
    
    :param df: DataFrame containing the necessary data.
    :param prompt_index: The index of the prompt in the DataFrame.
    :param delta_method: Method for generating deltas ('permutations' or 'combinations').
    :param return_modal_components: Flag indicating whether to return modal components.
    :param modal_transformations: Flag indicating whether to perform modal transformations.
    :return: A list of deltas or modal components, depending on flags.
    """
    df = df[['prompt', 'entry_point', 'test', 'plus_input', 'plus', 'canonical_solution']].copy()
    prompt = str(df.iloc[prompt_index]['prompt'])
    entry_point = str(df.iloc[prompt_index]['entry_point'])

    if prompt_index in humaneval_manual_prompt_dict.keys():
        function_header, docstring, examples = humaneval_manual_prompt_dict[prompt_index]
        docstring = docstring.strip().replace('"""', '').replace("'''", "")
        examples = examples.strip().replace('"""', '').replace("'''", "")
    else:
        function_header = str(general_adapter.extract_function_header(prompt, entry_point))
        docstring = extract_humaneval_docstring(prompt, function_header, ['For example', 'For Example', 'Example', 'example', '>>>', '>>', f'\n{entry_point}'])
        examples = prompt.split(docstring)[1].strip().replace('"""', '').replace("'''", "")
        docstring = docstring.strip().replace('"""', '').replace("'''", "")
    normalized_function_header = function_header.replace(entry_point, 'func')

    if return_modal_components:
        return [
            str(df.iloc[prompt_index]['canonical_solution'])
        ]
    
    if modal_transformations:
        docstring_trans = docstring.title()
        function_header_deadcode = f'{function_header}\n\tif False:\n\t\tx=[_ for i in range(42)]'
        entry_point_trans = transform_func_name(entry_point)
        function_header_name = function_header.replace(entry_point, entry_point_trans)
        examples_trans = examples.replace(entry_point, entry_point_trans)

        return [f'{function_header}\n"""\n{docstring_trans}\n{examples}\n"""\n',
                f'{function_header_deadcode}\n"""\n{docstring}\n{examples}\n"""\n',
                f'{function_header_name}\n"""\n{docstring.replace(entry_point, entry_point_trans)}\n{examples_trans}\n"""\n',
        ]

    return [f'{prompt}',
            f'{function_header}\n"""\n{examples}\n"""\n',
            f'{docstring}\nCreate a function named {entry_point}\n{examples}\n',
            f'{normalized_function_header}\n"""\n{docstring}\n"""\n',
            f'\n{docstring}\n{examples}\n{function_header}\n',
            f'{docstring}\n{function_header}\n"""\n{examples}\n"""\n',
            f'{function_header}\n"""\n{examples}\n{docstring}\n"""\n',
        ]