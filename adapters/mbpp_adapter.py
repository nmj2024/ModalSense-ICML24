import ast
import itertools
from . import general_adapter

def parse_function_inputs(input_str):
    """
    Parses the input string to extract function arguments by evaluating
    the structure safely using abstract syntax trees (AST).
    
    :param input_str: The input string containing function arguments.
    :return: A list of parsed arguments.
    """
    input_str = input_str.split('==')[0].strip()  # Get substring before the first '=='
    start_idx = input_str.find('(')
    end_idx = input_str.rfind(')')
    params_str = input_str[start_idx + 1:end_idx]
    tree = ast.parse(f"f({params_str})")
    args = tree.body[0].value.args
    inputs = [ast.literal_eval(arg) for arg in args]
    return inputs

def extract_mbpp_examples(prompt, start_word):
    """
    Extracts examples from the prompt starting from the given start word.
    
    :param prompt: The complete prompt text.
    :param start_word: The word indicating the start of examples.
    :return: Extracted examples as a string.
    """
    prompt = prompt.replace('"""', '')
    start_pos = prompt.find(start_word)
    examples = prompt[start_pos:].strip()
    return examples

def extract_mbpp_docstring(prompt, stop_word):
    """
    Extracts the docstring from the prompt up to the given stop word.
    
    :param prompt: The complete prompt text.
    :param stop_word: The word indicating the end of the docstring.
    :return: Extracted docstring as a string.
    """
    prompt = prompt.replace('"""', '')
    stop_pos = prompt.find(stop_word)
    docstring = prompt[:stop_pos].strip()
    return docstring

def extract_mbpp_test_list(entry_point, plus_input, expected_output):
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

def create_function_header(entry_point, canonical_solution):
    """
    Creates a function header from the entry point and the canonical solution.
    
    :param entry_point: The function name.
    :param canonical_solution: The canonical solution code.
    :return: The function header as a string.
    """
    for line in canonical_solution.split('\n'):
        if entry_point in line:
            return line
    return f'def {entry_point}():'  # Corrected to be a valid function definition

def transform_func_name(entry_point):
    """
    Transforms a snake_case function name to CamelCase.
    
    :param entry_point: The snake_case name of the function.
    :return: The CamelCase version of the function name.
    """
    if '_' in entry_point:
        func_elements = [i.capitalize() for i in entry_point.split('_')]
        return ''.join(func_elements)
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
    df = df[['prompt', 'entry_point', 'plus_input', 'plus', 'canonical_solution']].copy()
    prompt = str(df.iloc[prompt_index]['prompt'])
    entry_point = str(df.iloc[prompt_index]['entry_point'])
    canonical_solution = str(df.iloc[prompt_index]['canonical_solution'])
    function_header = create_function_header(entry_point, canonical_solution)
    docstring = extract_mbpp_docstring(prompt, 'assert')
    examples = extract_mbpp_examples(prompt, 'assert')
    normalized_function_header = function_header.replace(entry_point, 'func')

    if return_modal_components:
        return [
            str(df.iloc[prompt_index]['canonical_solution'])
        ]
    
    if modal_transformations:
        docstring_trans = docstring.replace('Write', 'Return')
        docstring_trans = docstring_trans.title()
        function_header_deadcode = f'{function_header}\n\tif False:\n\t\tx=[_ for i in range(42)]'
        entry_point_trans = transform_func_name(entry_point)
        function_header_name = function_header.replace(entry_point, entry_point_trans)
        examples_trans = examples.replace(entry_point, entry_point_trans)

        return [f'{function_header}\n"""\n{docstring_trans}\n{examples}\n"""\n',
                f'{function_header_deadcode}\n"""\n{docstring}\n{examples}\n"""\n',
                f'{function_header_name}\n"""\n{docstring.replace(entry_point, entry_point_trans)}\n{examples_trans}\n"""\n',
        ]

    return [f'{function_header}\n{prompt.strip()}',
            f'{function_header}\n"""\n{examples}\n"""\n',
            f'{docstring}\nCreate a function named {entry_point}\n{examples}\n',
            f'{normalized_function_header}\n"""\n{docstring}\n"""\n',
            f'\n{docstring}\n{examples}\n{function_header}\n',
            f'{docstring}\n{function_header}\n"""\n{examples}\n"""\n',
            f'{function_header}\n"""\n{examples}\n{docstring}\n"""\n',
        ]