# ModalSense: Multi-modal Sensitivity Analysis for LLM-guided Code Generation

Our research introduces a groundbreaking approach to analyzing the sensitivity of Large Language Models (LLMs) to variations in multi-modal prompts for code generation. By employing `ModalSens`, a novel algorithm, we quantify the effects of prompt changes—including text, code, and their structure—on the quality of generated code, leveraging delta-debugging principles to enhance model performance through precise prompt tuning. This study marks a significant stride in understanding and optimizing LLM interactions for code-related tasks, offering valuable insights into improving code generation accuracy and effectiveness.

## Installation

To install the required packages, use the following command:

```pip install -r requirements.txt```

This will install all the necessary packages required for this project.

## File Descriptions and Process Flow

### Main Execution
- **run_runtime_evaluation.py**: This script is designed for evaluating the runtime performance of various models. It includes functionalities to run test cases within a process with timeout control, parse command-line arguments, and measure execution time for each test case.
- **run_openai_model_experiments.py**: Utilizes OpenAI's models to generate outputs for a given dataset. It is equipped with argument parsing for script configuration and functions for loading datasets and generating outputs using OpenAI's API.
- **run_google_model_experiments.py**: Similar to the OpenAI experiments, this script is tailored for generating outputs using Google's Generative AI models. It provides argument setup for script execution, functions for output generation using Google's API, and code extraction from model outputs.
- **run_closed_source_model_experiments.py**: Focuses on evaluating closed source LLM models on specific datasets. It includes an argument parser setup to configure the number of prompts and other execution parameters.
- **run_analysis.py**: Dedicated to analyzing the results from model experiments. It categorizes errors, reads and prepares data from experiment outputs, and computes statistical results to evaluate model performance.

### Tools Submodule Files
- **evaluate_deltas.py**: Evaluates differences between model outputs and expected outputs, identifying discrepancies to understand model performance across datasets.
- **llm.py**: Provides an interface for interacting with LLMs, facilitating communication and response handling between experiment scripts and models.
- **models.py**: Defines and configures the LLMs used in experiments, enabling easy model selection and comparative analysis through standardized parameters.
- **process_wizardcoder.py**: Prepares data for WizardCoder model experiments, ensuring inputs are formatted for optimal evaluation accuracy.
- **wizardcoder_evaluate.sh**: Automates batch evaluations for the WizardCoder model, simplifying large-scale evaluation execution via a command-line interface.

### Adapters Submodule Files
- **mbpp_adapter.py**: Formats MBPP dataset prompts for compatibility with LLMs, automating the processing of test cases for coding task evaluations.
- **humaneval_adapter.py**: Converts HumanEval dataset challenges for LLM processing, assessing models' creative coding and problem-solving abilities.
- **general_adapter.py**: Provides a generalized approach to formatting prompts for a variety of datasets, ensuring flexibility in experiment design.
- **manual_prompts_comp.py**: Manages manually created prompts for specific experiments, allowing for customization in prompt design to analyze nuanced model responses.
