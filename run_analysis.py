import argparse
import pandas as pd
from tabulate import tabulate

def categorize_error(error):
    """
    Categorizes the given error into predefined categories.
    
    Parameters:
    - error: The error message as a string.
    
    Returns:
    - A string representing the category of the error.
    """
    if error is None or pd.isna(error):
        return "No Error"
    elif error.startswith("Syntax Error"):
        return "Syntax Errors"
    elif error.startswith("Semantic Error"):
        return "Semantic Errors"
    elif error.startswith("Runtime Error"):
        return "Runtime Errors"
    else:
        return "Other Errors"

def read_and_prepare_data(file_path):
    """
    Reads the JSONL file into a DataFrame and prepares the data for analysis.
    
    Parameters:
    - file_path: Path to the JSONL file.
    
    Returns:
    - A pandas DataFrame with the data from the JSONL file.
    """
    return pd.read_json(file_path, lines=True)

def compute_result_statistics(df):
    """
    Computes statistics on test results and prints a summary table.
    
    Parameters:
    - df: DataFrame containing the test results.
    """
    result_counts = df['result'].value_counts().reset_index()
    result_counts.columns = ['Result', 'Count']
    result_counts['Percent'] = (result_counts['Count'] / result_counts['Count'].sum()) * 100
    print("Result Counts:")
    print(tabulate(result_counts, headers='keys', tablefmt='grid'))
    print("\n")

def compute_error_statistics(df):
    """
    Computes statistics on errors and prints a summary table.
    
    Parameters:
    - df: DataFrame containing error data.
    """
    error_counts = df['error'].value_counts().reset_index()
    error_counts.columns = ['Error Type', 'Count']
    error_counts['Percent'] = (error_counts['Count'] / error_counts['Count'].sum()) * 100
    print("Error Counts:")
    print(tabulate(error_counts, headers='keys', tablefmt='grid'))
    print("\n")

def compute_and_display_error_categories(df):
    """
    Categorizes errors, computes frequencies, and prints detailed error analysis.
    
    Parameters:
    - df: DataFrame containing error data.
    """
    df['error_category'] = df['error'].apply(categorize_error)
    filtered_df = df[df['error_category'] != "No Error"]
    
    # Categorized Error Percentages per Delta
    delta_error_stats = filtered_df.groupby(['delta', 'error_category']).size().unstack(fill_value=0)
    delta_error_percentages = delta_error_stats.div(delta_error_stats.sum(axis=1), axis=0) * 100
    print("Categorized Error Percentages per Delta:")
    print(tabulate(delta_error_percentages, headers='keys', tablefmt='grid', floatfmt=".2f"))
    print("\n")
    
    # Most Common Error per Delta
    most_common_errors = filtered_df.groupby('delta')['error'].agg(lambda x: x.value_counts().idxmax())
    most_common_errors_counts = filtered_df.groupby('delta')['error'].agg(lambda x: x.value_counts().max())
    most_common_errors_df = pd.DataFrame({'Delta': most_common_errors.index, 'Most Common Error': most_common_errors.values, 'Count': most_common_errors_counts.values})
    print("Most Common Error per Delta and Frequency:")
    print(tabulate(most_common_errors_df, headers='keys', tablefmt='grid'))
    print("\n")

def compute_sensitivity_per_delta(df):
    """
    Computes and displays the sensitivity of error categorization across different deltas.
    
    Parameters:
    - df: DataFrame containing categorized error data.
    """
    df['error_category'] = df['error'].apply(categorize_error)
    filtered_df = df[df['error_category'] != "No Error"]
    delta_error_stats = filtered_df.groupby(['delta', 'error_category']).size().unstack(fill_value=0)
    delta_error_percentages = delta_error_stats.div(delta_error_stats.sum(axis=1), axis=0) * 100
    sensitivity = delta_error_percentages.subtract(delta_error_percentages.loc['delta_0'], axis=1)
    print("Categorized Sensitivity per Delta:")
    print(tabulate(sensitivity, headers='keys', tablefmt='grid', floatfmt=".2f"))

def write_analysis_to_file(df, output_file):
    """
    Writes the analysis results to a text file.
    
    Parameters:
    - df: DataFrame containing the analyzed data.
    - output_file: Path to the output text file.
    """
    with open(output_file, 'w') as file:
        # Result and error counts
        file.write("Result Counts:\n")
        file.write(tabulate(df['result'].value_counts().reset_index(), headers='keys', tablefmt='grid'))
        file.write("\n\nError Counts:\n")
        file.write(tabulate(df['error'].value_counts().reset_index(), headers='keys', tablefmt='grid'))
        file.write("\n\n")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="JSONL file")
    args = parser.parse_args()

    df = read_and_prepare_data(args.file)
    compute_result_statistics(df)
    compute_error_statistics(df)
    compute_and_display_error_categories(df)
    compute_sensitivity_per_delta(df)
    write_analysis_to_file(df, f"{args.file}_ANALYSIS.txt")

if __name__ == "__main__":
    main()