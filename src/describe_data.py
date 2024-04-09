import pandas as pd

def describe_csv(file_path):
    """
    Loads a CSV file, describes its contents, and returns a summary of the dataset.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        str: A formatted string containing the dataset summary.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Get basic information about the DataFrame
        file_name = file_path.split('/')[-1]
        num_rows, num_columns = df.shape

        # Describe the DataFrame
        description = df.describe(include='all')

        # Extract data types
        data_types = description.dtypes.to_dict()
        data_types = {col: get_user_friendly_dtype(dtype) for col, dtype in data_types.items()}

        # Check for missing values
        missing_values = df.isnull().sum().to_dict()

        # Check for duplicates
        duplicates = df.duplicated().sum()

        # Generate the summary
        summary = f"1. Dataset name and format: {file_name} (CSV)\n"
        summary += f"2. Number of rows: {num_rows}\n"
        summary += f"3. Number of columns: {num_columns}\n"
        summary += "4. List of columns with their data types:\n"
        for column, dtype in data_types.items():
            summary += f"   - {column}: {dtype}\n"

        summary += "\n5. Assessment of data quality issues:\n"
        summary += "   - Missing values:\n"
        if sum(missing_values.values()) > 0:
            for col, num_missing in missing_values.items():
                if num_missing > 0:
                    summary += f"     {col}: {num_missing}\n"
        else:
            summary += "     No missing values found\n"

        summary += f"   - Potential duplicates: {duplicates}\n"

        # Statistical measures for numerical columns
        summary += "\n6. Statistical Measures:\n"
        for col in df.select_dtypes(include=['number']):
            summary += f"   - {col}:\n"
            summary += f"     Mean: {df[col].mean():.2f}\n"
            summary += f"     Median: {df[col].median():.2f}\n"
            summary += f"     Standard Deviation: {df[col].std():.2f}\n"

        return summary

    except FileNotFoundError:
        print(f"File '{file_path}' not found. Please provide a valid file path.")
        return None


def get_user_friendly_dtype(dtype):
    """
    Returns a user-friendly label for a given data type.

    Args:
        dtype (pandas.dtype): The data type of a column.

    Returns:
        str: The user-friendly label for the data type.
    """
    if dtype.kind == 'f':
        return 'decimal'
    elif dtype.kind == 'i':
        return 'whole number (integer)'
    elif dtype.kind == 'O':
        return 'text'
    else:
        return str(dtype)
    
# Example usage:
csv_file_path = r"CoDA\ML Projects\Predictive Maintance Model\data\raw\machine failure.csv"
dataset_summary = describe_csv(csv_file_path)
if dataset_summary:
    print(dataset_summary)