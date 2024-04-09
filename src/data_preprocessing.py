import pandas as pd

# Load the data
df = pd.read_csv(r'..\data\raw\machine failure.csv')

# Check for missing values
print(df.isnull().sum())

# Handle missing values (if any)
# Since there are no missing values, we can skip this step

# Convert data types (if necessary)
df['UDI'] = df['UDI'].astype(float)
df['Machine failure'] = df['Machine failure'].astype(int)
df['TWF'] = df['TWF'].astype(int)
df['HDF'] = df['HDF'].astype(int)
df['PWF'] = df['PWF'].astype(int)
df['OSF'] = df['OSF'].astype(int)
df['RNF'] = df['RNF'].astype(int)

# Check for duplicates
print(f"Number of duplicates: {df.duplicated().sum()}")

# Drop duplicates (if any)
# Since there are no duplicates, we can skip this step

# Save the preprocessed data
df.to_csv('..\data\processed\machine_failure_preprocessed.csv', index=False)
