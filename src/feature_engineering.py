import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def create_derived_features(df):
    # Calculate equipment age
    df['Equipment Age'] = df['Tool wear [min]'] / 60  # Assuming minutes are converted to hours

    # Calculate vibration metrics
    df['Rotational Speed RMS'] = df['Rotational speed [rpm]'].rolling(window=10).std()
    df['Torque RMS'] = df['Torque [Nm]'].rolling(window=10).std()

    # Calculate temperature gradient
    df['Temperature Gradient'] = df['Process temperature [K]'] - df['Air temperature [K]']

    return df

def encode_categorical_features(df):
    # One-hot encode categorical variables
    encoder = OneHotEncoder()
    product_id_encoded = encoder.fit_transform(df['Product ID'].values.reshape(-1, 1)).toarray()
    product_id_df = pd.DataFrame(product_id_encoded, columns=[f'Product ID_{i}' for i in range(product_id_encoded.shape[1])])
    df = pd.concat([df, product_id_df], axis=1)

    type_encoded = encoder.fit_transform(df['Type'].values.reshape(-1, 1)).toarray()
    type_df = pd.DataFrame(type_encoded, columns=[f'Type_{i}' for i in range(type_encoded.shape[1])])
    df = pd.concat([df, type_df], axis=1)

    return df

def engineer_features(df):
    df = create_derived_features(df)
    df = encode_categorical_features(df)
    return df

# Load the preprocessed data
df = pd.read_csv(r'CoDA\ML Projects\Predictive Maintance Model\data\processed\machine_failure_preprocessed.csv')

# Engineer the features
df = engineer_features(df)

# Save the engineered features
df.to_csv(r'CoDA\ML Projects\Predictive Maintance Model\data\processed\machine_failure_engineered.csv', index=False)