import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the engineered data
df = pd.read_csv(r'..\data\processed\machine_failure_engineered.csv')

# Encode the categorical variables
label_encoder = LabelEncoder()
df['Product ID'] = label_encoder.fit_transform(df['Product ID'])
df['Type'] = label_encoder.fit_transform(df['Type'])

# Split the data into features and target
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, r'..\models\hgb_model.pkl')
