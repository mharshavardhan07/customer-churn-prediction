import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from flask import Flask, request, jsonify, render_template

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/Telecom_customer_churn.csv')

# Strip spaces and lowercase column names
df.columns = df.columns.str.strip().str.lower()

# Print dataset columns for debugging
print("Dataset Columns:", df.columns.tolist())

# Identify the correct 'churn' column dynamically
target_col = None
for col in df.columns:
    if "churn" in col.lower():
        target_col = col
        break

if target_col is None:
    raise KeyError("Churn column not found in dataset. Check dataset columns.")

# Convert target variable to binary
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

# Convert TotalCharges to numeric (handling missing values)
if 'totalcharges' in df.columns:
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Splitting data into train & test
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing numerical data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save Model
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Flask App
app = Flask(__name__)

# Load Model
def load_model():
    global model
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)[0]
        return jsonify({'churn_prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
