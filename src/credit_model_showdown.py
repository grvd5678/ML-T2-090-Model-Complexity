import time
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Fetch and Prepare Data
print("Loading and cleaning our financial data... 🏦")
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
X = credit_data.data
y = credit_data.target.map({'good': 1, 'bad': 0})

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['category', 'object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

X_processed = preprocessor.fit_transform(X)

# 2. Classic Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# --- The Simple Baseline (Logistic Regression) ---
start_time = time.time()
simple_model = LogisticRegression(max_iter=1000)
simple_model.fit(X_train, y_train)
simple_preds = simple_model.predict(X_test)
simple_time = time.time() - start_time
simple_acc = accuracy_score(y_test, simple_preds)

# --- The Complex Model (Random Forest) ---
start_time = time.time()
complex_model = RandomForestClassifier(n_estimators=100, random_state=42)
complex_model.fit(X_train, y_train)
complex_preds = complex_model.predict(X_test)
complex_time = time.time() - start_time
complex_acc = accuracy_score(y_test, complex_preds)

# 3. The Big Reveal!
print("\n" + "="*50)
print(f"SIMPLE MODEL (Logistic Regression)")
print(f"Accuracy: {simple_acc:.4f} | Time: {simple_time:.4f}s")
print("="*50)
print(classification_report(y_test, simple_preds, target_names=['Bad Credit (0)', 'Good Credit (1)']))

print("\n" + "="*50)
print(f"COMPLEX MODEL (Random Forest)")
print(f"Accuracy: {complex_acc:.4f} | Time: {complex_time:.4f}s")
print("="*50)
print(classification_report(y_test, complex_preds, target_names=['Bad Credit (0)', 'Good Credit (1)']))