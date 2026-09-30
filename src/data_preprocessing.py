import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Fetching the classic dataset
print("Loading our financial data... 🏦")
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
X = credit_data.data
y = credit_data.target

# 2. Converting the target to classic binary (1 = good, 0 = bad)
y = y.map({'good': 1, 'bad': 0})

# 3. Separating our feature types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['category', 'object']).columns

# 4. Building the traditional preprocessing pipeline
print("Applying old-school data transformations... 🔄")
preprocessor = ColumnTransformer(
    transformers=[
        # Standardize numbers (mean of 0, variance of 1)
        ('num', StandardScaler(), numerical_cols),
        # Convert categories to 1s and 0s, dropping the first to avoid dummy traps!
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

# 5. Execute the transformation!
X_processed = preprocessor.fit_transform(X)

print(f"\nOriginal feature count: {X.shape[1]}")
print(f"New encoded feature count: {X_processed.shape[1]}")
print("Data is officially clean, scaled, and ready for modeling! 💅")