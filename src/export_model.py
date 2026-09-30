# import joblib
# from sklearn.datasets import fetch_openml
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression

# print("Fetching the OpenML German Credit data... 🏦")
# credit_data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
# X = credit_data.data
# y = credit_data.target.map({'good': 1, 'bad': 0})

# numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
# categorical_cols = X.select_dtypes(include=['category', 'object']).columns

# # 1. Our flawless, traditional preprocessing rules
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numerical_cols),
#         ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
#     ])

# # 2. Bundling the preprocessing and the winning model together!
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', LogisticRegression(max_iter=1000))
# ])

# print("Training the final Logistic Regression pipeline... 🧠")
# # We train on all the data now since this is the final production version
# pipeline.fit(X, y)

# # 3. Freezing the math for the cloud
# joblib.dump(pipeline, 'docs/credit_risk_model.joblib')
# print("Masterpiece successfully frozen and saved to docs/credit_risk_model.joblib! ❄️💅")

import os
import joblib
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

print("Fetching the OpenML German Credit data... 🏦")
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
X = credit_data.data
y = credit_data.target.map({'good': 1, 'bad': 0})

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['category', 'object']).columns

# 1. Traditional preprocessing rules
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

# 2. Bundling the preprocessing and the winning model together
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

print("Training the final Logistic Regression pipeline... 🧠")
pipeline.fit(X, y)

# 3. Dynamic path resolution to root docs/
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_path = os.path.join(base_dir, 'docs', 'credit_risk_model.joblib')

os.makedirs(os.path.join(base_dir, 'docs'), exist_ok=True)
joblib.dump(pipeline, output_path)
print(f"Masterpiece successfully frozen and saved to: {output_path} ❄️💅")