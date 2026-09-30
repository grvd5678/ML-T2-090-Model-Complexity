import requests
from sklearn.datasets import fetch_openml

print("Fetching a real application from the dataset to test our API... 📥")
# Grab just the very first row of the dataset to ensure perfect formatting
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
sample_applicant = credit_data.data.iloc[0].to_dict()

print("Sending applicant data to the local AWS-ready endpoint... 🚀")
url = 'http://127.0.0.1:5000/predict'
response = requests.post(url, json=sample_applicant)

print("\n--- 🏦 BANK DECISION ---")
print(response.json())