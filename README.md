# ML-T2-090: Credit Risk Assessment

## Overview
This project evaluates the business trade-offs between a traditional, easily explainable baseline model (Logistic Regression) and a complex ensemble model (Random Forest) for predicting credit risk in highly regulated banking environments.

## Architecture & Deployment
- **Local API:** Built using a classic Flask server.
- **Model Serialization:** Handled via `joblib` for pipeline freezing.
- **Cloud Ready:** Designed for serverless deployment via AWS Lambda & API Gateway.

## How to Run Locally
1. Install the exact environment dependencies:
   `pip install -r requirements.txt`
2. Spin up the local prediction server:
   `python src/app.py`
3. In a separate terminal, test the API:
   `python src/test_api.py`

## Conclusion
The traditional, simpler baseline model provided better business-critical recall for the minority class while maintaining the strict explainability legally required by financial institutions.
