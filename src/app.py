import joblib
import pandas as pd
from flask import Flask, request, jsonify

print("Spinning up the classic prediction server... 🚀")
app = Flask(__name__)

# Load our frozen masterpiece
pipeline = joblib.load('docs/credit_risk_model.joblib')

@app.route('/predict', methods=['POST'])
def predict_risk():
    try:
        applicant_data = request.get_json()
        # Convert JSON directly into a traditional DataFrame
        df = pd.DataFrame([applicant_data])
        
        # Run the math
        prediction = pipeline.predict(df)[0]
        confidence = pipeline.predict_proba(df).max()
        
        # Map our binary output back to business logic
        result = "Good Credit Risk" if prediction == 1 else "Bad Credit Risk"
        
        return jsonify({
            "status": "success",
            "prediction": result,
            "confidence_score": round(confidence, 4)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # Running strictly locally for now
    app.run(port=5000, debug=True)