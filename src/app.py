import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

print("Spinning up the classic prediction server... 🚀")
app = Flask(__name__)

# Load our frozen masterpiece
pipeline = joblib.load('docs/credit_risk_model.joblib')

LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ML-T2-090: Credit Risk Inference API</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(17, 24, 39, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --text-main: #f9fafb;
      --text-muted: #9ca3af;
      --success: #10b981;
      --danger: #ef4444;
      --badge-bg: rgba(59, 130, 246, 0.15);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: radial-gradient(circle at 50% -20%, #1e293b, var(--bg) 80%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 2rem 1rem;
      line-height: 1.5;
    }
    .container {
      max-width: 1040px;
      margin: 0 auto;
    }
    header {
      text-align: center;
      margin-bottom: 2.5rem;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.35rem 0.9rem;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 600;
      margin-bottom: 1rem;
      letter-spacing: 0.02em;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 10px #10b981;
    }
    h1 {
      font-size: 2.2rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-bottom: 0.5rem;
      background: linear-gradient(135deg, #ffffff 30%, #94a3b8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 1.05rem;
      max-width: 650px;
      margin: 0 auto;
    }
    .grid-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.2rem;
      backdrop-filter: blur(12px);
      transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
      transform: translateY(-2px);
      border-color: rgba(59, 130, 246, 0.4);
    }
    .metric-label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.35rem;
    }
    .metric-val {
      font-size: 1.45rem;
      font-weight: 700;
      color: #ffffff;
    }
    .metric-sub {
      font-size: 0.75rem;
      color: #60a5fa;
      margin-top: 0.2rem;
    }
    .main-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 2rem;
      backdrop-filter: blur(16px);
      margin-bottom: 2rem;
      box-shadow: 0 20px 40px -15px rgba(0,0,0,0.5);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1.5rem;
      padding-bottom: 1.2rem;
      border-bottom: 1px solid var(--card-border);
    }
    .card-title {
      font-size: 1.25rem;
      font-weight: 700;
    }
    .sample-buttons {
      display: flex;
      gap: 0.6rem;
    }
    .btn-sample {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 0.45rem 0.85rem;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-sample:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: var(--primary);
    }
    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    .input-group {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }
    label {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
    }
    input, select {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 0.55rem 0.75rem;
      border-radius: 8px;
      font-family: inherit;
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.2s;
    }
    input:focus, select:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    .btn-predict {
      width: 100%;
      background: linear-gradient(135deg, var(--primary), #1d4ed8);
      color: white;
      border: none;
      padding: 0.85rem;
      font-size: 1rem;
      font-weight: 700;
      border-radius: 10px;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
    }
    .btn-predict:hover {
      background: linear-gradient(135deg, var(--primary-hover), #1e40af);
      transform: translateY(-1px);
      box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.4);
    }
    .result-box {
      margin-top: 1.5rem;
      padding: 1.25rem;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid var(--card-border);
      display: none;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .result-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
    }
    .result-badge {
      padding: 0.4rem 1rem;
      border-radius: 9999px;
      font-weight: 700;
      font-size: 0.95rem;
      display: inline-block;
    }
    .badge-good {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
    }
    .badge-bad {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #f87171;
    }
    .confidence-meter {
      height: 8px;
      background: rgba(255,255,255,0.08);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 0.5rem;
    }
    .confidence-fill {
      height: 100%;
      background: var(--primary);
      width: 0%;
      transition: width 0.5s ease;
    }
    .api-code {
      font-family: 'JetBrains Mono', monospace;
      background: #060911;
      padding: 1rem;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.06);
      font-size: 0.82rem;
      color: #e2e8f0;
      overflow-x: auto;
      margin-top: 0.75rem;
    }
    footer {
      text-align: center;
      color: var(--text-muted);
      font-size: 0.85rem;
      margin-top: 2rem;
      padding-top: 1rem;
      border-top: 1px solid var(--card-border);
    }
    footer a {
      color: #60a5fa;
      text-decoration: none;
    }
    footer a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="status-pill">
        <span class="status-dot"></span>
        API Online & Ready
      </div>
      <h1>Credit Risk Assessment API</h1>
      <p class="subtitle">ML-T2-090: Empirical comparison proving simple interpretable models outperform complex ensembles on business-critical recall.</p>
    </header>

    <div class="grid-metrics">
      <div class="metric-card">
        <div class="metric-label">Selected Production Model</div>
        <div class="metric-val">Logistic Regression</div>
        <div class="metric-sub">Linear & Fully Explainable</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Business Recall (Bad Credit)</div>
        <div class="metric-val">54.0%</div>
        <div class="metric-sub">+10% higher than Random Forest (44%)</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Inference Latency</div>
        <div class="metric-val">0.0165 s</div>
        <div class="metric-sub">14.6x faster engineering efficiency</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Compliance Mandate</div>
        <div class="metric-val">FCRA & ECOA</div>
        <div class="metric-sub">Instant Adverse Action transparency</div>
      </div>
    </div>

    <div class="main-card">
      <div class="card-header">
        <div class="card-title">🏦 Interactive Applicant Risk Evaluator</div>
        <div class="sample-buttons">
          <button type="button" class="btn-sample" onclick="loadSample('good')">Load Safe Applicant</button>
          <button type="button" class="btn-sample" onclick="loadSample('bad')">Load High-Risk Applicant</button>
        </div>
      </div>

      <form id="riskForm" onsubmit="handlePredict(event)">
        <div class="form-grid">
          <div class="input-group">
            <label>Checking Status</label>
            <input type="text" id="checking_status" value="<0" required>
          </div>
          <div class="input-group">
            <label>Duration (Months)</label>
            <input type="number" id="duration" value="6" required>
          </div>
          <div class="input-group">
            <label>Credit History</label>
            <input type="text" id="credit_history" value="critical/other existing credit" required>
          </div>
          <div class="input-group">
            <label>Purpose</label>
            <input type="text" id="purpose" value="radio/tv" required>
          </div>
          <div class="input-group">
            <label>Credit Amount (DM)</label>
            <input type="number" id="credit_amount" value="1169" required>
          </div>
          <div class="input-group">
            <label>Savings Status</label>
            <input type="text" id="savings_status" value="no known savings" required>
          </div>
          <div class="input-group">
            <label>Employment Duration</label>
            <input type="text" id="employment" value=">=7" required>
          </div>
          <div class="input-group">
            <label>Installment Rate (% income)</label>
            <input type="number" id="installment_commitment" value="4" required>
          </div>
          <div class="input-group">
            <label>Personal Status / Gender</label>
            <input type="text" id="personal_status" value="male single" required>
          </div>
          <div class="input-group">
            <label>Other Debtors / Parties</label>
            <input type="text" id="other_parties" value="none" required>
          </div>
          <div class="input-group">
            <label>Present Residence Since (Yrs)</label>
            <input type="number" id="residence_since" value="4" required>
          </div>
          <div class="input-group">
            <label>Property Magnitude</label>
            <input type="text" id="property_magnitude" value="real estate" required>
          </div>
          <div class="input-group">
            <label>Applicant Age</label>
            <input type="number" id="age" value="67" required>
          </div>
          <div class="input-group">
            <label>Other Payment Plans</label>
            <input type="text" id="other_payment_plans" value="none" required>
          </div>
          <div class="input-group">
            <label>Housing</label>
            <input type="text" id="housing" value="own" required>
          </div>
          <div class="input-group">
            <label>Existing Credits Count</label>
            <input type="number" id="existing_credits" value="2" required>
          </div>
          <div class="input-group">
            <label>Job Category</label>
            <input type="text" id="job" value="skilled" required>
          </div>
          <div class="input-group">
            <label>Number of Dependents</label>
            <input type="number" id="num_dependents" value="1" required>
          </div>
          <div class="input-group">
            <label>Telephone Registered</label>
            <input type="text" id="own_telephone" value="yes" required>
          </div>
          <div class="input-group">
            <label>Foreign Worker</label>
            <input type="text" id="foreign_worker" value="yes" required>
          </div>
        </div>

        <button type="submit" class="btn-predict" id="submitBtn">
          <span>⚡ Evaluate Credit Risk</span>
        </button>
      </form>

      <div class="result-box" id="resultBox">
        <div class="result-header">
          <div>
            <div style="font-size: 0.8rem; color: var(--text-muted);">Model Decision</div>
            <div id="resultBadge" class="result-badge">Evaluating...</div>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 0.8rem; color: var(--text-muted);">Confidence Score</div>
            <div id="confidenceVal" style="font-size: 1.2rem; font-weight: 700;">--</div>
          </div>
        </div>
        <div class="confidence-meter">
          <div id="confidenceFill" class="confidence-fill"></div>
        </div>
        <div style="margin-top: 1rem; font-size: 0.8rem; color: var(--text-muted);">Live JSON Response:</div>
        <pre class="api-code" id="rawJson"></pre>
      </div>
    </div>

    <div class="main-card">
      <div class="card-title" style="margin-bottom: 0.5rem;">🔌 Developer REST API Usage</div>
      <p style="color: var(--text-muted); font-size: 0.85rem;">Send an HTTP POST request with applicant parameters in JSON format to <code>/predict</code>.</p>
      <pre class="api-code">curl -X POST https://ml-t2-090-model-complexity.onrender.com/predict \\
     -H "Content-Type: application/json" \\
     -d '{
       "checking_status": "<0", "duration": 6, "credit_history": "critical/other existing credit",
       "purpose": "radio/tv", "credit_amount": 1169, "savings_status": "no known savings",
       "employment": ">=7", "installment_commitment": 4, "personal_status": "male single",
       "other_parties": "none", "residence_since": 4, "property_magnitude": "real estate",
       "age": 67, "other_payment_plans": "none", "housing": "own", "existing_credits": 2,
       "job": "skilled", "num_dependents": 1, "own_telephone": "yes", "foreign_worker": "yes"
     }'</pre>
    </div>

    <footer>
      Project ID: <strong>ML-T2-090</strong> | Machine Learning Capstone | Author: <strong>Gourav Das</strong><br>
      Repository: <a href="https://github.com/grvd5678/ML-T2-090-Model-Complexity" target="_blank">github.com/grvd5678/ML-T2-090-Model-Complexity</a>
    </footer>
  </div>

  <script>
    const samples = {
      good: {
        checking_status: "<0", duration: 6, credit_history: "critical/other existing credit",
        purpose: "radio/tv", credit_amount: 1169, savings_status: "no known savings",
        employment: ">=7", installment_commitment: 4, personal_status: "male single",
        other_parties: "none", residence_since: 4, property_magnitude: "real estate",
        age: 67, other_payment_plans: "none", housing: "own", existing_credits: 2,
        job: "skilled", num_dependents: 1, own_telephone: "yes", foreign_worker: "yes"
      },
      bad: {
        checking_status: "0<=X<200", duration: 48, credit_history: "existing paid",
        purpose: "radio/tv", credit_amount: 5951, savings_status: "<100",
        employment: "1<=X<4", installment_commitment: 2, personal_status: "female div/dep/mar",
        other_parties: "none", residence_since: 2, property_magnitude: "real estate",
        age: 22, other_payment_plans: "none", housing: "own", existing_credits: 1,
        job: "skilled", num_dependents: 1, own_telephone: "none", foreign_worker: "yes"
      }
    };

    function loadSample(type) {
      const data = samples[type];
      for (const [key, val] of Object.entries(data)) {
        const el = document.getElementById(key);
        if (el) el.value = val;
      }
    }

    async function handlePredict(e) {
      e.preventDefault();
      const btn = document.getElementById('submitBtn');
      const box = document.getElementById('resultBox');
      const badge = document.getElementById('resultBadge');
      const confVal = document.getElementById('confidenceVal');
      const confFill = document.getElementById('confidenceFill');
      const rawJson = document.getElementById('rawJson');

      btn.disabled = true;
      btn.innerText = "⏳ Evaluating Applicant...";

      const payload = {};
      const inputs = document.querySelectorAll('#riskForm input');
      inputs.forEach(inp => {
        payload[inp.id] = (inp.type === 'number') ? Number(inp.value) : inp.value;
      });

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        box.style.display = 'block';
        rawJson.innerText = JSON.stringify(data, null, 2);

        if (data.status === 'success') {
          const isGood = data.prediction === 'Good Credit Risk';
          badge.className = 'result-badge ' + (isGood ? 'badge-good' : 'badge-bad');
          badge.innerText = (isGood ? '✅ ' : '⚠️ ') + data.prediction;

          const pct = Math.round((data.confidence_score || 0) * 100);
          confVal.innerText = pct + '%';
          confFill.style.width = pct + '%';
          confFill.style.background = isGood ? 'var(--success)' : 'var(--danger)';
        } else {
          badge.className = 'result-badge badge-bad';
          badge.innerText = 'Error: ' + data.message;
        }
      } catch (err) {
        box.style.display = 'block';
        badge.className = 'result-badge badge-bad';
        badge.innerText = 'Network error or cold start: ' + err.message;
      } finally {
        btn.disabled = false;
        btn.innerText = "⚡ Evaluate Credit Risk";
      }
    }
  </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(LANDING_PAGE_HTML)

@app.route('/predict', methods=['GET', 'POST'])
def predict_risk():
    if request.method == 'GET':
        return jsonify({
            "status": "info",
            "message": "The /predict endpoint accepts HTTP POST requests with an applicant JSON payload. Visit the root URL / in your browser for the interactive evaluator.",
            "sample_endpoint": "/predict",
            "method": "POST"
        })

    try:
        applicant_data = request.get_json()
        if not applicant_data:
            return jsonify({"status": "error", "message": "No JSON payload provided in request body."}), 400

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
            "confidence_score": round(float(confidence), 4)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # Running strictly locally for now
    app.run(port=5000, debug=True)