import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

with open('model/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tenure = int(request.form.get('tenure', 12))
    monthly = float(request.form.get('monthly', 65.0))
    contract = int(request.form.get('contract', 0))
    tech_support = int(request.form.get('tech_support', 0))
    fiber = int(request.form.get('fiber', 0))
    
    features = np.array([[tenure, monthly, contract, tech_support, fiber]])
    prob = model.predict_proba(features)[0][1] * 100
    is_churn = prob > 50
    
    tier = "High Risk" if prob > 65 else ("Medium Risk" if prob > 35 else "Low Risk / Retained")
    color = "#ef4444" if prob > 65 else ("#f59e0b" if prob > 35 else "#10b981")
    
    reasons = []
    if contract == 0:
        reasons.append("Month-to-month rolling contract (no multi-year commitment lock)")
    if monthly > 80:
        reasons.append("High monthly invoice charges above portfolio benchmark")
    if tech_support == 0:
        reasons.append("Zero dedicated technical support add-on subscribed")
    if not reasons:
        reasons.append("Strong loyalty indicator: Multi-year tenure and active support")
        
    return render_template('index.html',
                           predicted=True,
                           prob=round(prob, 1),
                           tier=tier,
                           color=color,
                           reasons=reasons,
                           tenure=tenure,
                           monthly=monthly)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
