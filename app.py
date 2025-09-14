from flask import Flask, render_template, request, jsonify
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

app = Flask(__name__)
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

models = {}
for name in ['diabetes_model.pkl', 'heart_model.pkl']:
    p = MODEL_DIR / name
    if p.exists():
        models[name.split('_')[0]] = load_model(p)
    else:
        models[name.split('_')[0]] = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    model_name = data.get('model')
    inputs = data.get('inputs', {})
    if model_name not in models or models[model_name] is None:
        return jsonify({'error': f'Model {model_name} not available.'}), 400
    mdl = models[model_name]
    features = mdl['features']
    row = [float(inputs.get(f, 0)) for f in features]
    X = np.array(row).reshape(1, -1)
    Xs = mdl['scaler'].transform(X)
    prob = mdl['model'].predict_proba(Xs)[0][1]
    risk = float(prob)
    # Simple textual advice rules
    advice = []
    if risk > 0.7:
        advice.append("High risk: consult a healthcare professional as soon as possible.")
    elif risk > 0.4:
        advice.append("Moderate risk: consider lifestyle changes and schedule a check-up.")
    else:
        advice.append("Low risk: maintain healthy lifestyle and regular checkups.")
    return jsonify({'risk': risk, 'advice': advice})

@app.route("/chat", methods=["POST"])
def chat():
    # Minimal rule-based chatbot to collect user inputs step-by-step.
    payload = request.json or {}
    history = payload.get('history', [])
    last = history[-1] if history else {"sender":"bot","text":"Hello! Do you want a Diabetes or Heart Disease check today? Reply 'diabetes' or 'heart'."}
    # Very simple flow state stored in history messages
    # Determine if user selected model
    texts = [m.get('text','').lower() for m in history if m.get('sender')=='user']
    model = None
    if any('diabet' in t for t in texts):
        model = 'diabetes'
    elif any('heart' in t for t in texts):
        model = 'heart'
    # If model selected, ask for required features
    if model and model in models and models[model] is not None:
        features = models[model]['features']
        # check which features we have collected
        collected = {}
        for t in texts:
            # expects "feature: value" or "age 45" style. Very forgiving parser.
            parts = t.replace(':',' ').split()
            for i, w in enumerate(parts):
                for f in features:
                    if w.startswith(f[:3]):  # match by prefix
                        # try to get next token as value
                        if i+1 < len(parts):
                            try:
                                val = float(parts[i+1])
                                collected[f] = val
                            except:
                                pass
        missing = [f for f in features if f not in collected]
        if not missing:
            # run prediction
            resp = {
                'sender': 'bot',
                'text': f"I have all inputs. Running {model} model now..."
            }
            return jsonify({'reply': resp, 'action': 'predict', 'model': model, 'inputs': collected})
        else:
            # ask for the next missing feature
            nextq = missing[0]
            resp = {
    'sender': 'bot',
    'text': f"Please provide your {nextq} (numeric). Example: {nextq} 70"
}

            return jsonify({'reply': resp})
    else:
        # ask the user to choose model
        resp = {'sender':'bot', 'text': "Please type 'diabetes' or 'heart' to start a check."}
        return jsonify({'reply': resp})

if __name__ == "__main__":
    app.run(debug=True)
