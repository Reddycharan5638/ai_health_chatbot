import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

out_dir = Path(__file__).resolve().parent.parent / "models"
out_dir.mkdir(parents=True, exist_ok=True)

def make_synthetic_diabetes(n=5000, seed=42):
    rng = np.random.RandomState(seed)
    age = rng.randint(20, 80, size=n)
    bmi = rng.normal(28, 6, size=n)
    glucose = rng.normal(120, 30, size=n)
    bp = rng.normal(80, 12, size=n)
    insulin = rng.normal(80, 40, size=n)
    # create a continuous risk score and add noise so classes vary
    risk_score = (bmi/30) + (glucose/140) + (age/60) + (bp/120) + rng.normal(0, 0.5, size=n)
    prob = 1/(1+np.exp(-2*(risk_score-1.6)))
    y = (prob > rng.uniform(0,1,size=n)*0.6).astype(int)  # randomized threshold
    # ensure both classes exist
    if y.mean() in (0.0, 1.0):
        # flip some labels to ensure both classes
        k = max(1, int(0.02*n))
        y[:k] = 1 - y[:k]
    X = pd.DataFrame({'age': age, 'bmi': bmi, 'glucose': glucose, 'bp': bp, 'insulin': insulin})
    return X, y

def make_synthetic_heart(n=5000, seed=24):
    rng = np.random.RandomState(seed)
    age = rng.randint(30, 85, size=n)
    chol = rng.normal(220, 40, size=n)
    bp = rng.normal(135, 18, size=n)
    smoker = rng.binomial(1, 0.25, size=n)
    diabetes = rng.binomial(1, 0.12, size=n)
    risk_score = (age/60) + (chol/250) + (bp/160) + smoker*0.6 + diabetes*0.8 + rng.normal(0,0.6,size=n)
    prob = 1/(1+np.exp(-2*(risk_score-1.7)))
    y = (prob > rng.uniform(0,1,size=n)*0.7).astype(int)
    if y.mean() in (0.0, 1.0):
        k = max(1, int(0.02*n))
        y[:k] = 1 - y[:k]
    X = pd.DataFrame({'age': age, 'cholesterol': chol, 'bp': bp, 'smoker': smoker, 'diabetes': diabetes})
    return X, y

def train_and_save(X, y, filename):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(Xs, y)
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': list(X.columns)}, f)
    print("Saved", filename)

if __name__ == "__main__":
    Xd, yd = make_synthetic_diabetes()
    train_and_save(Xd, yd, out_dir / "diabetes_model.pkl")
    Xh, yh = make_synthetic_heart()
    train_and_save(Xh, yh, out_dir / "heart_model.pkl")
    print("Training complete.")
