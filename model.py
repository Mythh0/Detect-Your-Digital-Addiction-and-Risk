from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_model():
    samples = [
        ([1.5, 5,  10,  0.0, 20,  0.5, 15], 0),
        ([2.0, 8,  20,  0.0, 30,  0.8, 20], 0),
        ([1.0, 4,   8,  0.0, 15,  0.3, 10], 0),
        ([2.5, 10, 25,  0.1, 35,  1.0, 25], 0),
        ([3.0, 12, 30,  0.0, 40,  1.2, 30], 0),
        ([4.0, 25, 60,  0.3, 80,  2.0, 45], 1),
        ([5.0, 35, 80,  0.5, 100, 2.5, 55], 1),
        ([4.5, 30, 70,  0.4, 90,  2.2, 50], 1),
        ([5.5, 40, 90,  0.6, 110, 3.0, 60], 1),
        ([6.0, 45, 100, 0.7, 120, 3.5, 65], 1),
        ([7.0, 60, 120, 1.0, 180, 4.5, 90],  2),
        ([8.0, 80, 150, 1.5, 220, 5.5, 110], 2),
        ([9.0, 100,180, 2.0, 260, 6.5, 130], 2),
        ([7.5, 70, 140, 1.2, 200, 5.0, 100], 2),
        ([10.0,120,200, 2.5, 300, 7.5, 150], 2),
    ]
    X = np.array([s[0] for s in samples])
    y = np.array([s[1] for s in samples])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y)
    print(f"✅ Model ready! Accuracy: {clf.score(X_scaled,y)*100:.1f}%")
    return clf, scaler

def predict(clf, scaler, features):
    labels = {0:"Low Addiction", 1:"Moderate Addiction", 2:"High Addiction"}
    x = np.array(features).reshape(1,-1)
    x_scaled = scaler.transform(x)
    pred = clf.predict(x_scaled)[0]
    proba = clf.predict_proba(x_scaled)[0]
    risk = round(float(proba[1]*40 + proba[2]*100 + pred*15), 1)
    risk = min(100, max(0, risk))
    return {
        "label": labels[pred],
        "risk_score": risk,
        "confidence": round(float(max(proba))*100, 1),
        "class": int(pred)
    }
