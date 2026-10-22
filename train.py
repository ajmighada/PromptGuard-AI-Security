from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "prompts.csv")
assert set(df.columns) == {"text", "label"}
assert set(df["label"].unique()) == {0, 1}

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25,
    random_state=42, stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
print("Evaluation on a held-out 25% of synthetic examples:")
print(classification_report(y_test, predictions, target_names=["Ordinary", "Suspicious"], zero_division=0))
print("Confusion matrix [actual rows 0/1, predicted columns 0/1]:")
print(matrix)
print("NOTE: Metrics on small synthetic data do not establish real-world reliability.")

models = BASE / "models"
models.mkdir(exist_ok=True)
joblib.dump(model, models / "promptguard.joblib")
with (models / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump({
        "sample_count": int(len(df)),
        "test_count": int(len(X_test)),
        "accuracy": round(float(report["accuracy"]), 3),
        "suspicious_precision": round(float(report["1"]["precision"]), 3),
        "suspicious_recall": round(float(report["1"]["recall"]), 3),
        "suspicious_f1": round(float(report["1"]["f1-score"]), 3),
        "confusion_matrix": matrix.tolist(),
    }, f, indent=2)
print("Model saved to models/promptguard.joblib")
