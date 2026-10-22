"""Evaluate a locally trained PromptGuard model against a separate synthetic challenge set.

This is a small educational test, not a real-world security benchmark.
Run: .\\.venv\\Scripts\\python.exe evaluate_final.py
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent
TEST_PATH = BASE / "data" / "final_test.csv"
TRAIN_PATH = BASE / "data" / "prompts.csv"
MODEL_PATH = BASE / "models" / "promptguard.joblib"

if not MODEL_PATH.exists():
    raise SystemExit("Model missing. First run: .\\.venv\\Scripts\\python.exe train.py")
if not TEST_PATH.exists():
    raise SystemExit("Cannot find data/final_test.csv. Put the downloaded test file in data/.")

tests = pd.read_csv(TEST_PATH)
if not {"text", "label"}.issubset(tests.columns):
    raise SystemExit("final_test.csv must have columns 'text' and 'label'.")
if tests["text"].isna().any() or not set(tests["label"].unique()).issubset({0, 1}):
    raise SystemExit("The final test file has missing text or invalid labels.")

# Verify that none of the challenge samples are exact duplicates of training rows.
if TRAIN_PATH.exists():
    train = pd.read_csv(TRAIN_PATH)
    training_texts = set(train["text"].astype(str).str.strip().str.casefold())
    test_texts = set(tests["text"].astype(str).str.strip().str.casefold())
    overlapping = test_texts & training_texts
    if overlapping:
        raise SystemExit(f"Found {len(overlapping)} test sentences also in prompts.csv. Stop: test leakage.")

model = joblib.load(MODEL_PATH)  # Load only a model you trained yourself.
predictions = model.predict(tests["text"].tolist())
label1_index = list(model.classes_).index(1)
scores = model.predict_proba(tests["text"].tolist())[:, label1_index]

results = tests.copy()
results["prediction"] = predictions.astype(int)
results["suspicious_score"] = scores.round(4)
results["correct"] = results["label"] == results["prediction"]

accuracy = accuracy_score(results["label"], results["prediction"])
matrix = confusion_matrix(results["label"], results["prediction"], labels=[0, 1])
report = classification_report(
    results["label"], results["prediction"], labels=[0, 1],
    target_names=["Ordinary", "Suspicious"], zero_division=0,
)

print("\nPROMPTGUARD — SEPARATE SYNTHETIC CHALLENGE SET")
print("=" * 52)
print(f"Test samples: {len(results)}")
print(f"Correct: {int(results['correct'].sum())} / {len(results)}")
print(f"Accuracy: {accuracy:.1%}")
print("\n" + report)
print("Confusion matrix [actual 0/1 rows, predicted 0/1 columns]:")
print(matrix)
print("\nIncorrect examples:")
wrong = results.loc[~results["correct"], ["text", "label", "prediction", "suspicious_score"]]
if wrong.empty:
    print("None in this small set.")
else:
    for _, row in wrong.iterrows():
        print(f"- Expected {row['label']}, predicted {row['prediction']} (score {row['suspicious_score']:.1%}): {row['text']}")

reports = BASE / "reports"
reports.mkdir(exist_ok=True)
results.to_csv(reports / "final_evaluation.csv", index=False, encoding="utf-8")
summary = (
    "PromptGuard — synthetic challenge set (not a real-world security benchmark)\n"
    f"Test samples: {len(results)}\n"
    f"Correct: {int(results['correct'].sum())} / {len(results)}\n"
    f"Accuracy: {accuracy:.1%}\n\n"
    + report + "\nConfusion matrix [actual rows 0/1, predicted columns 0/1]:\n"
    + str(matrix) + "\n"
)
(reports / "final_summary.txt").write_text(summary, encoding="utf-8")
print("\nSaved: reports/final_evaluation.csv and reports/final_summary.txt")
print("WARNING: This is only a small hand-written synthetic challenge set.")
print("Do not treat this result as real-world prompt-injection detection accuracy.")
