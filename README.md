# PromptGuard — AI instruction-attack detector (student prototype)

A small machine-learning experiment that flags potentially suspicious instructions embedded in
external text an AI assistant might read (emails, webpages, or retrieved documents).
**It is not a production prompt-injection defense** and does not prevent any actions.

## Stack
Python, pandas, scikit-learn (TF-IDF + logistic regression), Streamlit, joblib.

## Run on Windows (PowerShell, from this folder)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

A browser page should open. On subsequent runs, you can skip creating/installing the environment.
Press Ctrl+C in the terminal to stop the app.

## What is evaluated?
`train.py` splits the **synthetic** examples in `data/prompts.csv` 75%/25%, trains a
TF-IDF + logistic-regression pipeline, prints precision/recall/F1 on the hold-out split,
and writes results to `models/metrics.json`. The examples were created for a tutorial,
not gathered from actual security incidents. Dataset is small and not representative.
Some examples are instruction-override/jailbreak text, which may not be true prompt injection
without a lower-trust context. Accuracy on this dataset is **not a real-world detection rate**.

## Limitations
- The classifier sees text only, not the trust boundary or the user's actual instructions.
- It can overfit words from this small synthetic dataset.
- Harmless educational quotations can be false positives; novel attacks can be missed.
- `predict_proba` is not a calibrated security probability.
- This demo never executes actions or sends data anywhere.
- Load joblib files only if you created/trust them.

## Ideas for improvement
1. Add 30+ realistic benign snippets and 30+ suspicious snippets from different styles.
2. Keep a separate, manually written test set rather than using only random split.
3. Document false positives and false negatives and compare against a simple regex baseline.
4. Show a screenshot in this README and describe one thing you learned.

## CV (only after building, testing, and understanding it)
"PromptGuard – ML-Prototyp zur Erkennung verdächtiger Anweisungen | Python, scikit-learn, Streamlit"
## Model Evaluation

PromptGuard was evaluated on a separate synthetic challenge set containing 40 examples: 20 ordinary inputs and 20 potentially suspicious instructions.

### Evaluation Results

| Metric                     | Result |
| -------------------------- | ------ |
| Accuracy                   | 90.0%  |
| Correct predictions        | 36/40  |
| Suspicious-class precision | 83%    |
| Suspicious-class recall    | 100%   |
| Suspicious-class F1-score  | 0.91   |
| False positives            | 4      |
| False negatives            | 0      |

### Confusion Matrix

|                     | Predicted Normal | Predicted Suspicious |
| ------------------- | ---------------- | -------------------- |
| Actually Normal     | 16               | 4                    |
| Actually Suspicious | 0                | 20                   |

The model correctly identified all 20 suspicious examples in this evaluation but incorrectly flagged four harmless sentences.

The results demonstrate the limitations of keyword-based statistical learning, particularly when processing educational content or text discussing security instructions.

These results are based on a small synthetic dataset and do not represent real-world detection performance.

Detailed evaluation results are available in the `reports/` directory.

## Application Screenshots

### 1. Suspicious Instruction Detection

![Suspicious Input Detection](screenshots/01-suspicious.png)

### 2. Explainable AI Predictions

![Feature-Level Explanation](screenshots/02-explanation.png)

### 3. Normal Input Classification

![Normal Input Detection](screenshots/03-normal.png)

## Explainable AI

PromptGuard provides feature-level explanations for its predictions.

Using TF-IDF feature values and Logistic Regression coefficients, the application identifies words and phrases that contribute positively or negatively to the classification.

This allows users to investigate the model's behavior and identify potentially misleading statistical associations.

These explanations describe model behavior rather than the actual intentions behind the input.

## Development and Learning

This project was developed using a Python starter template and subsequently extended through hands-on experimentation.

The development process included:

* Expanding the training dataset with additional examples.
* Investigating false positives and classification errors.
* Implementing feature-level explanations.
* Creating additional evaluation scripts.
* Evaluating the model against separate synthetic test examples.
* Analyzing the limitations of TF-IDF and Logistic Regression for security-related text classification.

## Future Improvements

* Expand the dataset with more diverse and realistic examples.
* Compare Logistic Regression with other classification models.
* Evaluate performance on larger, independently sourced datasets.
* Improve the distinction between malicious instructions and harmless quotations.
* Investigate context-aware approaches to prompt-injection detection.

