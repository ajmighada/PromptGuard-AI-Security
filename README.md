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
