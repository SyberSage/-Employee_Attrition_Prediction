# Employee Attrition Prediction

A data science project predicting employee attrition using the IBM HR Analytics dataset,
built with pandas and scikit-learn, and deployed as an interactive Streamlit app.

**Live app:** https://employee-attrition-prediction-sybersage.streamlit.app/

## Project Overview

This project explores two prediction targets from the dataset:

1. **Attrition** (whether an employee leaves the company) — successfully predicted
2. **Performance Rating** — investigated, but found not to be predictable from the
   available features (see Findings below)

## Approach

- **EDA** — explored target balance, and key relationships (e.g. OverTime vs. Attrition,
  Income vs. Attrition)
- **Preprocessing** — dropped non-informative columns, encoded categorical features with
  One-Hot Encoding, scaled numeric features with StandardScaler
- **Feature Engineering** — created 7 new features: `TenureRatio`, `IncomePerYearWorked`,
  `OverallSatisfaction`, `TenureCategory`, `PerformanceScore`, `EngagementScore`, `AgeGroup`
- **Modeling** — trained and compared 4 classifiers: Logistic Regression, Decision Tree,
  Random Forest, and K-Nearest Neighbors
- **Evaluation** — Accuracy, Precision, Recall, F1, AUC-ROC, and confusion matrices for
  every model
- **Deployment** — final model saved with `joblib` and served through a Streamlit app

## Results — Attrition Prediction

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.776 | 0.391 | **0.723** | **0.507** | **0.817** |
| Decision Tree | 0.786 | 0.333 | 0.340 | 0.337 | 0.605 |
| Random Forest | 0.847 | 0.667 | 0.085 | 0.151 | 0.757 |
| K-Nearest Neighbors | 0.840 | 0.500 | 0.085 | 0.145 | 0.579 |

**Logistic Regression** was selected as the final model. Despite not having the highest
raw accuracy, it achieved by far the best Recall and AUC — meaning it catches the most
employees who are actually at risk of leaving, which matters more than overall accuracy
for this business problem, given the ~16% class imbalance in the dataset.

## Findings — Performance Rating Prediction

An initial pass produced suspiciously perfect scores (AUC = 1.0) for two of the four
models. Investigation revealed **data leakage**: `PercentSalaryHike` has non-overlapping
ranges across the two rating classes (11–19% for Rating 3, 20–25% for Rating 4), and
correlates ~0.77 with the target — effectively encoding the answer.

After removing `PercentSalaryHike` and the derived `PerformanceScore` feature, all four
models scored an AUC between **0.44 and 0.53** — no better than random guessing. A
correlation check across all remaining numeric features confirmed this, with the
strongest relationship found being just **-0.044**.

**Conclusion:** Performance Rating, as recorded in this dataset, is not meaningfully
explained by any of the available employee attributes once the leaked salary-hike signal
is removed. This is reported as a finding rather than forced into an unreliable model.

## Project Structure

```
├── 1_EDA_corrected.ipynb     # Full notebook: EDA -> preprocessing -> feature
│                              #   engineering -> modeling -> evaluation (both targets)
├── attrition_app.py           # Streamlit app serving the Attrition model
├── attrition_model.pkl        # Saved Logistic Regression pipeline (preprocessing + model)
├── requirements.txt
└── README.md
```

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the notebook (`1_EDA_corrected.ipynb`) top to bottom to reproduce the analysis
   and regenerate `attrition_model.pkl`, or use the one already included.
3. Launch the app:
   ```
   streamlit run attrition_app.py
   ```
4. Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Dataset

IBM HR Analytics Employee Attrition & Performance dataset (1,470 rows, 35 columns).
