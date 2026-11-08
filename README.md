# 👥 Employee Attrition Analysis and Prediction

A machine learning project that **predicts which employees are at risk of leaving**. It combines a full scikit-learn classification pipeline with a Streamlit app that scores an employee's attrition risk from live inputs.

The project is built with **Python, scikit-learn and Streamlit**.

## 🚀 Live Demo

**Streamlit App:** `<add your deployed link here>`

## 📌 Project Overview

Employee turnover increases recruitment and training costs, reduces productivity and disrupts teams. This project analyzes the **IBM HR Analytics Employee Attrition dataset** (1,470 employees, 35 columns) to find the factors associated with attrition and to build models that flag at-risk employees early, so retention action can be taken before they leave.

**Prediction targets**

* **Attrition** (primary): will the employee leave? (Yes / No)
* **Performance Rating** (secondary): explored, but the data does not contain enough signal to predict it reliably, so it is not included in the app (see [Performance Rating](#-performance-rating-a-negative-result))

The project covers:

* 🧹 Data preprocessing and cleaning
* 📊 Exploratory data analysis (EDA)
* 🛠️ Feature engineering (tenure categories, performance metrics, engagement scores)
* 🤖 Model building and comparison (4 algorithms)
* 📈 Model evaluation (Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix, training time)
* 🖥️ Streamlit app for live attrition risk scoring

## 💼 Business Use Cases

* **Employee Retention:** identify at-risk employees and target retention strategies.
* **Cost Optimization:** reduce recruitment, onboarding and training costs.
* **Workforce Planning:** align retention strategies with organizational goals and employee satisfaction.

## ✨ Features

* Live attrition risk scoring from 29 employee inputs, shown as a risk score with a Low / Medium / High level
* Input validation that blocks impossible combinations (for example, more years in a role than at the company)
* Model, preprocessing and feature engineering saved together as a single pipeline, so the app only needs raw employee fields
* Imbalance-aware evaluation: recall, F1 and AUC are reported alongside accuracy, not hidden behind it

## 📂 Dataset

| Property | Detail |
| --- | --- |
| Source | IBM HR Analytics Employee Attrition dataset |
| File | `Employee-Attrition.xlsx` |
| Size | 1,470 rows × 35 columns |
| Missing values | None |
| Class balance | 1,233 stayed (≈84%), 237 left (≈16%) |

**Data notes**

* `EmployeeCount`, `Over18` and `StandardHours` are constant, so they carry no information and were dropped.
* `EmployeeNumber` is only an identifier and was dropped before modeling.
* `PerformanceRating` contains only the values 3 and 4 (1,244 and 226 employees) rather than the documented 1–4 range, and was encoded as 0 / 1.
* `PercentSalaryHike` correlates strongly with `PerformanceRating` (≈0.77), which is a **data leakage risk** for the Performance Rating model. It, and the engineered `PerformanceScore` that is partly built from it, are excluded from that model.

## 🛠️ Feature Engineering

New features were created in the three categories required by the project brief, plus extra derived features:

| Category | Feature | How it is built |
| --- | --- | --- |
| Tenure categories | `TenureCategory` | Bands of `YearsAtCompany`: New (0–2y), Early (3–5y), Established (6–10y), Veteran (11y+) |
| Performance metrics | `PerformanceScore` | Average of min-max scaled `PercentSalaryHike` and `TrainingTimesLastYear` |
| Engagement scores | `EngagementScore` | Average of min-max scaled `JobInvolvement` and `StockOptionLevel` |
| Additional | `OverallSatisfaction` | Mean of Job, Environment and Relationship satisfaction and Work-Life Balance |
| Additional | `TenureRatio` | `YearsInCurrentRole` / `YearsAtCompany` |
| Additional | `IncomePerYearWorked` | `MonthlyIncome` / (`TotalWorkingYears` + 1) |
| Additional | `AgeGroup` | Bands of `Age`: 18–25, 26–35, 36–45, 46–60 |

**Avoiding leakage in the scaling:** the data is split into train and test sets *before* feature engineering. The min/max values used for scaling are learned from the training rows only and then applied to both sets.

## 🧠 Machine Learning Models

Four scikit-learn models were trained and compared on a stratified 80/20 split (`random_state=42`):

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors

Logistic Regression, Decision Tree and Random Forest use `class_weight='balanced'` to handle the ~16% attrition minority class. K-Nearest Neighbors does not support class weights and was run without them.

Each model is a single `Pipeline`: **feature engineering → one-hot encoding and scaling → classifier**. Feeding raw columns in and getting predictions out is what lets the Streamlit app stay simple.

### Why not accuracy alone?

Because only ~16% of employees left, a model can score high accuracy while missing most leavers. Random Forest reached **84.7% accuracy but only 8.5% recall** on attrition, and K-Nearest Neighbors reached 84.0% accuracy with 8.5% recall, so both caught very few of the employees who actually left. For this reason, model selection focuses on **AUC-ROC, Recall and F1** in addition to accuracy.

### Results

Test-set results (294 employees, 47 of whom left):

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Train time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.776 | 0.391 | **0.723** | **0.507** | **0.817** | 0.144 |
| Decision Tree | 0.786 | 0.333 | 0.340 | 0.337 | 0.605 | 0.097 |
| Random Forest | 0.847 | 0.667 | 0.085 | 0.151 | 0.757 | 0.564 |
| K-Nearest Neighbors | 0.840 | 0.500 | 0.085 | 0.145 | 0.579 | 0.125 |

**Selected model:** Logistic Regression, chosen because it has the highest AUC-ROC (0.817), meaning it ranks likely leavers above stayers better than the other models, and the best Recall (72.3%), meaning it catches the most employees who actually left. It does not have the highest raw accuracy, but accuracy is misleading on this imbalanced data.

**Limitations (honest notes)**

* The best AUC-ROC (0.817) is below the example target of about 85% in the project brief. No hyperparameter tuning or cross-validation has been done yet.
* The test set contains only 47 leavers, so these numbers will move on a different split.
* Training times depend on the machine they were run on.

## 🔄 Machine Learning Workflow

```text
Raw Employee Dataset
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis (EDA)
   ↓
Stratified Train/Test Split (before feature engineering)
   ↓
Feature Engineering (scaling bounds learned from training data only)
   ↓
Model Training (Logistic Regression, Decision Tree, Random Forest, KNN)
   ↓
Model Evaluation & Comparison
   ↓
Best Model Selection (highest AUC-ROC)
   ↓
Save pipeline (attrition_model.joblib)
   ↓
Streamlit App
```

## 🔑 Key Drivers of Attrition

Attrition rates in the dataset, which the Logistic Regression coefficients agree with:

* **Overtime:** 30.5% of employees working overtime left, compared with 10.4% of those who did not.
* **Business travel:** 24.9% of frequent travellers left, compared with 8.0% of non-travellers.
* **Job role:** Sales Representatives (39.8%) and Laboratory Technicians (23.9%) have the highest attrition. Research Directors have the lowest-risk model coefficient.
* **Career stage:** employees at Job Level 1 left at 26.3% versus 10.1% for all other levels, and single employees left at 25.5% versus 12.5% for married and 10.1% for divorced employees.
* **Stock options:** employees with no stock options left at 24.4%, compared with 9.4% at level 1.

> These are **associations, not proven causes**. Several features overlap (for example `Age` and `AgeGroup`, or `YearsAtCompany` and `TenureCategory`), so the model splits credit between them.

## 🎯 Retention Recommendations

Suggested actions based on the findings above (to be validated with the business before use):

| Finding | Suggested action |
| --- | --- |
| Overtime is associated with 3× higher attrition | Review workload and overtime in high-attrition roles; monitor overtime hours as an early warning |
| Frequent travel is associated with higher attrition | Review travel expectations for frequent travellers; offer flexibility or rotation |
| Sales Representatives and Laboratory Technicians leave most | Run stay interviews in these roles; review pay, career path and manager support |
| Entry-level and single employees leave more | Strengthen onboarding, mentoring and visible progression for early-career staff |
| Employees without stock options leave more | Review whether equity or other long-term incentives can be extended |

## 📉 Performance Rating: a negative result

Predicting **Performance Rating** was explored as a second target. After removing the leaking feature (`PercentSalaryHike`) and the engineered `PerformanceScore`, all four models scored an AUC-ROC between **0.44 and 0.53**, which is roughly the same as random guessing. A correlation check across the remaining numeric features agreed: none exceeded ±0.053 with Performance Rating.

The dataset does not contain enough genuine signal to predict Performance Rating, so no such prediction is offered in the app rather than presenting an unreliable one.

## 🧭 Roadmap

Not yet implemented:

* Dashboard pages for attrition trends (department, role, overtime, income, tenure, satisfaction)
* A key-drivers chart built from the model coefficients
* A ranked list of employees by risk score
* Hyperparameter tuning and cross-validation to improve the model
* Per-employee explanations of what raised or lowered a risk score

## 🛠️ Technologies Used

**Programming:** Python

**Data Science:** Pandas, NumPy, Scikit-learn

**Visualization:** Matplotlib, Seaborn

**App:** Streamlit

**Model saving:** Joblib, Cloudpickle

**Development Tools:** Jupyter Notebook, VS Code, GitHub

## 📂 Project Structure

> Adjust the names to match your repository exactly.

```text
Employee-Attrition-Analysis-and-Prediction/
│
├── pipeline.ipynb              # EDA, feature engineering, model building, evaluation, model saving
├── attrition_app.py            # Streamlit app
├── attrition_model.joblib      # Saved pipeline (REQUIRED for the app to run)
├── Employee-Attrition.xlsx     # Dataset
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### 2. Navigate to the project directory

```bash
cd your-repository-name
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. (Optional) Re-create the model

Run `pipeline.ipynb` from top to bottom. The last modeling cell saves `attrition_model.joblib`.

### 7. Run the Streamlit application

```bash
streamlit run attrition_app.py
```

The application will open in your browser.

## 📦 Requirements

`requirements.txt` should list:

```text
streamlit
pandas
numpy
scikit-learn
joblib
cloudpickle
matplotlib
seaborn
openpyxl
```

> **Pin exact versions** (for example `scikit-learn==x.y.z`, using the versions in the environment where you ran the notebook). The saved model file is sensitive to the scikit-learn, pandas and Python versions it was created with, and the app shows a warning if the scikit-learn version differs. `cloudpickle` is required to load the model.

## 🌐 Deployment

The application can be deployed using **Streamlit Community Cloud**:

1. Push the project to GitHub, **including `attrition_model.joblib`**.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select this repository.
5. Select `attrition_app.py` as the main file.
6. In the advanced settings, choose a Python version that matches the one used to train the model.
7. Deploy the application.

## ⚠️ Disclaimer

This project is intended **for educational and demonstration purposes only**. The dataset is a well-known, publicly available HR dataset. The class imbalance (~16% attrition) and the leakage risk between `PercentSalaryHike` and `PerformanceRating` are documented and handled in the notebook.

The app's score is a **relative risk score, not a literal probability**: the model is tuned to catch employees who leave, so its scores run higher than the real attrition rate (about 16% in this data).

The model uses personal attributes such as age, gender and marital status. These should be reviewed for fairness before any real-world use. Predictions are **decision support, not authoritative judgments about individual employees**, and should not be the sole basis for real HR decisions.

## 👨‍💻 Author

Yoga Prabhu

Data Science | Machine Learning | Python

Portfolio: https://sybersage.github.io/Portfolio/

GitHub: `<add your GitHub profile link>`

LinkedIn: `<add your LinkedIn profile link>`

---

⭐ If you found this project useful, consider giving the repository a star!
