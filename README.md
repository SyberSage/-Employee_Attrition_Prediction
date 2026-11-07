# 👥 Employee Attrition Analysis and Prediction

A machine learning and HR analytics web application that **predicts which employees are at risk of leaving** and explains **why**, combining a full classification pipeline with an interactive Streamlit dashboard for exploring attrition trends and scoring employees live.

The application is built with **Python, scikit-learn and Streamlit**. HR teams can explore attrition patterns across the workforce, see the key drivers behind turnover, and view a ranked list of employees with the highest probability of leaving.

## 🚀 Live Demo

**Streamlit App:** `<add your deployed link here>`

## 📌 Project Overview

Employee turnover increases recruitment and training costs, reduces productivity and disrupts teams. This project analyzes the **IBM HR Analytics Employee Attrition dataset** (1,470 employees, 35 columns) to identify the factors that drive attrition and to build models that flag at-risk employees early, so retention action can be taken before they leave.

**Prediction targets**

* **Attrition** (primary): will the employee leave? (Yes / No)
* **Performance Rating** (secondary): see the leakage note in the Data section

The project covers the complete pipeline:

* 🧹 Data Preprocessing & Cleaning
* 📊 Exploratory Data Analysis (EDA)
* 🛠️ Feature Engineering (tenure categories, performance metrics, engagement scores)
* 🤖 Model Building & Comparison
* 📈 Model Evaluation (Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix)
* 🖥️ Interactive Streamlit Dashboard

## 💼 Business Use Cases

* **Employee Retention:** identify at-risk employees and target retention strategies.
* **Cost Optimization:** reduce recruitment, onboarding and training costs.
* **Workforce Planning:** align retention strategies with organizational goals and employee satisfaction.

## ✨ Features

* Sidebar navigation across Overview, Insights and Prediction pages
* Attrition trends by department, job role, overtime, income, tenure and satisfaction
* Key drivers of attrition surfaced from the trained model
* Ranked list of employees by predicted probability of leaving
* Live attrition prediction from custom employee inputs
* Imbalance-aware evaluation: recall and F1 are reported alongside accuracy, not hidden behind it

## 📂 Dataset

| Property | Detail |
| --- | --- |
| Source | IBM HR Analytics Employee Attrition dataset |
| File | `EmployeeAttrition.xlsx` |
| Size | 1,470 rows × 35 columns |
| Missing values | None |
| Class balance | 1,233 stayed (≈84%), 237 left (≈16%) |

**Data notes**

* `EmployeeCount`, `Over18` and `StandardHours` are constant, so they carry no information and were dropped.
* `EmployeeNumber` is only an identifier and is excluded from modeling (kept to label the at-risk list).
* `PerformanceRating` contains only the values 3 and 4 rather than the documented 1–4 range.
* `PercentSalaryHike` correlates strongly with `PerformanceRating` (≈0.77), which is a **data leakage risk** for the Performance Rating model and is handled explicitly.

## 🛠️ Feature Engineering

New features were created in the three categories required by the project brief, plus a few extra derived features:

| Category | Features |
| --- | --- |
| Tenure categories | Tenure bands based on `YearsAtCompany` |
| Performance metrics | Features built from performance rating, promotion and salary-hike columns |
| Engagement scores | Scores built from satisfaction, involvement and work-life-balance columns |
| Additional | `TenureRatio`, `IncomePerYearWorked`, `OverallSatisfaction` |

The engineered features were validated empirically: adding them gave a measurable improvement in **F1 and Recall** for Logistic Regression.

## 🧠 Machine Learning Models

The following scikit-learn models were trained and compared, all using `class_weight='balanced'` to handle the ~16% attrition minority class:

* Logistic Regression
* Decision Tree
* Random Forest

### Why not accuracy alone?

Because only ~16% of employees left, a model can score high accuracy while missing most leavers. In early experiments, Random Forest reached **84.4% accuracy but only 10.6% recall** on attrition, meaning it caught very few of the employees who actually left. For this reason, model selection focuses on **Recall, F1-score and AUC-ROC** in addition to accuracy.

### Results

> ⚠️ Fill this table with your final test-set numbers before submission. Do not copy figures from earlier experiments.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Train time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Decision Tree | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Random Forest | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

**Selected model:** `<name>`, chosen because `<reason, e.g. best Recall/F1 trade-off>`.

## 🔄 Machine Learning Workflow

```text
Raw Employee Dataset
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering
   ↓
Train/Test Split (stratified)
   ↓
Model Training (Logistic Regression, Decision Tree, Random Forest)
   ↓
Model Evaluation & Comparison
   ↓
Best Model Selection
   ↓
At-Risk Employee Ranking
   ↓
Streamlit Dashboard
```

## 🔑 Key Drivers of Attrition

> Fill this in from your feature importance / coefficient results.

* `<driver 1, e.g. OverTime>`
* `<driver 2, e.g. low job satisfaction>`
* `<driver 3, e.g. low monthly income>`

## 🎯 Retention Recommendations

> Fill this in from your findings. Suggested structure:

* **Finding:** `<what the data shows>` → **Action:** `<retention strategy>`

## 🛠️ Technologies Used

**Programming:** Python

**Data Science:** Pandas, NumPy, Scikit-learn

**Visualization:** Matplotlib, Seaborn

**Dashboard:** Streamlit

**Development Tools:** Jupyter Notebook, VS Code, GitHub

## 📂 Project Structure

> Adjust the file names to match your repository.

```text
Employee-Attrition-Analysis-and-Prediction/
│
├── 01_preprocessing_and_eda.ipynb
├── 02_feature_engineering.ipynb
├── 03_model_building.ipynb
│
├── app.py
│
├── EmployeeAttrition.xlsx
│
├── requirements.txt
│
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

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Requirements

Example `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
joblib
```

If your application uses additional packages, add them to `requirements.txt`.

## 🌐 Deployment

The application can be deployed using **Streamlit Community Cloud**:

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select this repository.
5. Select `app.py` as the main file.
6. Deploy the application.

## ⚠️ Disclaimer

This project is intended **for educational and demonstration purposes only**. The dataset is a well-known, publicly available HR dataset, and the class imbalance (~16% attrition) and the leakage risk between `PercentSalaryHike` and `PerformanceRating` are documented and handled in the notebooks. Predictions should be treated as **decision support, not authoritative judgments about individual employees**, and should not be the sole basis for real HR decisions.

## 👨‍💻 Author

Yoga Prabhu

Data Science | Machine Learning | Python

Portfolio: https://sybersage.github.io/Portfolio/

GitHub: `<add your GitHub profile link>`

LinkedIn: `<add your LinkedIn profile link>`

---

⭐ If you found this project useful, consider giving the repository a star!
