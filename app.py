import pickle
from pathlib import Path

import cloudpickle  
import joblib
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="Employee Attrition Predictor")

# ---------------------------------------------------------------------------
# Load the saved model (pipeline = feature engineering + encoding/scaling + model)
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "attrition_model.joblib"


@st.cache_resource  # load once, not on every click
def load_artifact():
    # Only unpickle a file you created yourself (pickle can run code when loaded).
    return pickle.loads(joblib.load(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error("Model file `attrition_model.joblib` not found. Run the notebook's save cell, "
             "then place the file in the same folder as this app.")
    st.stop()

artifact = load_artifact()
model = artifact["pipeline"]
input_columns = artifact["input_columns"]

if artifact.get("sklearn_version") != sklearn.__version__:
    st.warning(
        f"This model was saved with scikit-learn {artifact.get('sklearn_version')} but the app is running "
        f"{sklearn.__version__}. Predictions may be unreliable; use the same version in both places."
    )


def risk_level(score: float) -> str:
    if score >= 0.50:   # same cut-off the model itself uses to say "will leave"
        return "High"
    if score >= 0.30:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Page text
# ---------------------------------------------------------------------------
st.title("Employee Attrition Predictor")
st.markdown("""
This app estimates how likely an employee is to leave the company (**Attrition**),
using a Logistic Regression model trained on the IBM HR Analytics dataset.

The model was selected after comparing four algorithms — Logistic Regression, Decision Tree,
Random Forest, and K-Nearest Neighbors — on Accuracy, Precision, Recall, F1, and AUC.
Logistic Regression had the highest AUC (0.817), meaning it ranks at-risk employees above
the rest better than the other models, and also the best Recall (72.3%), meaning it catches
the most employees who actually left. It does not have the highest raw accuracy.

Enter an employee's details below to get a risk score.
""")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("employee_form"):

    st.subheader("Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=60, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with col2:
        distance_from_home = st.number_input("Distance From Home (miles)", min_value=1, max_value=30, value=5)
        education = st.slider("Education Level (1=Below College, 5=Doctor)", 1, 5, 3)
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree",
            "Human Resources", "Other"
        ])

    st.subheader("Job Details")
    col3, col4 = st.columns(2)
    with col3:
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ])
        job_level = st.slider("Job Level", 1, 5, 2)
    with col4:
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=1)

    st.subheader("Compensation")
    col5, col6 = st.columns(2)
    with col5:
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=25000, value=5000)
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
    with col6:
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65)
        monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=27000, value=14000)
        percent_salary_hike = st.slider("Percent Salary Hike (%)", 11, 25, 15)

    st.subheader("Satisfaction & Engagement")
    col7, col8 = st.columns(2)
    with col7:
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
    with col8:
        work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
        job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
        stock_option_level = st.slider("Stock Option Level (0-3)", 0, 3, 1)

    st.subheader("Tenure & Work History")
    col9, col10 = st.columns(2)
    with col9:
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=8)
        years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
        years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=3)
    with col10:
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
        years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=3)
        training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)

    submitted = st.form_submit_button("Predict Attrition Risk")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:

    # Catch impossible combinations before predicting
    problems = []
    if years_in_current_role > years_at_company:
        problems.append("Years In Current Role can't be more than Years At Company.")
    if years_with_curr_manager > years_at_company:
        problems.append("Years With Current Manager can't be more than Years At Company.")
    if years_since_last_promotion > years_at_company:
        problems.append("Years Since Last Promotion can't be more than Years At Company.")
    if years_at_company > total_working_years:
        problems.append("Years At Company can't be more than Total Working Years.")
    if problems:
        for message in problems:
            st.error(message)
        st.stop()

    raw_input = {
        "Age": age, "BusinessTravel": business_travel, "DailyRate": daily_rate,
        "Department": department, "DistanceFromHome": distance_from_home,
        "Education": education, "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction, "Gender": gender,
        "HourlyRate": hourly_rate, "JobInvolvement": job_involvement,
        "JobLevel": job_level, "JobRole": job_role,
        "JobSatisfaction": job_satisfaction, "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income, "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked, "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance, "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    }

    # Raw columns only, in the order the model was trained on.
    # The saved pipeline does the feature engineering, encoding and scaling itself.
    df_input = pd.DataFrame([raw_input])[input_columns]

    score = float(model.predict_proba(df_input)[0][1])
    level = risk_level(score)

    st.subheader("Result")
    message = f"{level} attrition risk — risk score: {score:.1%}"
    if level == "High":
        st.error(message)
    elif level == "Medium":
        st.warning(message)
    else:
        st.success(message)

    st.progress(min(max(score, 0.0), 1.0))
    st.caption(
        "This is a relative risk score, not a literal probability. The model is tuned to catch "
        "employees who leave. High means 50% or more, Medium 30–50%, Low under 30%. "
        "Use it to prioritise conversations, not to make decisions about individuals."
    )


st.divider()
st.subheader("A note on Performance Rating")
st.markdown("""
This project also explored predicting **Performance Rating** as a second target, but that
model isn't included here because it isn't reliable enough to use.

After removing a feature that was leaking the answer directly (`PercentSalaryHike`, which
correlated ~0.77 with Performance Rating), four different models — Logistic Regression,
Decision Tree, Random Forest, and K-Nearest Neighbors — all scored an AUC between 0.44 and
0.53. That's roughly the same as random guessing. A correlation check across all remaining
features confirmed this: none exceeded a correlation of ±0.053 with Performance Rating.

In short, this dataset doesn't contain enough genuine signal to predict Performance Rating,
so no such prediction is offered here rather than presenting an unreliable one.
""")
