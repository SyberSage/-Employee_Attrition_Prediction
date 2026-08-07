import streamlit as st
import pandas as pd
import joblib

model = joblib.load("attrition_model.pkl")

st.title("Employee Attrition Predictor")
st.markdown("""
This app estimates whether an employee is likely to leave the company (**Attrition**),
based on a Logistic Regression model trained on the IBM HR Analytics dataset.
 
The model was selected after comparing four algorithms — Logistic Regression, Decision Tree,
Random Forest, and K-Nearest Neighbors — on Accuracy, Precision, Recall, F1, and AUC.
Logistic Regression was chosen for its best Recall (72.3%) and AUC (0.817), meaning it catches
the most actual at-risk employees — the metric that matters most for this use case, even though
it doesn't have the single highest raw accuracy.
 
Enter an employee's details below to get a prediction.
""")
 
with st.form("employee_form"):

    st.subheader("Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=60, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with col2:
        distance_from_home = st.number_input("Distance From Home (km)", min_value=1, max_value=30, value=5)
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


if submitted:

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

    df_input = pd.DataFrame([raw_input])

    # --- Feature Engineering ---
    df_input["TenureRatio"] = df_input["YearsInCurrentRole"] / df_input["YearsAtCompany"].replace(0, 1)
    df_input["IncomePerYearWorked"] = df_input["MonthlyIncome"] / (df_input["TotalWorkingYears"] + 1)

    satisfaction_cols = ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]
    df_input["OverallSatisfaction"] = df_input[satisfaction_cols].mean(axis=1)

    df_input["TenureCategory"] = pd.cut(
        df_input["YearsAtCompany"], bins=[-1, 2, 5, 10, 100],
        labels=["New (0-2y)", "Early (3-5y)", "Established (6-10y)", "Veteran (10y+)"]
    )

    
    hike_scaled = (df_input["PercentSalaryHike"] - 11) / (25 - 11)
    training_scaled = (df_input["TrainingTimesLastYear"] - 0) / (6 - 0)
    df_input["PerformanceScore"] = (hike_scaled + training_scaled) / 2

    involvement_scaled = (df_input["JobInvolvement"] - 1) / (4 - 1)
    stock_scaled = (df_input["StockOptionLevel"] - 0) / (3 - 0)
    df_input["EngagementScore"] = (involvement_scaled + stock_scaled) / 2

    df_input["AgeGroup"] = pd.cut(
        df_input["Age"], bins=[17, 25, 35, 45, 60],
        labels=["18-25", "26-35", "36-45", "46-60"]
    )

    # --- Predict ---
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"High attrition risk — estimated probability: {probability:.1%}")
    else:
        st.success(f"Low attrition risk — estimated probability: {probability:.1%}")

    st.progress(min(probability, 1.0))


st.divider()
st.subheader("A note on Performance Rating")
st.markdown("""
This project also explored predicting **Performance Rating** as a second target, but that
model isn't included here because it isn't reliable enough to use.
 
After removing a feature that was leaking the answer directly (`PercentSalaryHike`, which
correlated ~0.77 with Performance Rating), four different models — Logistic Regression,
Decision Tree, Random Forest, and K-Nearest Neighbors — all scored an AUC between 0.44 and
0.53. That's roughly the same as random guessing. A correlation check across all remaining
features confirmed this: none exceeded a correlation of ±0.045 with Performance Rating.
 
In short, this dataset doesn't contain enough genuine signal to predict Performance Rating,
so no such prediction is offered here rather than presenting an unreliable one.
""")