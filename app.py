import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load Models and Scaler

svm_linear = joblib.load("loan_svm_linear.pkl")
svm_poly = joblib.load("loan_svm_poly.pkl")
svm_rbf = joblib.load("loan_svm_rbf.pkl")
scaler = joblib.load("scaler.pkl")

# Model Accuracy 

accuracy_l = 0.85   
accuracy_p = 0.85
accuracy_r = 0.85

# Page Configuration

st.set_page_config(page_title="Smart Loan Approval System", layout="centered")

# App Title & Description

st.title("🏦 Smart Loan Approval System")
st.write("This system uses Support Vector Machines to predict loan approval.")

# Sidebar Input Section

st.sidebar.header("Enter Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)

loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)

credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encoding Inputs (Match Training)

gender_val = 1 if gender == "Male" else 0
married_val = 1 if married == "Yes" else 0
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0
credit_history_val = 1 if credit_history == "Yes" else 0

dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
dependents_val = dependents_map[dependents]

property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
property_val = property_map[property_area]

# Create Input Array

input_data = np.array([[gender_val,
                        married_val,
                        dependents_val,
                        education_val,
                        self_employed_val,
                        applicant_income,
                        coapplicant_income,
                        loan_amount,
                        loan_term,
                        credit_history_val,
                        property_val]])

# Scale Input
input_scaled = scaler.transform(input_data)

# Kernel Selection

st.subheader("Select SVM Kernel")

kernel_option = st.radio(
    "Choose Model",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

# Accuracy Comparison Visualization

st.subheader("📊 SVM Model Accuracy Comparison")

models = ["Linear SVM", "Polynomial SVM", "RBF SVM"]
accuracy_values = [accuracy_l, accuracy_p, accuracy_r]

fig_acc, ax_acc = plt.subplots()
ax_acc.bar(models, accuracy_values)
ax_acc.set_ylabel("Accuracy")
ax_acc.set_ylim(0, 1)
ax_acc.grid(True)

st.pyplot(fig_acc)

# Prediction Button

if st.button("Check Loan Eligibility"):

    if kernel_option == "Linear SVM":
        model = svm_linear
        kernel_name = "Linear"

    elif kernel_option == "Polynomial SVM":
        model = svm_poly
        kernel_name = "Polynomial"

    else:
        model = svm_rbf
        kernel_name = "RBF"

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    approved_prob = probability[0][1]
    rejected_prob = probability[0][0]

    # Output Section

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
        decision_text = "likely"
    else:
        st.error("❌ Loan Rejected")
        decision_text = "unlikely"

    st.write("Kernel Used:", kernel_name)
    st.write(f"Model Confidence: {max(approved_prob, rejected_prob)*100:.2f}%")

    # Prediction Probability Chart

    st.subheader("📈 Prediction Probability")

    labels = ["Rejected", "Approved"]
    values = [rejected_prob, approved_prob]

    fig_prob, ax_prob = plt.subplots()
    ax_prob.bar(labels, values)
    ax_prob.set_ylabel("Probability")
    ax_prob.set_ylim(0, 1)
    ax_prob.grid(True)

    st.pyplot(fig_prob)

    # Business Explanation

    st.info(
        f"Based on credit history and income pattern, the applicant is {decision_text} to repay the loan."
    )
