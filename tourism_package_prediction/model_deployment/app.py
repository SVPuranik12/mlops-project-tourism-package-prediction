import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="svpuranik/tourism-package-prediction-models", filename="optimal_tourism_package_prediction_model_v1.joblib")

# Load the model
tourism_pakeage_predictor = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Visit With Us : Tourism Package Prediction App")
st.write(">> This app predicts the potential buyer of the tourism package offered by the company")
st.write(">> Use the text box to add the values and dropdown to select an option to get a prediction.")

# Collect user input
st.write(">> Customer details section:")
gender                      = st.selectbox("Gender", ["Male", "Female"])
age_customer                = st.number_input("Age of the customer (in years)", min_value=18, max_value=65, value=18)
marital_status              = st.selectbox("Marital status of the customer", ["Single", "Divorced", "Married"])
number_of_person_visiting   = st.number_input("Number of persons accompaning the customer on the trip ", min_value=1, max_value=10, value=1)
number_of_children_visiting = st.number_input("Number of children (below age 5 years) accompaning the customer on the trip ", min_value=0, max_value=10, value=0)
number_of_trips             = st.number_input("Number of trips customer is taking yearly on an average ", min_value=1, max_value=10, value=1)
occupation                  = st.selectbox("Customer's Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
designation                 = st.selectbox("Customer's designation in organization", ["Manager", "Senior Manager", "Executive", "AVP", "VP"])
monthly_income              = st.number_input("Monthly income of the customer ", min_value=1000.00, max_value=100000.00, value=1000.00)
passport                    = st.selectbox("Valid Passport Availability",  ["Yes", "No"])
own_car                     = st.selectbox("Customer owns Car", ["Yes", "No"])
city_tier                   = st.selectbox("City Tier", ["1", "2", "3"])

st.write(">> Company Interaction details section:")
type_of_contact             = st.selectbox("Customer connect method", ["Self Enquiry", "Company Invited"])
product_pitched             = st.selectbox("Product pitched to the customer", ["Standard", "Basic", "King", "Deluxe", "Super Deluxe"])
duration_of_pitch           = st.number_input("Duration of sales pitch ", min_value=5, max_value=180, value=5)
number_of_followups         = st.number_input("Number of followups by sales person ", min_value=1, max_value=10, value=1)
preferred_property_star     = st.selectbox("Prefered Property Stars by the customer", ["1", "2", "3", "4", "5"])
pitch_satisfaction_score    = st.selectbox("Pitch Satisfaction Score given by the customer", ["1", "2", "3", "4", "5"])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': age_customer,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': 1 if passport == "Yes" else 0,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': 1 if own_car == "Yes" else 0,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = tourism_pakeage_predictor.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "to buy" if prediction == 1 else "not to buy"
    st.write(f"Based on the customer details and company interaction information provided, the customer is likely **{result}** the tourism package offered by the company.")
