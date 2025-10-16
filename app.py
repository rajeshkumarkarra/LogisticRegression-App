import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    model = joblib.load('logistic_regression_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'logistic_regression_model.pkl' is in the same directory.")
    st.stop()

# Define the preprocessing steps (must match the training preprocessing)
# This function takes raw inputs from the Streamlit form and prepares them for the model
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    # Create a pandas DataFrame from input
    data = {'pclass': [pclass],
            'sex': [sex],
            'age': [age],
            'sibsp': [sibsp],
            'parch': [parch],
            'fare': [fare],
            'embarked': [embarked]}
    input_df = pd.DataFrame(data)

    # Handle missing values (based on training data's median/mode)
    # Use the median/mode values calculated from the original training notebook
    input_df['age'] = input_df['age'].fillna(28.0) # Median age from training_df
    input_df['fare'] = input_df['fare'].fillna(14.4542) # Median fare from training_df
    input_df['embarked'] = input_df['embarked'].fillna('S') # Mode embarked from training_df


    # Encode categorical variables (based on training data encoding)
    # 'Sex': 'female' -> 0, 'male' -> 1 (LabelEncoder)
    input_df['sex'] = input_df['sex'].map({'female': 0, 'male': 1})

    # 'Embarked': 'S' -> 0, 'C' -> 1, 'Q' -> 2 (pandas factorize based on unique_embarked order during training)
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2} # Based on the order from pd.factorize during training
    input_df['embarked'] = input_df['embarked'].map(embarked_mapping)
    # Handle potential NaN values if a category unseen during training is entered (though selectbox limits this)
    input_df['embarked'] = input_df['embarked'].fillna(-1) # Use -1 for unseen categories as factorize might

    # Select and reorder columns to match the features used during model training (X_train)
    # Features used were ['pclass', 'Sex_male', 'age', 'sibsp', 'parch', 'fare'] based on cell e9db61f6-9ee4-4850-ac81-763f55a265c9
    # Ensure column names match exactly
    processed_df = input_df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].copy()
    processed_df.rename(columns={'sex': 'Sex_male'}, inplace=True) # Rename 'sex' to 'Sex_male' to match training features


    return processed_df

# Streamlit App Title
st.title("Titanic Survival Prediction")

st.write("Enter passenger details to predict survival:")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0.0, 100.0, 30.0) # Use float for slider to match age data type
sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.slider("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", value=50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])


# Prediction button
if st.button("Predict Survival"):
    # Preprocess input data
    input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    # Display result
    if prediction[0] == 1:
        st.success(f"Predicted Survival: Yes (Probability: {prediction_proba[0]:.2f})")
    else:
        st.error(f"Predicted Survival: No (Probability: {prediction_proba[0]:.2f})")
