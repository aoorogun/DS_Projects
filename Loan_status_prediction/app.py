import streamlit as st
import pandas as pd
from sklearn import svm
from joblib import load
from sklearn.preprocessing import MinMaxScaler

@st.cache
def load_model():
    # Load the model
    model = load('/Users/adebolaorogun/DS_Projects/Loan_status_prediction/model/svm_model.joblib')
    return model

def main():
    st.title('Loan Approval Prediction')

    model = load_model()
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.number_input('Applicant Income', min_value=0)
    coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
    loan_amount = st.number_input('Loan Amount', min_value=0)
    loan_amount_term = st.number_input('Loan Amount Term', min_value=0)
    credit_history = st.selectbox('Credit History', [0, 1])
    property_area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])

 # Predict button
    if st.button('Predict Loan Status'):
        # Preprocess inputs
        input_data = pd.DataFrame([[gender, married, dependents, education, self_employed, 
                                    applicant_income, coapplicant_income, loan_amount, 
                                    loan_amount_term, credit_history, property_area]],
                                columns=['Gender', 'Married', 'Dependents', 'Education',
                                        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
                                        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
                                        'Property_Area'])

        # Check for null or missing values in 'CoapplicantIncome'
        if input_data['CoapplicantIncome'].isnull().any():
            st.error("Missing value in Coapplicant Income.")
            return

        # Initialize a new MinMaxScaler
        scaler = MinMaxScaler()

        # Reshape and Scale CoapplicantIncome
        input_data['CoapplicantIncome'] = scaler.fit_transform(input_data['CoapplicantIncome'].values.reshape(-1, 1))

        # Replace categorical values
        input_data.replace({'Married': {'No': 0, 'Yes': 1},
                            'Gender': {'Male': 1, 'Female': 0},
                            'Self_Employed': {'No': 0, 'Yes': 1},
                            'Education': {'Graduate': 1, 'Not Graduate': 0},
                            'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                            # Add other categorical replacements if necessary
                            }, inplace=True)

        # Predict
        prediction = model.predict(input_data)

        # Output
        st.success(f'The predicted loan status is: {prediction[0]}')

if __name__ == '__main__':
    main()
