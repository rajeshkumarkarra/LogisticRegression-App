# Titanic Survival Prediction App

This repository contains the code and resources for building and deploying a simple logistic regression model to predict survival on the Titanic, and serving it as a web application using Streamlit.

## Project Overview

The goal of this project is to demonstrate a basic machine learning workflow, including data loading, exploration, preprocessing, model training, evaluation, and deployment.

## Files in this Repository

*   `app.py`: The Streamlit application script that loads the trained model and provides a web interface for predictions.
*   `logistic_regression_model.pkl`: The trained logistic regression model saved using `joblib`.
*   `requirements.txt`: Lists the Python libraries required to run the Streamlit application.
*   `README.md`: This file, providing an overview of the project.

## Project Steps

The following steps were performed in the accompanying Colab notebook (though the notebook itself is not included here):

1.  **Data Loading**: The Titanic dataset (`titanic.csv`) was downloaded and loaded into pandas DataFrames.
2.  **Data Exploration**: Initial data analysis was performed to understand the dataset's structure, summary statistics, and identify missing values. Visualizations were created to explore relationships between features and the target variable (Survived).
3.  **Data Preprocessing**:
    *   Missing values in 'Age', 'Fare', and 'Embarked' columns were handled (imputation with median/mode).
    *   Categorical features ('Sex', 'Embarked') were encoded into numerical representations using Label Encoding and pandas `factorize`.
    *   Irrelevant columns ('Cabin', 'Ticket', 'Name') were dropped.
4.  **Model Building & Training**:
    *   A logistic regression model was chosen for its simplicity and interpretability.
    *   The data was split into training and validation sets.
    *   The logistic regression model was trained on the training data.
5.  **Model Evaluation**: The trained model was evaluated on the validation set using metrics like accuracy, precision, recall, F1-score, and ROC-AUC. An ROC curve was plotted.
6.  **Model Saving**: The trained logistic regression model object was saved to a file (`logistic_regression_model.pkl`) using the `joblib` library.
7.  **Streamlit Application Script**: A Python script (`app.py`) was created to build the Streamlit web application. This script includes:
    *   Loading the saved model.
    *   A function to preprocess user input to match the format expected by the model.
    *   Streamlit components for user input (sliders, selectboxes, number input).
    *   Code to make predictions and display the results.
8.  **Requirements File**: A `requirements.txt` file was created listing the Python libraries needed for the Streamlit app (`streamlit`, `pandas`, `scikit-learn==1.6.1`, `joblib`, `numpy`). Note the explicit version for `scikit-learn` to ensure compatibility.
9.  **Deployment (using Streamlit Cloud)**:
    *   The `app.py`, `logistic_regression_model.pkl`, and `requirements.txt` files were added to a GitHub repository.
    *   The application was deployed using Streamlit Cloud by pointing it to the GitHub repository.

## Running the Streamlit App Locally (Optional)

If you want to run the app locally on your machine:

1.  Clone this repository: `git clone https://github.com/YOUR_USERNAME/LogisticRegression-App.git`
2.  Navigate to the repository directory: `cd LogisticRegression-App`
3.  Create a virtual environment (recommended): `python -m venv venv`
4.  Activate the virtual environment:
    *   Windows: `venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`
5.  Install the required libraries: `pip install -r requirements.txt`
6.  Run the Streamlit app: `streamlit run app.py`

This will start the Streamlit app, and you can access it in your web browser, usually at `http://localhost:8501`.

## Deployment to Streamlit Cloud

This application was designed for easy deployment on Streamlit Cloud. Simply:

1.  Fork this repository or create a new one and add the files.
2.  Go to [Streamlit Cloud](https://share.streamlit.io/) and log in.
3.  Click "New app" and select your GitHub repository.
4.  Configure the branch and main file path (`app.py`).
5.  Click "Deploy!".

## License

[Specify your license here, e.g., MIT License]

## Acknowledgments

*   Dataset from Kaggle (Titanic - Machine Learning from Disaster)
*   Streamlit for providing an easy way to deploy data apps.
*   Colaboratory for the interactive notebook environment.
