# Breast Cancer Prediction App

This Streamlit app predicts whether a person is likely to have breast cancer based on input parameters such as radius, texture, perimeter, area, and smoothness.

## Dataset

The dataset used for training the model is the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). The data has been preprocessed to remove irrelevant columns and to convert the diagnosis column to binary values (1 for malignant, 0 for benign).

## Model

The prediction model used is Logistic Regression, trained on the preprocessed dataset. The model achieves an accuracy score of approximately 95%.

## Usage

To run the app locally, follow these steps:

1. Clone this repository:
   ``` git clone https://github.com/MrAzyz/Breast-Cancer-Prediction-App.git ```
2. Navigate to the project directory:
   ``` cd Breast-Cancer-Prediction-App ```
3. Install the required dependencies:
   ``` pip install -r requirements.txt ```
4. Run the Streamlit app:
   ``` streamlit run app.py ```
5.  Open your web browser and go to `http://localhost:8501` to view the app.

## Input Parameters

- The user can adjust input parameters using sliders in the sidebar. The parameters include radius, texture, perimeter, area, and smoothness.

## Prediction

- The app displays the probability of the person being sick with breast cancer as a percentage. This probability is based on the input parameters provided by the user.
   





