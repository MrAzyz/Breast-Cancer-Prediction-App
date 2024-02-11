import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import plotly.graph_objs as go
import numpy as np

st.set_page_config(page_title="BREASTCANCER PREDICTION",page_icon=":cancer:",layout="centered")

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

# Load data
data = pd.read_csv("Dataset/data.csv")

# Cleaning data
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Divide into target variable and predictors
y = data["diagnosis"]
X = data.drop(["diagnosis"], axis=1)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)

# Train the model
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Function to predict probabilities
def predict_proba(input_data):
    input_data_scaled = scaler.transform(input_data)
    proba = lr.predict_proba(input_data_scaled)
    return proba

def main():
    st.title("Breast Cancer Prediction")

    # Sidebar with input parameters
    st.sidebar.title("Input Parameters")
    input_data = {}
    for column in X.columns:
        min_val = X[column].min()
        max_val = X[column].max()
        default_val = X[column].mean()
        input_data[column] = st.sidebar.slider(f"{column} ({min_val:.2f} - {max_val:.2f})", min_val, max_val, default_val)

    input_df = pd.DataFrame(input_data, index=[0])

    # Make prediction based on input
    proba = predict_proba(input_df)

    # Convert probabilities to percentages
    proba_percent = [f"{p*100:.2f}%" for p in proba.flatten()]

    # Plot the probabilities using Plotly pie chart
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=["Not Sick", "Sick"],
        values=proba.flatten(),
        textinfo='label+percent',
        marker=dict(colors=['blue', 'red'])
    ))

    fig.update_layout(
        title='Probability of Being Sick',
    )

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
