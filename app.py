import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🧠 Customer Segmentation with KMeans")

st.write("Enter customer details to predict which cluster they belong to:")

income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, value=50)

if st.button("Predict Cluster"):
    input_data = np.array([[income, score]])
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)
    cluster_names = {
        0: "Mid Score, Mid Income ,Cluster 0",
        1: "High Score, High Income ,Cluster 1",
        2: "Low Score, High Income ,Cluster 2",
        3: "Low Score, Low Income ,Cluster 3",
        4: "High Score, Low Income ,Cluster 4",
        
    }

    label = cluster_names.get(cluster[0], "Unknown Cluster")
    st.success(f"📌 This customer is in: **{label}**")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load original data
df = pd.read_csv("Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
df['Income'] = X['Annual Income (k$)']
df['Score'] = X['Spending Score (1-100)']

# Predict cluster for all customers
X_scaled = scaler.transform(X)
df['Cluster'] = model.predict(X_scaled)

# Visualization
st.subheader("📊 Customer Segmentation Visualization")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Income', y='Score', hue='Cluster', palette="Set2", ax=ax)
ax.set_title("Customer Segmentation (k=5)")
st.pyplot(fig)
