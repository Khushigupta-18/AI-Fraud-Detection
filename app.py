import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

# Load the trained model
model = joblib.load(r"D:\Fintech_AI_Project\fraud_detection_model.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("Machine Learning + EDA + Fraud Analytics")

# Load data
df = pd.read_csv(r"D:\Fintech_AI_Project\creditcard_cleaned.csv")

# Dataset Overview
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), width="stretch")

with col2:
    st.info(f"""
Rows: {df.shape[0]:,}

Columns: {df.shape[1]}
""")

# KPI Metrics
total_transactions = len(df)
fraud_transactions = df["Class"].sum()
normal_transactions = total_transactions - fraud_transactions
fraud_rate = (fraud_transactions / total_transactions) * 100

st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Fraud Transactions", f"{fraud_transactions:,}")
col3.metric("Normal Transactions", f"{normal_transactions:,}")
col4.metric("Fraud Rate", f"{fraud_rate:.4f}%")

# Dataset Statistics
st.subheader("📈 Dataset Statistics")
st.dataframe(df.describe(), width="stretch")

# Fraud vs Non-Fraud
st.subheader("Fraud vs Non-Fraud Transactions")

fig = px.histogram(
    df,
    x="Class",
    title="Transaction Distribution"
)

st.plotly_chart(fig, width="stretch")

# Amount Distribution
st.subheader("Transaction Amount Distribution")

fig = px.histogram(
    df,
    x="Amount",
    nbins=50,
    title="Amount Distribution"
)

st.plotly_chart(fig, width="stretch")

# Fraud by Hour
st.subheader("Fraud Transactions by Hour")

fraud_hour = (
    df[df["Class"] == 1]
    .groupby("Hour")
    .size()
    .reset_index(name="Fraud_Count")
)

fig = px.line(
    fraud_hour,
    x="Hour",
    y="Fraud_Count",
    markers=True,
    title="Fraud Transactions by Hour"
)

st.plotly_chart(fig, width="stretch")

# Sidebar Filters
st.sidebar.header("Filters")

selected_hour = st.sidebar.slider(
    "Select Hour Range",
    0,
    23,
    (0, 23)
)

filtered_df = df[
    (df["Hour"] >= selected_hour[0]) &
    (df["Hour"] <= selected_hour[1])
]

st.subheader("Filtered Transactions")
st.dataframe(filtered_df.head(20), width="stretch")


# Random Forest

st.subheader("🌟 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values("Importance", ascending=False)
    .head(10)
)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Important Features"
)

st.plotly_chart(fig, width="stretch")

# COnfusion Matrix

X = df.drop("Class", axis=1)
y = df["Class"]

y_pred = model.predict(X)

st.subheader("🎯 Confusion Matrix")

cm = confusion_matrix(y, y_pred)

fig = px.imshow(
    cm,
    text_auto=True,
    labels=dict(
        x="Predicted",
        y="Actual"
    ),
    title="Confusion Matrix"
)

st.plotly_chart(fig, width="stretch")

# ROC Curve
y_prob = model.predict_proba(X)[:,1]

fpr, tpr, _ = roc_curve(
    y,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

st.subheader("📉 ROC Curve")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"AUC = {roc_auc:.4f})"
    )
)

fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
    )
)

fig.update_layout(
    title="ROC Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    width=800,
    height=600
)

st.plotly_chart(fig, width="stretch")


# Live Fraud Prediction Tool

st.sidebar.header("🚀 Fraud Prediction")

amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

hour = st.sidebar.slider(
    "Hour",
    0,
    23,
    12
)

row_id = st.sidebar.number_input(
    "Transaction Row",
    0,
    len(df)-1,
    0
)

if st.sidebar.button("Predict"):

    sample = (
        df
        .drop("Class", axis=1)
        .iloc[[row_id]]
    )

    prediction = model.predict(sample)[0]

    if prediction == 1:
        st.error("🚨 Fraud Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")