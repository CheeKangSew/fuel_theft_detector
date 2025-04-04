import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
import openai
import matplotlib.pyplot as plt

# 
openai.api_key = os.getenv("OPENAI_API_KEY")

def prepare_features(df):
    df['TransactionDateTime'] = pd.to_datetime(df['TransactionDate'] + ' ' + df['TransactionTime'], errors='coerce', dayfirst=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['FuelQuantity'] = df['Quantity']
    df['CostPerLiter'] = df['Amount'] / df['FuelQuantity']
    df['Hour'] = df['TransactionDateTime'].dt.hour
    df['DayOfWeek'] = df['TransactionDateTime'].dt.dayofweek

    feature_cols = ['FuelQuantity', 'CostPerLiter', 'Hour', 'DayOfWeek']
    feature_df = df[feature_cols].fillna(0)
    return df, feature_df

def detect_anomalies_with_isolation_forest(df, features):
    clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df['AnomalyScore'] = clf.fit_predict(features)
    df['AnomalyValue'] = clf.decision_function(features)
    df['Flagged'] = df['AnomalyScore'] == -1
    return df

def generate_explanation_gpt(row):
    prompt = (
        f"Review the following fuel transaction for suspicious activity and explain why it may be flagged:\n"
        f"Driver: {row['DriverFullName']}\n"
        f"Vehicle: {row['VehicleRegistrationNo']}\n"
        f"Date & Time: {row['TransactionDateTime']}\n"
        f"Fuel Quantity: {row['FuelQuantity']} L\n"
        f"Cost per Liter: RM {row['CostPerLiter']:.2f}\n"
        f"Station: {row['PetrolStationName']}\n"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a fuel fraud detection analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"[GPT Error] {str(e)}"

st.title("🚨 Fuel Theft Detection Dashboard (ML-Based)")

uploaded_file = st.file_uploader("Upload a fuel transaction CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df, features = prepare_features(df)
    df = detect_anomalies_with_isolation_forest(df, features)

    drivers = df['DriverFullName'].dropna().unique().tolist()
    selected_driver = st.selectbox("Filter by driver (optional)", ["All"] + drivers)

    if selected_driver != "All":
        df = df[df['DriverFullName'] == selected_driver]

    flagged_df = df[df['Flagged']].copy()
    if not flagged_df.empty:
        flagged_df['Explanation'] = flagged_df.apply(generate_explanation_gpt, axis=1)

        st.subheader("🔎 Flagged Transactions")
        st.dataframe(flagged_df[['TransactionNo', 'TransactionDateTime', 'DriverFullName', 'FuelQuantity', 'CostPerLiter', 'AnomalyValue']])

        csv = flagged_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Flagged Transactions as CSV",
            data=csv,
            file_name='flagged_transactions.csv',
            mime='text/csv'
        )

        st.subheader("🌐 Anomaly Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['AnomalyValue'], bins=50, color='skyblue', edgecolor='black')
        ax.set_title("Distribution of Anomaly Scores")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Number of Transactions")
        st.pyplot(fig)

    else:
        st.success("No suspicious transactions found.")

    st.subheader("🧑‍💬 Ask the AI Analyst")
    user_query = st.text_input("Ask about flagged transactions, fuel patterns, or driver behavior:")
    if user_query:
        try:
            context_data = flagged_df[['DriverFullName', 'VehicleRegistrationNo', 'TransactionDateTime', 'FuelQuantity', 'CostPerLiter']].head(10).to_string()
            prompt = (
                f"Context:\n{context_data}\n\n"
                f"User Question: {user_query}\n"
                f"Answer as an expert fuel fraud analyst."
            )
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that explains suspicious fuel activity."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250
            )
            st.markdown("**Response:**")
            #st.write(response.choices[0].message['content'].strip())
            st.write(response.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"[GPT Error] {str(e)}")
else:
    st.info("Please upload a CSV file to begin analysis.")
