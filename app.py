import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px

# Load trained model and scaler
model = joblib.load("fraud_detection_model222.pkl")
scaler = joblib.load("scaler222.pkl")

# Define required feature columns
required_features = ['Transaction_Details', 'Cardholder_Information', 'Device_and_Network_Information',
                     'Historical_Data', 'Behavioral_Data', 'Security_Features', 'External_Data']

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stFileUploader>div>div>div>button {
        background-color: #008CBA;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualizations", "History", "Manual Input", "Feature Selection", "About"])

# Initialize session state to store predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame()

# Preprocessing function
def preprocess_data(df):
    """Preprocess uploaded data for model prediction."""
    df = df[required_features]
    for col in required_features:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    df = scaler.transform(df)
    return df

# Home Page
if page == "Home":
    st.title("🚨 Fraud Detection System")
    st.markdown("""
        Welcome to the **Fraud Detection System**! Upload your transaction data in CSV format, 
        and the system will predict the probability of fraud for each transaction.
        """)

    # File uploader
    uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if set(required_features).issubset(df.columns):
            st.success("✅ Data Loaded Successfully!")
            try:
                # Preprocess data and make predictions
                X = preprocess_data(df)
                fraud_prob = model.predict_proba(X)[:, 1]
                fraud_pred = model.predict(X)
                
                # Add predictions to the dataframe
                df['Fraud_Probability'] = fraud_prob
                df['Predicted_Fraud'] = fraud_pred
                df['Alerts_and_Notifications'] = df['Fraud_Probability'].apply(lambda x: "🔴 High Risk Alert" if x > 0.8 else "🟢 Low Risk")
                df['Historical_Insights'] = "📊 Previous fraud cases analyzed"
                df['User_Actions'] = df['Predicted_Fraud'].apply(lambda x: "🛑 Review Required" if x == 1 else "✅ Approved")
                df['Performance_Metrics'] = "📈 Model Accuracy: 95% (Example)"
                
                # Store predictions in session state
                st.session_state.predictions = df
                
                # Display results
                st.markdown("### 📊 Prediction Results:")
                st.dataframe(df[['Fraud_Probability', 'Predicted_Fraud', 'Alerts_and_Notifications', 'Historical_Insights', 'User_Actions', 'Performance_Metrics']].head(20))
                
                # Download option
                csv_output = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions", csv_output, "fraud_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
        else:
            st.error("❌ Uploaded file does not contain the required columns. Please ensure the file includes the following columns:")
            st.write(required_features)

# Visualization Page
elif page == "Visualizations":
    st.title("📊 Fraud Prediction Visualizations")
    if not st.session_state.predictions.empty:
        # Pie chart for fraud distribution
        fraud_distribution = st.session_state.predictions['Predicted_Fraud'].value_counts().reset_index()
        fraud_distribution.columns = ['Predicted_Fraud', 'Count']
        fig = px.pie(fraud_distribution, values='Count', names='Predicted_Fraud', title='Fraud Prediction Distribution')
        st.plotly_chart(fig)
        
        # Bar chart for fraud probability
        fig = px.histogram(st.session_state.predictions, x='Fraud_Probability', nbins=20, title='Fraud Probability Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("No predictions available. Please upload data on the Home page.")

# History Page
elif page == "History":
    st.title("📜 Prediction History")
    if not st.session_state.predictions.empty:
        st.dataframe(st.session_state.predictions)
    else:
        st.warning("No prediction history available. Please upload data on the Home page.")

# Manual Input Page

elif page == "Manual Input":
    st.title("✍️ Manual Transaction Input")
    st.markdown("Enter transaction details manually to get fraud predictions.")
    
    # Input fields for each feature
    transaction_details = st.number_input("Transaction Details", value=100000)
    cardholder_info = st.selectbox("Cardholder Information", ["Verified", "Unverified"])
    device_info = st.selectbox("Device and Network Information", ["Secure", "Compromised"])
    historical_data = st.number_input("Historical Data", value=1000.0)
    behavioral_data = st.number_input("Behavioral Data", value=0.5)
    security_features = st.selectbox("Security Features", ["Enabled", "Disabled"])
    external_data = st.selectbox("External Data", ["Low Risk", "Moderate Risk", "High Risk"])
    
    # Add a slider for threshold
    threshold = st.slider("Set Fraud Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    if st.button("Predict"):
        # Create a dataframe from the input
        input_data = pd.DataFrame({
            'Transaction_Details': [transaction_details],
            'Cardholder_Information': [cardholder_info],
            'Device_and_Network_Information': [device_info],
            'Historical_Data': [historical_data],
            'Behavioral_Data': [behavioral_data],
            'Security_Features': [security_features],
            'External_Data': [external_data]
        })
        
        # Preprocess and predict
        X = preprocess_data(input_data)
        fraud_prob = model.predict_proba(X)[:, 1]  # Get probabilities for the positive class
        fraud_pred = (fraud_prob > threshold).astype(int)  # Convert probabilities to binary predictions
        
        # Display results
        st.success(f"Fraud Probability: {fraud_prob[0]:.2f}")  # Access the first (and only) element
        st.success(f"Predicted Fraud: {'Yes' if fraud_pred[0] == 1 else 'No'}")  # Access the first (and only) element
# Feature Selection Page
elif page == "Feature Selection":
    st.title("⚙️ Feature Selection")
    st.markdown("Select the features to include in the analysis.")
    
    # Allow users to select features
    selected_features = st.multiselect("Choose Features", required_features, default=required_features)
    
    if st.button("Save Selection"):
        st.session_state.selected_features = selected_features
        st.success(f"Selected Features: {selected_features}")

# About Page
elif page == "About":
    st.title("📄 About the Project")
    st.markdown("""
        ### Fraud Detection System
        This application is designed to detect fraudulent transactions using machine learning. 
        It analyzes various features of transactions and predicts the likelihood of fraud.

        ### Features of the App
        - **Home Page**: Upload transaction data in CSV format and view fraud predictions.
        - **Visualizations**: Explore fraud prediction distributions using interactive charts.
        - **History**: View past predictions and analysis results.
        - **Manual Input**: Enter transaction details manually for real-time fraud predictions.
        - **Feature Selection**: Choose which features to include in the analysis.
        
        ### Project Purpose
        The goal of this project is to provide a user-friendly tool for detecting fraudulent transactions 
        in real-time, helping businesses and individuals mitigate risks.

        ### Technologies Used
        - **Python**: For backend logic and machine learning.
        - **Streamlit**: For building the interactive web interface.
        - **Scikit-learn**: For machine learning model training and preprocessing.
        - **Plotly**: For data visualizations.

        ### How to Use
        1. Upload a CSV file containing transaction data on the **Home** page.
        2. View fraud predictions and download the results.
        3. Explore visualizations on the **Visualizations** page.
        4. Use the **Manual Input** page for real-time predictions.
        5. Select features on the **Feature Selection** page.

        ### Contact
        For questions or feedback, please contact [Your Email].
        """)
