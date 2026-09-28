[readme_md.md](https://github.com/user-attachments/files/32773396/readme_md.md)
# 💳 Credit Card Fraud Detection System

An interactive, end-to-end Machine Learning web application designed to detect fraudulent credit card transactions in real time. Built with **Streamlit**, **Scikit-learn**, and **Plotly**, this tool provides both batch file prediction and single-transaction risk scoring to help businesses and financial institutions mitigate fraudulent activities.

---

## 📌 Features

- **🏠 Home / Batch Predictions:** Upload transaction datasets in CSV format, process multiple transactions simultaneously, and download prediction outputs.
- **✍️ Manual Input (Real-Time Scoring):** Input specific transaction parameters manually to receive instant fraud risk classifications.
- **📊 Interactive Visualizations:** Dynamic data charts powered by Plotly to analyze distribution patterns, anomaly clusters, and fraud risk metrics.
- **⚙️ Feature Selection:** Dynamically toggle and choose which transaction features to factor into the model inference.
- **📜 History Tracking:** Review logs of prior predictions and data analysis runs within the active session.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.10
- **Web Interface:** [Streamlit](https://streamlit.io/)
- **Machine Learning & Preprocessing:** [Scikit-learn](https://scikit-learn.org/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Interactive Visualizations:** [Plotly](https://plotly.com/python/)

---

## 📂 Project Structure

```text
├── app.py                      # Main Streamlit web application
├── c61_.ipynb                  # Jupyter notebook: EDA, feature engineering & model training
├── scaler222.pkl               # Preprocessing scaler for input normalization
├── transaction_data.csv        # Sample dataset for testing batch uploads
├── requirements.txt            # Project dependencies and versions
└── README.md                   # Project documentation
```

> **Note on Model Weights:** The trained classification model artifact (`fraud_detection_model222.pkl`) exceeds browser upload limits. Download or place the model file directly in the root directory before running the application.

---

## 📱Screenshots

<img width="1917" height="985" alt="Screenshot 2026-09-29 015400" src="https://github.com/user-attachments/assets/53350404-2c46-4338-8cd8-13f6041eb6e9" /><img width="1917" height="980" alt="Screenshot 2026-09-29 020119" src="https://github.com/user-attachments/assets/807d8339-7c4c-41ff-b749-8d4a0d99f96b" />
<img width="650" height="872" alt="Screenshot 2026-09-29 015912" src="https://github.com/user-attachments/assets/741cacef-78d3-4211-bf1a-10815bef6f07" /><img width="648" height="867" alt="Screenshot 2026-09-29 015433" src="https://github.com/user-attachments/assets/9909fa09-eff6-44cf-bfa6-2e8bf1514921" />
![Uploading Screenshot 2026-09-29 015400.png…]()


---
## 🚀 Getting Started Locally

### 1. Clone or Download the Repository
```bash
git clone https://github.com/<khan-abdullah-junaid>/<Credit-Card-Fraud-Dtection-system>.git
cd <your-repository-name>
```

### 2. Set Up a Python 3.10 Virtual Environment
Creating an isolated environment ensures consistent dependency versions:

**On Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
Ensure your environment is active, then install the exact package specifications:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Launch the local web server:
```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 📖 How to Use the Application

1. **Batch Upload:** Navigate to the **Home** tab, drag and drop a transaction CSV file (such as `transaction_data.csv`), and view generated predictions with risk probability flags.
2. **Single Transaction Check:** Navigate to the **Manual Input** tab, fill in the transaction attributes, and click **Predict** to receive an immediate verdict.
3. **Data Insights:** Visit the **Visualizations** tab to inspect interactive distributions and identify anomaly patterns across fraud vs. non-fraud transactions.
4. **Feature Exploration:** Use the **Feature Selection** view to understand feature importance and customize inference inputs.

---

## 👤 Author & Contact

- **Developer:** [Khan Abdullah Junaid]
- **GitHub:** [https://github.com/khan-abdullah-junaid]
- **LinkedIn:** [www.linkedin.com/in/khan-abdullah-junaid-29a567439]
- **Email:** [khan.abdullah.12055@gmail.com]
