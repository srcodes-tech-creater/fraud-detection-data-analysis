# 🛡️ FraudGuard AI - Fraud Detection Data Analysis

A Machine Learning and Data Science web application for analyzing financial transaction data and predicting potentially fraudulent transactions.

This project was developed as part of a **Data Science Internship** by **S. R. Divya Dharshini**, B.Tech Artificial Intelligence and Data Science student.

---

## 🌐 Live Application

The application will be deployed using **Streamlit Community Cloud**.

Live Link: Coming Soon

---

## 📌 Project Overview

Fraud detection is an important problem in the financial industry. This project uses data analysis and Machine Learning techniques to identify suspicious financial transactions.

The application provides:

- Transaction data analysis
- Fraud pattern visualization
- Machine Learning model evaluation
- Single transaction fraud prediction
- Batch fraud prediction using CSV files
- Fraud probability calculation
- Fraud risk classification
- Downloadable prediction reports

---

## 🚀 Features

### 📊 Data Analysis

The application allows users to explore the fraud transaction dataset and understand:

- Dataset dimensions
- Missing values
- Data types
- Summary statistics
- Fraud distribution
- Transaction patterns

### 📈 Interactive Visualizations

Interactive charts help analyze fraud patterns and relationships between transaction features.

Features include:

- Fraud vs legitimate transaction distribution
- Feature distribution analysis
- Interactive histograms
- Scatter plots
- Feature relationship analysis

### 🤖 Machine Learning Model

The application uses a **Random Forest Classifier** to identify potentially fraudulent transactions.

The model training process includes:

- Data preprocessing
- Missing value handling
- Feature selection
- Train-test split
- SMOTE oversampling for class imbalance
- Random Forest classification
- Fraud probability prediction
- Threshold optimization

### 📊 Model Performance

The application evaluates the Machine Learning model using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- Feature Importance

### 🔍 Single Transaction Fraud Prediction

Users can enter transaction information manually.

The system provides:

- Fraud probability
- Legitimate probability
- Fraud prediction
- Low / Medium / High risk classification
- Prediction report download

### 📁 Batch Fraud Prediction

Users can upload a CSV file containing multiple transactions.

The system:

1. Validates the required model features
2. Processes the uploaded dataset
3. Predicts fraud for all transactions
4. Calculates fraud probability
5. Displays fraud statistics
6. Allows downloading prediction results

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data analysis and processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Imbalanced-learn | Handling class imbalance |
| Random Forest | Fraud classification |
| Streamlit | Web application |
| Plotly | Interactive visualizations |
| Matplotlib | Data visualization |
| Joblib | Model storage and loading |

---

## 📂 Project Structure

```text
fraud-detection-data-analysis/
│
├── app.py
├── README.md
├── requirements.txt
│
├── assets/
│   └── style.css
│
├── data/
│   ├── raw/
│   │   └── Dataset files
│   │
│   └── processed/
│       └── merged_fraud_data.csv
│
├── models/
│   └── fraud_model.pkl
│
├── pages/
│   ├── 1_Data_Analysis.py
│   ├── 2_Visualizations.py
│   ├── 3_Model_Performance.py
│   ├── 4_Fraud_Prediction.py
│   └── 5_Batch_Prediction.py
│
└── src/
    ├── data_preprocessing.py
    ├── model_training.py
    └── ui.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fraud-detection-data-analysis.git
```

### 2. Open the project folder

```bash
cd fraud-detection-data-analysis
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install required libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

First preprocess the dataset:

```bash
python src/data_preprocessing.py
```

Train the Machine Learning model:

```bash
python src/model_training.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This project can be deployed using Streamlit Community Cloud.

Deployment process:

1. Upload the project to GitHub.
2. Connect the GitHub repository to Streamlit Community Cloud.
3. Select the repository.
4. Select `app.py` as the main file.
5. Click Deploy.

---

## 📊 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
Data Cleaning
     ↓
Feature Selection
     ↓
Train-Test Split
     ↓
SMOTE
     ↓
Random Forest Model
     ↓
Threshold Optimization
     ↓
Fraud Prediction
     ↓
Interactive Streamlit Dashboard
```

---

## ⚠️ Model Limitation

Fraud datasets are often highly imbalanced.

This means fraudulent transactions represent a small percentage of the total dataset.

To address this issue, the project uses:

- SMOTE oversampling
- Balanced training
- Threshold optimization
- Precision, Recall and F1 Score evaluation

Model performance depends on the quality, size and characteristics of the dataset.

---

## 🔮 Future Improvements

Future versions of the project may include:

- XGBoost fraud detection
- Real-time transaction monitoring
- Fraud alert notifications
- Database integration
- User authentication
- API deployment
- Explainable AI using SHAP
- Advanced anomaly detection

---

## 👩‍💻 Author

**S. R. Divya Dharshini**

B.Tech - Artificial Intelligence and Data Science  
Asan Memorial College of Engineering and Technology  

Data Science | Machine Learning | Data Analytics | Python

---

## ⭐ Acknowledgement

This project was developed for educational and internship purposes to demonstrate skills in:

- Data Analysis
- Data Preprocessing
- Machine Learning
- Fraud Detection
- Model Evaluation
- Streamlit Application Development
- Data Visualization

---

⭐ If you found this project useful, consider giving the repository a star!