
# 🏠 Housing Rent Prediction with Python and LSTM

This project explores a housing rent dataset to analyze rent trends, visualize key insights, and build a machine learning model (LSTM) to predict rent based on various property features.

---

## 📊 Project Overview

### 🔍 Part 1: Data Analysis and Visualization

The dataset is analyzed using Python libraries to understand patterns and relationships in rent pricing. Key actions performed:

- **Imported Libraries**:
  - `pandas`, `numpy`: For data handling and computation.
  - `matplotlib`, `plotly`: For visualization (bar and pie charts).
  
- **Dataset Loading & Preprocessing**:
  - Loaded `House_Rent_Dataset.csv`.
  - Checked for missing values and generated descriptive statistics.
  - Encoded categorical features numerically for model training.

- **Data Insights**:
  - Calculated **mean**, **median**, **max**, and **min** rents.
  - Visualized:
    - Rent distribution by **city**, **BHK**, **furnishing status**, and **area type**.
    - Number of available houses and tenant preferences (via pie charts).

---

## 🤖 Part 2: Machine Learning Model – LSTM

### 🧠 Goal:
Predict house rent based on the following features:
- BHK
- Size (in sq. ft.)
- Area Type
- City (by PIN code)
- Furnishing Status
- Tenant Preference
- Bathrooms

### 🔧 Model Details:
- Built using **Keras Sequential API** with **LSTM layers**.
- Compiled with:
  - **Optimizer**: `adam`
  - **Loss**: `mae` (mean absolute error)
- Trained on 90% of the dataset (`train_test_split` used).

---

## 🧪 User Input & Prediction

After training the model, the script allows the user to enter custom house details:

👉 The model outputs a **predicted rent value** based on the trained LSTM network.

---

enter house details to predict rent
Number of BHK: 2
Size of the House: 900
Area Type (Super Area = 1, Carpet Area = 2, Built Area = 3): 2
Pin Code of the City: 110001
Furnishing Status of the House (Unfurnished = 0, Semi-Furnished = 1, Furnished = 2): 1
Tenant Type (Bachelors = 1, Bachelors/Family = 2, Only Family = 3): 3
Number of bathrooms: 2
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 158ms/step
Predicted House Price =  [[23898.379]]

## 📁 Files Included

- `housing_rent_prediction.py` – Full source code for loading data, analysis, visualization, and prediction.
- `House_Rent_Dataset.csv` – Input dataset (required for execution).
- `README.md` – Project overview and usage instructions.

---

## ▶️ How to Run

1. Clone or download this repository.
2. Install required libraries:

```bash
pip install pandas numpy matplotlib plotly sklearn keras tensorflow

## 📌 Key Learnings
Exploratory Data Analysis (EDA) with real-world housing data.

Creating interactive visualizations using Plotly.

Encoding categorical data for machine learning.

Building LSTM models for regression tasks.
