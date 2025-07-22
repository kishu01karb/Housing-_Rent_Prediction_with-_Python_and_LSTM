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
- Number of Bathrooms

### 🔧 Model Details:
- Built using **Keras Sequential API** with **LSTM layers**.
- Compiled with:
  - **Optimizer**: `adam`
  - **Loss**: `mae` (mean absolute error)
- Trained on 90% of the dataset using `train_test_split`.

---

## 🧪 User Input & Prediction

After training the model, the script allows the user to enter custom house details to predict rent.

### 🔢 Sample Input:
```text
Enter house details to predict rent:
Number of BHK: 2
Size of the House (sq.ft): 900
Area Type (Super Area = 1, Carpet Area = 2, Built Area = 3): 2
Pin Code of the City: 110001
Furnishing Status (Unfurnished = 0, Semi-Furnished = 1, Furnished = 2): 1
Tenant Type (Bachelors = 1, Bachelors/Family = 2, Only Family = 3): 3
Number of Bathrooms: 2

Predicted House Rent = ₹23,898.38


👉 The model outputs a **predicted rent value** based on the trained LSTM network.

---

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
