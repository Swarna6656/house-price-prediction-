# house-price-prediction-
# https://house-price-prediction-d4f5.onrender.com


---

# 🏠 House Price Prediction using Multiple Linear Regression (MLR)

This project implements a **House Price Prediction system** using **Multiple Linear Regression (MLR)** with an Object-Oriented Programming (OOP) approach in Python. It includes data preprocessing, model training, evaluation, and model persistence.

---

## 📌 Project Overview

The goal of this project is to predict house prices based on various features such as:

* Number of bedrooms & bathrooms
* Square footage (living area & lot)
* Floors
* Waterfront & view
* Condition
* Year built & renovated
* City & country

The model is trained using **Linear Regression**, and evaluation is done using performance metrics like **R² Score** and **RMSE**.

---

## ⚙️ Technologies Used

* Python 🐍
* NumPy
* Pandas
* Scikit-learn
* Pickle (for model saving)

---

## 📂 Project Structure

```
├── data.csv              # Dataset file
├── model.pkl             # Saved trained model
├── main.py               # Main Python script (MLR class)
└── README.md             # Project documentation
```

---

## 🚀 Features

* ✅ Data preprocessing (dropping unnecessary columns)
* ✅ Manual encoding of categorical variables
* ✅ Train-test split
* ✅ Model training using Linear Regression
* ✅ Model evaluation (R² Score & RMSE)
* ✅ Model saving & loading using Pickle
* ✅ Test prediction with sample input

---

## 🧠 Model Workflow

1. Load dataset
2. Clean & preprocess data
3. Encode categorical features (city, country)
4. Split data into training & testing sets
5. Train Linear Regression model
6. Evaluate performance
7. Save model as `.pkl` file
8. Load model and test prediction

---

## 📊 Evaluation Metrics

* **R² Score** → Measures model accuracy
* **RMSE (Root Mean Squared Error)** → Measures prediction error

---

## ▶️ How to Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/house-price-mlr.git
cd house-price-mlr
```

### Step 2: Install Dependencies

```bash
pip install numpy pandas scikit-learn
```

### Step 3: Run the Script

```bash
python main.py
```

---

## 💾 Model Saving

The trained model is saved as:

```
Model.pkl
```

You can load it later using:

```python
import pickle

with open("Model.pkl", "rb") as f:
    model = pickle.load(f)
```

---

## 🔮 Sample Prediction

```python
model.predict([[3,1.5,1340,7912,1.5,0,0,3,1340,0,1995,2005,0,0]])
```

---

## ⚠️ Limitations

* Manual encoding of categorical data (not scalable)
* No feature scaling applied
* Only Linear Regression used (no hyperparameter tuning)

---

## 📈 Future Improvements

* 🔹 Use `pd.get_dummies()` for better encoding
* 🔹 Apply feature scaling (StandardScaler)
* 🔹 Implement Ridge & Lasso Regression
* 🔹 Hyperparameter tuning using GridSearchCV
* 🔹 Build a web app using Flask

---

## 🙌 Author

**Your Name**

* GitHub: https://github.com/Swarna6656/house-price-prediction-.git

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---

