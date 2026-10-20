# Protein Expression Yield Prediction Using Machine Learning

## Project Overview
This project predicts recombinant protein expression yield (mg/L) using experimental conditions and machine learning. It compares Linear Regression, Random Forest, SVR, and XGBoost to identify the best-performing model based on Mean Absolute Error (MAE).

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Joblib

## Input Features
- Media
- Optical Density (OD)
- IPTG concentration (mM)
- Temperature (°C)
- Expression time (hours)

## Workflow
1. Data cleaning and preprocessing
2. Handling missing values and encoding categorical features
3. Feature scaling and train-test splitting
4. Training and comparing machine learning models
5. Saving the selected model using Joblib
6. Predicting protein expression yield for new experimental conditions

## Best Model
**Linear Regression** was selected based on the lowest MAE among the four evaluated models.

- MAE: 123.710 mg/L
- RMSE: 221.684 mg/L
- R² Score: 0.169

The results indicate limited predictive performance, and further model improvement is required.

## How to Run
Run the prediction script from the project directory:

python predict_yield.py

Ensure the saved model is available at `Model/Protein_Expression_yield_model.pkl`.
