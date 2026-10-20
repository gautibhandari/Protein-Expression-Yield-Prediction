import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df= pd.read_csv(r"C:\Users\ADMIN\Desktop\Protein_Expression_Yield_Prediction\Data\protein_expression.csv")
df.head(3)

target = "Maximum_expression_yield_mg_L"

# Convert simple numeric values; ranges become NaN for now
df["Yield_clean"] = pd.to_numeric(df[target], errors="coerce")

print("Valid numeric yields:", df["Yield_clean"].notna().sum())
print("Values requiring review:", df["Yield_clean"].isna().sum())

df=df.dropna(subset=["Yield_clean"])
print("Values requiring review:", df["Yield_clean"].isna().sum())

features =df [["Media","OD_au","IPTG_mM", "Temperature_C", "Time_h","Yield_clean" ]]
features.head(3)

# Convert numerical columns to numeric data types
num_features = ["OD_au", "IPTG_mM", "Temperature_C", "Time_h"]

for col in num_features:
    features[col] = pd.to_numeric(features[col], errors="coerce")

cat_features = ["Media"]

# Preprocessing

preprocessor = ColumnTransformer([
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown ="ignore"))
]), cat_features),

    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy ="median")),
        ("scaler", StandardScaler())
        ]), num_features)

    ])

# Sepearte feature and targetr
X= features.drop(columns ='Yield_clean')
y= features['Yield_clean']


X_train, X_test, y_train, y_test =train_test_split(X,y,test_size = 0.2,random_state=42)

# Define models
models = {"Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200,random_state=42),
    "SVR": SVR(kernel="rbf"),
    "XGBoost": XGBRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,random_state=42)

}

results = []
trained_models = {}

# Train and evaluate each model
for name, regressor in models.items():
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    # Train each model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # storing the result
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    print(f"\n{name}")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

print("====Final Model Comparison====")
Performance_result = pd.DataFrame(results)
Performance_result = Performance_result.sort_values(by ="MAE", ascending= True)

# Select the best model
best_model_name = Performance_result.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print("Best model based on MAE:", best_model_name)

import os
# Create the folder if it doesn't exist
os.makedirs("Model_comparison", exist_ok=True)

# Save comparison result in csv
Performance_result.to_csv("Model_comparison/Comparison Result of Models", index=False)

# Save model
import joblib
import os

# Create the folder if it doesn't exist
os.makedirs("Model", exist_ok=True)


joblib.dump(best_model, "Model/Protein_Expression_yield_model.pkl")
print("Model Save Successfully")

# this code run in terminal 

# Load the saved model
model = joblib.load(r"C:\Users\ADMIN\Desktop\Protein_Expression_Yield_Prediction\Notebooks\Model\Protein_Expression_yield_model.pkl")


# Enter new experimental conditions
media = input("Enter Media (e.g., M9 or LB): ")
od = float(input("Enter OD_au: "))
iptg = float(input("Enter IPTG_mM: "))
temperature = float(input("Enter Temperature_C: "))
time = float(input("Enter Time_h: "))

# Create input DataFrame
new_data = pd.DataFrame([{
    "Media": media,
    "OD_au": od,
    "IPTG_mM": iptg,
    "Temperature_C": temperature,
    "Time_h": time
}])

# Predict protein expression yield
predicted_yield = model.predict(new_data)

print("\n===== Prediction Result =====")
print(f"Predicted Expression Yield: {predicted_yield[0]:.2f} mg/L")
