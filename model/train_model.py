import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import numpy as np


# ===================================
# LOAD DATASET
# ===================================

df = pd.read_csv("data/dataset.csv")

print("===================================")
print("DATASET LOADED")
print("===================================")

print("Original shape:", df.shape)


# ===================================
# TARGET
# ===================================

target_column = "Value"

df[target_column] = pd.to_numeric(
    df[target_column],
    errors="coerce"
)

df = df.dropna(subset=[target_column])


# ===================================
# FEATURES
# ===================================

feature_columns = [
    "Year",
    "Industry_aggregation_NZSIOC",
    "Industry_code_NZSIOC",
    "Industry_name_NZSIOC",
    "Units",
    "Variable_code",
    "Variable_name",
    "Variable_category"
]

X = df[feature_columns]
y = df[target_column]


# ===================================
# IDENTIFY COLUMN TYPES
# ===================================

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\n===================================")
print("COLUMN TYPES")
print("===================================")

print("Numerical columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ===================================
# PREPROCESSING
# ===================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ===================================
# TRAIN TEST SPLIT
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===================================")
print("TRAIN TEST SPLIT")
print("===================================")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ===================================
# PREPROCESS DATA
# ===================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:",
      X_train_processed.shape)

print("Processed testing shape:",
      X_test_processed.shape)


# ===================================
# MODELS
# ===================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=20
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# ===================================
# TRAIN AND EVALUATE
# ===================================

print("\n===================================")
print("MODEL TRAINING")
print("===================================")

results = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train
    model.fit(X_train_processed, y_train)

    # Predict
    predictions = model.predict(X_test_processed)
    # Save predictions for visualization
    if name == "Decision Tree":
     decision_tree_predictions = predictions

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"{name} completed.")

    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R²  :", r2)


# ===================================
# MODEL COMPARISON
# ===================================

print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

results_df = pd.DataFrame(results).T

print(results_df)
import matplotlib.pyplot as plt


# ===================================
# MODEL PERFORMANCE VISUALIZATION
# ===================================

# R² comparison
plt.figure(figsize=(8, 5))

plt.bar(
    results_df.index,
    results_df["R2"]
)

plt.title("Model Comparison - R² Score")
plt.xlabel("Model")
plt.ylabel("R² Score")

plt.xticks(rotation=15)
plt.tight_layout()

plt.show()


# RMSE comparison
plt.figure(figsize=(8, 5))

plt.bar(
    results_df.index,
    results_df["RMSE"]
)

plt.title("Model Comparison - RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")

plt.xticks(rotation=15)
plt.tight_layout()

plt.show()


# MAE comparison
plt.figure(figsize=(8, 5))

plt.bar(
    results_df.index,
    results_df["MAE"]
)

plt.title("Model Comparison - MAE")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error")

plt.xticks(rotation=15)
plt.tight_layout()

plt.show()
# ===================================
# SAVE BEST MODEL
# ===================================

import joblib
import os

# Create model directory
os.makedirs("model", exist_ok=True)

# Save Decision Tree model
joblib.dump(
    models["Decision Tree"],
    "model/decision_tree_model.pkl"
)

# Save preprocessing pipeline
joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)

print("\n===================================")
print("MODEL SAVED")
print("===================================")

print("Decision Tree saved to:")
print("model/decision_tree_model.pkl")

print("\nPreprocessor saved to:")
print("model/preprocessor.pkl")
# ===================================
# ACTUAL VS PREDICTED
# ===================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    decision_tree_predictions,
    alpha=0.5
)

# Perfect prediction reference line
minimum = min(
    y_test.min(),
    decision_tree_predictions.min()
)

maximum = max(
    y_test.max(),
    decision_tree_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted Values - Decision Tree")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")

plt.tight_layout()
plt.show()
# ===================================
# FINAL MODEL COMPARISON GRAPH
# ===================================

plt.figure(figsize=(9, 6))

plt.bar(
    results_df.index,
    results_df["R2"]
)

plt.title("Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("R² Score")

plt.xticks(rotation=15)
plt.ylim(0, 1)

plt.tight_layout()

plt.show()