import pandas as pd
import joblib


# ===================================
# LOAD MODEL
# ===================================

model = joblib.load(
    "model/decision_tree_model.pkl"
)

preprocessor = joblib.load(
    "model/preprocessor.pkl"
)


# ===================================
# USER INPUT
# ===================================

print("===================================")
print("ENTER DATA FOR PREDICTION")
print("===================================")

year = int(input("Enter Year: "))

industry_aggregation = input(
    "Enter Industry Aggregation: "
)

industry_code = input(
    "Enter Industry Code: "
)

industry_name = input(
    "Enter Industry Name: "
)

units = input(
    "Enter Units: "
)

variable_code = input(
    "Enter Variable Code: "
)

variable_name = input(
    "Enter Variable Name: "
)

variable_category = input(
    "Enter Variable Category: "
)


# ===================================
# CREATE DATAFRAME
# ===================================

new_data = pd.DataFrame([
    {
        "Year": year,
        "Industry_aggregation_NZSIOC":
            industry_aggregation,
        "Industry_code_NZSIOC":
            industry_code,
        "Industry_name_NZSIOC":
            industry_name,
        "Units":
            units,
        "Variable_code":
            variable_code,
        "Variable_name":
            variable_name,
        "Variable_category":
            variable_category
    }
])


# ===================================
# PREPROCESS
# ===================================

new_data_processed = preprocessor.transform(
    new_data
)


# ===================================
# PREDICT
# ===================================

prediction = model.predict(
    new_data_processed
)


# ===================================
# RESULT
# ===================================

print("\n===================================")
print("PREDICTION RESULT")
print("===================================")

print("Predicted Value:", prediction[0])

print("===================================")