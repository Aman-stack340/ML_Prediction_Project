# ML Prediction Project

## Project Overview

This project builds machine learning models to predict the `Value` of a given record using industry and financial-related features.

The project compares three regression algorithms:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The models are trained and evaluated using a real-world dataset.

---

## Objective

The main objective of this project is to:

1. Load and clean the dataset.
2. Select relevant features.
3. Convert categorical data into numerical form.
4. Split the dataset into training and testing sets.
5. Train multiple machine learning models.
6. Evaluate model performance.
7. Visualize prediction performance.
8. Save the trained model.
9. Predict values for new input data.

---

## Dataset

The dataset contains 60,255 records and 10 columns.

Important columns include:

- Year
- Industry aggregation
- Industry code
- Industry name
- Units
- Variable code
- Variable name
- Variable category
- Value
- Industry code ANZSIC06

The target variable is:

`Value`

Non-numeric values in the target column were removed before training.

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Dataset loaded using Pandas.
2. `Value` converted to numeric format.
3. Rows with missing target values removed.
4. Identifier column excluded from the features.
5. Categorical columns identified.
6. One-hot encoding applied to categorical features.
7. Dataset divided into training and testing sets.

The train-test split used was:

- Training: 80%
- Testing: 20%
- Random state: 42

---

## Machine Learning Models

### 1. Linear Regression

Linear Regression was used as a baseline regression model.

### 2. Decision Tree Regressor

Decision Tree Regression was used to model nonlinear relationships between the input features and the target variable.

### 3. Random Forest Regressor

Random Forest Regression was used as an ensemble learning approach consisting of multiple decision trees.

---

## Model Evaluation

Because the target variable is continuous, this project uses regression metrics.

The metrics used are:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 34660.56 | 260703.12 | 0.1988 |
| Decision Tree | 7929.17 | 85701.52 | 0.9134 |
| Random Forest | 4817.13 | 109813.24 | 0.8578 |

The Decision Tree achieved the highest R² score on the test split, while Random Forest produced the lowest MAE.

---

## Visualization

The project includes visualizations for:

- Model R² comparison
- Model RMSE comparison
- Model MAE comparison
- Actual vs Predicted values

These visualizations help understand model performance beyond numerical metrics.

---

## Saved Models

The trained Decision Tree model is saved as:

```text
model/decision_tree_model.pkl