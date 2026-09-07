import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Read Dataset
df = pd.read_csv("Diabet_dataset.csv", sep=";")

# Features and Target
X = df[["Age", "Gender", "FBS", "HbA1c", "TG", "Chol"]]
y = df["Result"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


print ("*******************/////************")
print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTest target distribution:")
print(y_test.value_counts())

print("\nFirst 10 test indices:")
print(X_test.index[:10].tolist())

print ("-----------------")

# Define numerical and categorical features
numeric_features = ["Age", "FBS", "HbA1c", "TG", "Chol"]
categorical_features = ["Gender"]

# Encode the categorical feature
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# Create Logistic Regression pipeline
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Create Decision Tree pipeline
tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ))
    ]
)

# Train both models
logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)


# ==============================
# Decision Tree Feature Importance
# ==============================

tree = tree_model.named_steps["classifier"]

feature_names = tree_model.named_steps["preprocessor"].get_feature_names_out()

importance = tree.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nDecision Tree Feature Importance:")
print(feature_importance)


# Make predictions on the test set
y_pred_logistic = logistic_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)


# Calculate evaluation metrics for both models

logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_precision = precision_score(y_test, y_pred_logistic, pos_label="YES")
logistic_recall = recall_score(y_test, y_pred_logistic, pos_label="YES")
logistic_f1 = f1_score(y_test, y_pred_logistic, pos_label="YES")

tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_precision = precision_score(y_test, y_pred_tree, pos_label="YES")
tree_recall = recall_score(y_test, y_pred_tree, pos_label="YES")
tree_f1 = f1_score(y_test, y_pred_tree, pos_label="YES")


# Print results

print("\n--- Logistic Regression ---")
print(f"Accuracy:  {logistic_accuracy:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"Recall:    {logistic_recall:.4f}")
print(f"F1-score:  {logistic_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))


print("\n--- Decision Tree ---")
print(f"Accuracy:  {tree_accuracy:.4f}")
print(f"Precision: {tree_precision:.4f}")
print(f"Recall:    {tree_recall:.4f}")
print(f"F1-score:  {tree_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

