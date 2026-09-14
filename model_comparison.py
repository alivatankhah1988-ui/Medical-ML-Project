import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==============================
# 1. Load Dataset
# ==============================

data = pd.read_csv("Diabet_dataset.csv", sep=";")


# ==============================
# 2. Define Features and Target
# ==============================

X = data[["Age", "Gender", "FBS", "HbA1c", "TG", "Chol"]]
y = data["Result"]


# ==============================
# 3. Define Features
# ==============================

categorical_features = ["Gender"]

numerical_features = [
    "Age",
    "FBS",
    "HbA1c",
    "TG",
    "Chol"
]


# ==============================
# 4. Train/Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ==============================
# 5. Create Preprocessors
# ==============================

preprocessor_no_scaling = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


preprocessor_scaling = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            StandardScaler(),
            numerical_features
        )
    ]
)


# ==============================
# 6. Define Models
# ==============================

models = {

    "Logistic Regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor_scaling),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    ),

    "KNN": Pipeline(
        steps=[
            ("preprocessor", preprocessor_scaling),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]
    ),

    "Naive Bayes": Pipeline(
        steps=[
            ("preprocessor", preprocessor_scaling),
            ("classifier", GaussianNB())
        ]
    ),

    "Decision Tree": Pipeline(
        steps=[
            ("preprocessor", preprocessor_no_scaling),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=5,
                    random_state=42
                )
            )
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor_no_scaling),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42
                )
            )
        ]
    )
}


# ==============================
# 7. Model Evaluation
# ==============================

results = []


for model_name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Test prediction
    y_pred = model.predict(X_test)

    # Test metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="YES"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="YES"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="YES"
    )

    # 5-Fold Cross-Validation
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Store results
    results.append({
        "Model": model_name,
        "Test Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "CV Mean": cv_mean,
        "CV Std": cv_std
    })


# ==============================
# 8. Create Comparison Table
# ==============================

results_df = pd.DataFrame(results)


# ==============================
# 9. Convert Metrics to Percentage
# ==============================

percentage_columns = [
    "Test Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "CV Mean",
    "CV Std"
]

results_df[percentage_columns] = (
    results_df[percentage_columns] * 100
)


# ==============================
# 10. Sort by CV Mean
# ==============================

results_df = results_df.sort_values(
    by="CV Mean",
    ascending=False
).reset_index(drop=True)


# ==============================
# 11. Display Final Comparison
# ==============================

print("\n==============================================")
print("        FINAL MODEL COMPARISON")
print("==============================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Test Accuracy": "{:.2f}%".format,
            "Precision": "{:.2f}%".format,
            "Recall": "{:.2f}%".format,
            "F1-score": "{:.2f}%".format,
            "CV Mean": "{:.2f}%".format,
            "CV Std": "{:.2f}%".format
        }
    )
)


# ==============================
# 12. Best Model
# ==============================

best_model = results_df.iloc[0]

print("\n==============================================")
print("              BEST MODEL")
print("==============================================")

print("Model:", best_model["Model"])
print(f"CV Mean: {best_model['CV Mean']:.2f}%")
print(f"CV Std: {best_model['CV Std']:.2f}%")

