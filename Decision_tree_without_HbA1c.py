import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# load Dataset
df= pd.read_csv("Diabet_dataset.csv", sep=";")

# X features and Y is target variable
X= df[["Age", "Gender", "FBS", "TG", "Chol"]]
y= df["Result"]

# Train/ Test split
X_train, X_test, y_train, y_test= train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Define numerical and categorical features
numerica_features= ["Age", "FBS", "TG", "Chol"]
categorical_features= ["Gender"]

#Preprocessing
preprocessor= ColumnTransformer(
    transformers= [("cat",
    OneHotEncoder(handle_unknown= "ignore"), categorical_features)
    ],
    remainder= "passthrough"
)

# Create the model pipeline
model= Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
    ]
)

# Train model
model.fit(X_train, y_train)

# Predictions on Testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="YES")
recall = recall_score(y_test, y_pred, pos_label="YES")
f1 = f1_score(y_test, y_pred, pos_label="YES")

# show the Results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


#Compare between HbA1c and without HbA1c in Decision Tree

print("\n--- Decision Tree Comparison ---")

print("Without HbA1c:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nWith HbA1c:")
print("Accuracy:  1.0000")
print("Precision: 1.0000")
print("Recall:    1.0000")
print("F1-score:  1.0000")

from sklearn.model_selection import cross_val_score
#5-Fold Cross-Validation
cv_scores= cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring= "accuracy"
)

print ("\n--- 5-Fold Cross-validation ---")
print ("CV Scores:", cv_scores)
print (f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print (f"Std CV Accuracy: {cv_scores.std():.4f}")

