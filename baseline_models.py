import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_validate


#load dataset
df=pd.read_csv ("Diabet_dataset.csv", sep=";")

#Remove Completely empty columns
df=df.dropna(axis=1, how="all")


#Features and target
x= df.drop("Result", axis=1)
y= df["Result"]

#Train/Test Split
X_train, X_test, y_train, y_test =train_test_split(
    x,
    y,
    test_size=0.30, 
    random_state=42, 
    stratify=y
    )

print("Training set:   ", X_train.shape)
print("Test set:   ", X_test.shape)


# Categorical features
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
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

print("\nMissing Value:  ")
print(X_train.isnull().sum())

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred= model.predict(X_test)

print("\npredictions:  ")
print(y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

print ("\n Confusion Matrix:    ")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:  ",  
accuracy_score(y_test, y_pred))

print("precision: ", precision_score(y_test, y_pred, pos_label="YES"))
print("Recall:  ", recall_score(y_test, y_pred, pos_label="YES"))
print("F1_Score:   ", f1_score(y_test, y_pred, pos_label="YES"))

# Cross-Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}

cv_results = cross_validate(
    model,
    x,
    y,
    cv=cv,
    scoring=scoring
)

print("\nCross-Validation Results:")

for metric in scoring:
    scores = cv_results[f"test_{metric}"]
    print(
        f"{metric.capitalize()}: "
        f"{scores.mean():.4f} "
        f"+/- {scores.std():.4f}"
    )

# Check HbA1c relationship with Result

print ("\nHbA1c by Result: ")
print (
    df.groupby("Result")["HbA1c"].agg(["count", "mean","min","max"])
)

# HbA1c-only model
x_hba1c= df[["HbA1c"]]
y_hba1c= df[["Result"]]

X_train_h, X_test_h, y_train_h, y_test_h= train_test_split(
    x_hba1c,
    y_hba1c,
    test_size=0.30,
    random_state=42,
    stratify=y_hba1c
)

hba1c_model= LogisticRegression(max_iter=1000)
hba1c_model.fit(X_train_h, y_train_h)
y_pred_h= hba1c_model.predict(X_test_h)

print ("\nHbA1c-only Model:  ")
print ("Accuracy: ", accuracy_score(y_test_h,y_pred_h))
print ("Precision: ", precision_score(y_test_h, y_pred_h, pos_label="YES"))
print ("Recall: ", recall_score(y_test_h, y_pred_h, pos_label= "YES"))
print ("F1: ", f1_score(y_test_h, y_pred_h, pos_label="YES"))
print ("\Confusion Matrix:  ")
print (confusion_matrix(y_test_h, y_pred_h))


print ("\n HbA1c values near the boundary:  ")
print (
    df[
        (df["HbA1c"]>=5.0) &
        (df["HbA1c"]<=6.0)]
    [["HbA1c", "Result"]].sort_values("HbA1c")
)

print ("\n HbA1c vs Result:  ")
print (
    df.groupby(["HbA1c", "Result"]).size().unstack(fill_value=0)
)

print("\nFBS vs Result:")

print(
    df.groupby(["FBS", "Result"])
      .size()
      .unstack(fill_value=0)
)