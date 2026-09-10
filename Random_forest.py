import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix)

#=======================
# 1. Load Data
#=======================
data= pd.read_csv("Diabet_dataset.csv", sep=";")

#Remove Completely empty columns
data=data.dropna(axis=1, how="all")

print("Dataset shape:  ", data.shape)
print(data.head())

#=======================
# 2. Define X and y
#=======================

X= data.drop ("Result", axis=1)
y= data ["Result"]

#=======================
# 3. Define columns
#=======================
categorical_features= ["Gender"]
numerical_features= ["Age", "FBS", "HbA1c", "TG", "Chol"]

#=======================
# 4. Preprocessing
#=======================
preprocessor= ColumnTransformer(
    transformers=[
        ("cat",
        OneHotEncoder(handle_unknown="ignore"), 
                      categorical_features)],
                      remainder="passthrough")

#=======================
# 5. Create Random forest model
#=======================

model= Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ]
)


#=======================
# 6. Train / Test Split
#=======================

X_train, X_test, y_train, y_test= train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
print("\nTraining set: ", X_train.shape)
print("Testing set:    ", X_test.shape)


#=======================
# 7. Train the model
#=======================
model.fit(X_train, y_train)

#=======================
# 8. Prediction
#=======================
y_pred= model.predict(X_test)

#=======================
# 9. Evaluation
#=======================
accuracy= accuracy_score(y_test, y_pred)
precision= precision_score(y_test, y_pred, pos_label="YES")
recall= recall_score(y_test, y_pred, pos_label="YES")
f1= f1_score(y_test, y_pred, pos_label="YES")
cm= confusion_matrix(y_test, y_pred, labels=["NO", "YES"])

print ("Random Forest Results")
print ("-------------------")

print("Accuracy:   ", round(accuracy*100,2), "%")
print("Precision:  ", round(precision*100,2), "%")
print("Recall:     ", round(recall*100,2), "%")
print("F1_score:   ", round(f1*100,2), "%")

print ("\nConfusion Matrix : ")
print (cm)
