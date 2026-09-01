import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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


