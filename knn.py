import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#=======================
# 1. Load dataset
#=======================

data=pd.read_csv("Diabet_dataset.csv",sep=";")

#=======================
# 2. Define X and y
#=======================
X= data.drop("Result", axis=1)
y= data["Result"]

#=======================
# 3. Define Columns
#=======================
categorical_features= ["Gender"]
numeric_features= ["Age", "FBS", "HbA1c", "TG", "Chol"]

#=======================
# 4. Preprocessing
#=======================
preprocessor= ColumnTransformer(
    transformers=[("cat", 
                   OneHotEncoder(handle_unknown= "ignore"), 
                   categorical_features),
                  ("num", StandardScaler(),
                   numeric_features)
                   ])

#=======================
# 5. Create KNN pipeline
#=======================

knn_model= Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))]
    )

#=======================
# 6. Train/Test Split
#=======================

X_train, X_test, y_train, y_test= train_test_split(
    X, 
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
    )

#=======================
# 7. Train the model
#=======================
knn_model.fit(X_train,y_train)


#=======================
# 8. Prediction
#=======================
y_pred= knn_model.predict(X_test)

#=======================
# 9. Evaluation
#=======================
accuracy= accuracy_score(y_test, y_pred)
Precision= precision_score(y_test, y_pred, pos_label="YES")
recall= recall_score(y_test, y_pred, pos_label="YES")
f1= f1_score(y_test, y_pred, pos_label="YES")
cm= confusion_matrix(y_test,y_pred, labels=["NO", "YES"])


print("KNN Results")
print("-------------------")
print("Accuracy:   ", round(accuracy*100,2), "%")
print("Precision:  ", round(Precision*100,2), "%")
print("Recall:     ", round(recall*100,2), "%")
print("F1_score:   ", round(f1*100,2), "%")

print ("\nConfusion Matrix : ")
print (cm)

