import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

#load database
df= pd.read_csv("Diabet_dataset.csv", delimiter=";")

print ("\n HbA1c by Result:  ")
print (df.groupby("Result")["HbA1c"].describe())

#Remove Completely empty columns
df=df.dropna(axis=1, how="all")

# Seprate features and target
X= df.drop("Result", axis=1)
y= df["Result"]

# Convert categorical variable to numerical values
X= pd.get_dummies(X, drop_first=True)

# Check missing values
print("\nMissing values: ")
print(X.isna().sum())

print("\nInfinite values: ")
print((X== float("inf")).sum())
print((X== float("-inf")).sum())


# Train/ Test split
X_train, X_test, y_train, y_test= train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# create Decision Tree model
model= DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

#Predictions on training data
train_pred= model.predict(X_train)

#Training accuracy
train_accuracy= accuracy_score(y_train, train_pred)


# Make predictions
y_pred= model.predict(X_test)

# Evaluate the model
accuracy= accuracy_score(y_test, y_pred)

#Test accuracy
print("Training Accuracy:  ", train_accuracy)
print ("Test Accuracy:  ", accuracy)

precision= precision_score (y_test, y_pred, pos_label="YES")
recall= recall_score(y_test, y_pred, pos_label="YES")
f1= f1_score(y_test, y_pred, pos_label="YES")

cm= confusion_matrix(y_test, y_pred)

print("Desicion Tree Results")
print("---------------------")

print("Accuracy: ", accuracy)
print("precision: ", precision)
print("recall: ", recall)
print("f1: ", f1)

print("\nConfusion Matrix:  ")
print(cm)

print ("Tree depth: ",
model.tree_.max_depth)
print("Number of levels: ", 
model.tree_.n_leaves)

#cross_validation
cv_scores= cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring= "accuracy"
)

print ("\nCross-Validation Results  ")
print ("----------------------------")
print ("scores:", cv_scores)
print ("Mean Accuracy:", cv_scores.mean( ))
print ("Standard Deviation:  ", cv_scores.std( ))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#Visualize Decision Tree
plt.figure (figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names= model.classes_ ,
    filled= True
)

#plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")

plt.show()


depths= range(1,11)

train_accuracies= []
test_accuracies= []

for depth in depths:
    model= DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_accuracy= model.score(X_train, y_train)
    test_accuracy= model.score(X_test, y_test)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print (
      f"Depth= {depth} | "
      f"Train Accuracy= {train_accuracy:.4f}|"
      f"Test Accuracy= {test_accuracy:.4f}"
      )

