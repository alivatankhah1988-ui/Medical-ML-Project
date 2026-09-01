
import pandas as pd
import matplotlib.pyplot as plt


print("*************************")
#load dataset
df= pd.read_csv("Diabet_dataset.csv", sep=";")
print(df.head())

#show first 5 rows
print("First 5 rows")
print(df.head())

#Dataset shape
print("\nDataset shape")
print(df.shape)

#column names
print("\nColumn names:")
print(df.columns)

#Data type
print("\nData types:")
print(df.dtypes)

#check missing values
print("\nMissing values:")
print(df.isnull().sum())

#check duplicate rows
print("****DUPLICATED*****")
print("\nDuplicate rows:")
print(df.duplicated().sum())

#show duplicate rows
print('////////////////////////')

print("\nDuplicate rows:  ")
print(df[df.duplicated(keep=False)])

#show duplicate rows without the empty columns
print("///////////////////////////////")
print("\nDuplicate records:")
print(df[df.duplicated(subset=['Age', 'Gender', 'FBS', 'HbA1c', 'TG', 
'Chol', 'Result'], keep=False)]
[['Age', 'Gender', 'FBS', 'HbA1c', 'TG', 'Chol', 'Result']])

#Remove duplicate rows
df= df.drop_duplicates()
print("\nDataset shape after removing duplicates:")
print(df.shape)

#Remove completely empty columns
df= df.dropna(axis=1, how='all')

print("\nDataset shape after removing empty columns:")
print(df.shape)

#check target variable distribution
print("\nResult distribution:")
print(df["Result"].value_counts())

print("\nResult distribution(%):")
print(df["Result"].value_counts(normalize=True)*100)

#Statistical summary of numerical variables
print("\nStatistical summary:")
print(df.describe())

#check extreme TG values
print("\nHighest TG values:")
print(df["TG"].sort_values(ascending=False).head(10))

#Inspect the extreme TG record
print("\nRecord with the highest TG:")
print(df.loc[df["TG"].idxmax()])

#we want to check outlier , we have a record that is 10786 which is
#related to TG , then we want to use a method that called IQR(Interquartile Range)

#Calculate IQR for TG
Q1= df["TG"].quantile(0.25)
Q3= df["TG"].quantile(0.75)

IQR= Q3-Q1
print("\nTG IQR analysis:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

#Identify TG outliers using IQR
lower_bound= Q1-1.5*IQR
upper_bound= Q3+1.5*IQR

tg_outliers= df[(df["TG"]<lower_bound) | (df["TG"]>upper_bound)]
print("\nTG outliers:")
print(tg_outliers[["Age","Gender", "FBS", "HbA1c", "TG", "Chol", "Result"]])

print("\nNumber of TG outliers:")
print(len(tg_outliers))


print("*********////////////*********")
pd.set_option("display.max_columns",None)
print(df.groupby("Result")[["Age", "FBS", "HbA1c", "TG","Chol"]].mean())

print("***matplotlib****")
df.boxplot(column="HbA1c", by="Result")
plt.title("HbA1c by Diabetes Result")
plt.suptitle("")
plt.xlabel("Result")
plt.ylabel("HbA1c")
plt.show()


print(df.groupby("Result")["HbA1c"].quantile(0.25))
print(df.groupby("Result")["HbA1c"].quantile(0.50))
print(df.groupby("Result")["HbA1c"].quantile(0.75))


#compare HbA1c between YES and NO groups

print("\nHbA1c statistcs by Result:  ")
print(df.groupby("Result")['HbA1c'].describe())

for group in ["NO", "YES"]:
    data = df[df["Result"] == group]["HbA1c"]

    print(f"\nGroup: {group}")
    print(f"Median: {data.median()}")
    print(f"Q1: {data.quantile(0.25)}")
    print(f"Q3: {data.quantile(0.75)}")
    print(f"IQR: {data.quantile(0.75) - data.quantile(0.25)}")

#Compare FBS between diabetic and non-diabetic groups
# don't need this code now
#fbs_stats= df.groupby("Result")["FBS"].agg(["count", "mean", "median", "std", "min", "max"])
#print("\nFBS statistics by Result: ")
#print(fbs_stats)

print("\nRecord with TG= 10786: ")
print(df[df["TG"]==10786])

print("\nHighest TG Values: ")
print(df["TG"].sort_values(ascending=False).head(10))

print("********  Test / Train  show *******")

from sklearn.model_selection import train_test_split
X = df.drop("Result", axis=1)
y = df["Result"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True))

# we can see the percent of YES and NO with normalize parameter
print("\nTest class distribution:")
print(y_test.value_counts(normalize=True))
