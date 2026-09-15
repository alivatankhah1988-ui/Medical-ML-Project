import pandas as pd
import matplotlib.pyplot as plt

#================
# Load dataset
#================

df = pd.read_csv("Diabet_dataset.csv", sep=";")

#Remove Completely empty columns 
data=df.dropna(axis=1, how="all")
print("Dataset shape:  ", df.shape)
print("Dataset shape:  ", data.shape)



print("Result values:")
print(df["Result"].unique())
print(df["Result"].value_counts())

print("\nGender values:")
print(df["Gender"].unique())
print(df["Gender"].value_counts())

#================
# Count target classes
#================
result_counts = data["Result"].value_counts()

#================
# Count genders
#================
gender_counts= data["Gender"].value_counts()


#================
# Plot Result distribution
#================

plt.figure(figsize=(6, 4))
result_counts.plot(kind="bar", color= ["#5B9BD5", "#ED7D31"])

plt.title("Distribution of Diabetes Result")
plt.xlabel("Result")
plt.ylabel("Number of Samples")

plt.tight_layout()

#================
# Add Gender counts below the chart
#================
gender_text=(f"Female: {gender_counts.get('female',0)}  |" 
             f"   Male: {gender_counts.get('male',0)}"
)

plt.figtext(
    0.5, 0.01,
    gender_text,
    ha='center',
    fontsize=11
)



#================
# Add Gender counts below the chart
#================
Result_text=(f"YES: {result_counts.get('YES',0)}  |"   
             f"  NO: {result_counts.get('NO',0)}"
)

plt.figtext(
    0.5, -0.05,
    Result_text,
    ha='center',
    fontsize=11
)

#================
# Save figure
#================

plt.savefig("result_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

