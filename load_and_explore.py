import pandas as pd
df = pd.read_csv("student_placement_history.csv")
print(df.head())
print(df.shape)
print(df["Placed"].value_counts())
