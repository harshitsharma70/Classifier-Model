import pandas as pd
df = pd.read_csv("Student_placement_history.csv")
X = df.drop(columns = ["StudentID","Placed"])
Y = df["Placed"]

print(X.shape)
print(Y.shape)
print(list[any](X.columns))
