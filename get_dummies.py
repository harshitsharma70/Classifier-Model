import pandas as pd 
df = pd.read_csv("student_placement_history.csv")
X = df.drop(columns=["StudentID","Placed"])
X = pd.get_dummies(X)

print(X.shape)
print(list[any](X.columns))
