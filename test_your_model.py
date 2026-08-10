import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("student_placement_history.csv")
X = pd.get_dummies(df.drop(columns=["StudentID","Placed"]))
y = df["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

row = pd.DataFrame([{c: 0 for c in X.columns}])
row["Backlogs"] = 0
row["CGPA"] = 8.5
row["CommunicationScore"] = 9.0
row["Specialization_Cloud Computing"] = 1
row["AttendanceBand_Medium"] = 1
row["InternshipDone_No"] = 1

print("no internship ->", model.predict(row)[0])

# flip ONE field -- everything else stays identical
row["InternshipDone_No"] = 0
row["InternshipDone_Yes"] = 1

print("with internship ->", model.predict(row)[0])