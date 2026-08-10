import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

df = pd.read_csv("student_placement_history.csv")
X = pd.get_dummies(df.drop(columns=["StudentID","Placed"]))
y = df["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

dumb = DummyClassifier(strategy="most_frequent")
dumb.fit(X_train, y_train)
print(dumb.score(X_test, y_test))