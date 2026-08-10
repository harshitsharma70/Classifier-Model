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

importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False).head(5).round(4))