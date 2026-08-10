import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

df = pd.read_csv("student_placement_history.csv")

X = pd.get_dummies(df.drop(columns=["StudentID", "Placed"]))
y = df["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

print(export_text(model, feature_names=list(X.columns)))
print("leaves:", model.get_n_leaves(), "| depth:", model.get_depth())

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=["Not Placed", "Placed"],
    filled=True,
    rounded=True,
    fontsize=9,
)

plt.title("This is placement_predictor.py's actual model")
plt.savefig("the_actual_model.png")
plt.show()