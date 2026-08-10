import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("student_placement_history.csv")
X = pd.get_dummies(df.drop(columns=["StudentID","Placed"]))
y = df["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
 
cm = confusion_matrix(y_test, predictions)
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=["Not Placed","Placed"])
disp.plot()
plt.title("Placement Predictor .. Confusion Matrix")
plt.savefig("Confusion_matrix.png")
plt.show()