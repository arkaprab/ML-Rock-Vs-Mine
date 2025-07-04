import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
data = pd.read_csv(url, header=None)

X = data.iloc[:, :-1]
y = LabelEncoder().fit_transform(data.iloc[:, -1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC()
}

print("\nCross-Validation Accuracy:")
for name, model in models.items():
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"{name}: {score:.4f}")

final_model = RandomForestClassifier(n_estimators=100)
final_model.fit(X_train, y_train)
pred = final_model.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, pred):.4f}\n")
print("Classification Report:\n", classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ['Rock', 'Mine'])
plt.yticks([0, 1], ['Rock', 'Mine'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
