# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Load dataset
data = pd.read_csv("/content/iris (2).csv")


print(data.shape)

# Step 3: Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Default model (10 trees)
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy with 10 trees:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# Step 6: Hyperparameter tuning
best_score = 0
best_trees = 0

for n in range(1, 101):
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)

    if score > best_score:
        best_score = score
        best_trees = n

print("\nBest Accuracy:", best_score)
print("Best Number of Trees:", best_trees)
