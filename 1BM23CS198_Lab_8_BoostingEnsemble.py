import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("income.csv")

print(data.head())

# -----------------------------
# Data Preprocessing
# -----------------------------
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Features and target
X = data.drop('income_level', axis=1)
y = data['income_level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# AdaBoost Model (FIXED HERE)
# -----------------------------
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # ✅ changed
    n_estimators=10,
    learning_rate=1,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with 10 estimators:", accuracy)

# -----------------------------
# Fine-tuning
# -----------------------------
for n in [5, 10, 20, 50, 100]:
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),  # ✅ changed
        n_estimators=n,
        learning_rate=1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy with {n} estimators: {acc}")
