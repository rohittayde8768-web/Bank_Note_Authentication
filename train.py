```python
import pandas as pd
import numpy as np
import warnings
import pickle

warnings.filterwarnings("ignore")


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("Data/BankNote_Authentication.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. Separate Features and Target
# ==========================================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ==========================================
# 3. Train-Test Split
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=0
)


# ==========================================
# 4. Train Random Forest Classifier
# ==========================================

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X_train, y_train)

print("\nModel trained successfully!")


# ==========================================
# 5. Prediction
# ==========================================

y_pred = classifier.predict(X_test)


# ==========================================
# 6. Model Evaluation
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

score = accuracy_score(y_test, y_pred)

print("\n===================================")
print("Model Accuracy:", score)
print("===================================")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 7. Save Model
# ==========================================

with open("classifier.pkl", "wb") as pickle_out:
    pickle.dump(classifier, pickle_out)

print("\nclassifier.pkl created successfully!")


# ==========================================
# 8. Test Saved Model
# ==========================================

with open("classifier.pkl", "rb") as pickle_in:
    loaded_classifier = pickle.load(pickle_in)

print("\nSaved model loaded successfully!")


# ==========================================
# 9. Test Predictions
# ==========================================

sample_1 = [[3.62160, 8.6661, -2.8073, -0.44699]]

sample_2 = [[-2.5419, -0.6580, 2.6842, 1.19520]]

prediction_1 = loaded_classifier.predict(sample_1)
prediction_2 = loaded_classifier.predict(sample_2)

print("\nPrediction for Sample 1:", prediction_1)
print("Prediction for Sample 2:", prediction_2)

print("\n===================================")
print("Training completed successfully!")
print("===================================")
```
