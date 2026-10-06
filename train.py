import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# load the dataset
df=pd.read_csv('data/BankNote_Authentication.csv')  # Load your dataset

#feature selection and target variable
x=df.drop('class',axis=1)
y=df['class']

print(x.head())
print(y.head())

# train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

# Model training
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred=classifier.predict(x_test)

# Evaluate the model
score=accuracy_score(y_test,y_pred)
print("Accuracy:", score)

#save the model
joblib.dump(classifier, 'classifier.pkl')
print("Model saved successfully")
