import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("Data/BankNote_Authentication.csv")

X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# Implimentat Randomforest Classifier 
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
score=accuracy_score(y_test,y_pred)
score

# Create a pickle file using serilization
import pickle 

pickle_out=open("classifier.pkl","wb")
pickle.dump(classifier,pickle_out)

classifier.predict([[3.62160,8.6661,-2.8073,-0.44699]])

classifier.predict([[-2.5419,-0.6580,2.6842,1.19520]])