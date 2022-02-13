# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
import pickle

dataset = pd.read_csv('ACT_Bank_Data.csv')

column1=dataset["Age"]
column2=dataset["Gender"]
column3=dataset["Married"]
column4=dataset["Job"]
column5=dataset["Annual Income"]
data=np.array([column1,column2,column3,column4,column5])
last_column=np.array(dataset["Did he accept the credit card"])
x = np.transpose(data) #array of the data
y = last_column 
x_train = x
y_train = y
clf = tree.DecisionTreeClassifier()
sv=clf.fit(x_train, y_train)



# from sklearn.svm import SVC
# sv = SVC(kernel='linear').fit(x_train,)
# Saving model to disk

pickle.dump(sv, open('model.pkl','wb'))

