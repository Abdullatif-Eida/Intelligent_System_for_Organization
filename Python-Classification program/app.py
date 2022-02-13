import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import pandas
from sklearn import tree


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
dataset = pandas.read_csv('ACT_Bank_Data.csv')
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

@app.route('/')
def main_page():
    return render_template('main-page.html')

@app.route('/admin-main-page.html')
def admin_main_page():
    return render_template('admin-main-page.html')

@app.route('/classification_one_customer')
def classification_one():
    return render_template('Classificate_One_Customer.html')

@app.route('/classification_Customers')
def classification_many():
    return render_template('Classificate_Customers.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data1 = request.form['Age']
    data2 = request.form['Married']
    data3 = request.form['Gender']
    data4 = request.form['Job']
    data5 = request.form['Annual_income']
    arr = np.array([[data1, data2, data3, data4, data5]])
    pred = model.predict(arr)
    return render_template('Classificate_One_Customer.html', prediction_text=pred)

@app.route('/customers_predict',methods=['POST'])
def customers_predict():
    '''
    For rendering results on HTML GUI
    '''
    datafileset=pandas.read_csv(request.form['File Upload']) 
    column1=datafileset["Age"]
    column2=datafileset["Gender"]
    column3=datafileset["Married"]
    column4=datafileset["Job"]
    column5=datafileset["Annual Income"]
    data=np.array([column1,column2,column3,column4,column5])
    x = np.transpose(data) #array of the data
    x_test = x
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return render_template('Classificate_Customers.html', prediction_text=pred)

if __name__ == "__main__":
    app.run(debug=True)