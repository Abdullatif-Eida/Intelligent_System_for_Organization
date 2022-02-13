import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Age':2, 'Married':9, 'Gender':6, 'Job':8, 'Annual_income':4})

print(r.json())