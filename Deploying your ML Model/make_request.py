import requests

url = 'http://localhost:3000/predict'

r = requests.post(url,json={'text': 'very good'})
print(r.json())