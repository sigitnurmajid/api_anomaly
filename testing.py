import requests
import json

# URL endpoint
url = 'http://127.0.0.1:5000/predict_conductivity'

# Data input (dalam format JSON)
data = [
    {"value": 1.2},
    {"value": 1.3},
    {"value": 1.5},
    {"value": 1.6},
    {"value": 1.7}
]

# Kirim request POST
response = requests.post(url, json=data)

# Tampilkan response
if response.status_code == 200:
    print("Response JSON:", response.json())
else:
    print(f"Error: {response.status_code}")
