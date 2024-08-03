import requests

url = "http://127.0.0.1:8000/predict"
headers = {
    "accept": "application/json",
    # "Content-Type": "multipart/form-data"
}
files = {
    "file": open("sample_img.jpg", "rb")
}

response = requests.post(url, headers=headers, files=files)

print(response.json())
