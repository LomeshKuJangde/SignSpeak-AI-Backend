import requests
from pathlib import Path

# API URL
url = "http://127.0.0.1:5000/predict"

# Put your real image path here
image_path = Path(r"D:\Real time Sign Language\dataset\validation\A\1000.jpg")

print("Checking file exists:", image_path.exists())
print("Using file:", image_path)

if not image_path.exists():
    print("❌ ERROR: Image file not found. Check path again.")
else:
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}
        response = requests.post(url, files=files)

    print("Status Code:", response.status_code)
    print("Response Text:", response.text)