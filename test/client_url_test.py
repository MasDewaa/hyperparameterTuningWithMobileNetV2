import requests
import os

BASE_URL = "https://batik-deploy.railway.app/"  # Ganti dengan URL server Flask yang sesuai
ROOT_URL = f"{BASE_URL}/api/v1"
PREDICT_URL = f"{BASE_URL}/api/v1/predict"

IMAGE_PATH = "../Batik Nitik Grouped/Brendhi/11 Brendhi 1_rotate_90.jpg"

def test_root():
    print("\n=== 🔍 TEST ROOT ===")
    r = requests.get(ROOT_URL)
    print("[INFO] Status:", r.status_code)
    print("[INFO] Response:", r.text)

def test_predict_no_file():
    print("\n=== 🔍 TEST PREDICT TANPA FILE ===")
    r = requests.post(PREDICT_URL)
    print("[INFO] Status:", r.status_code)
    print("[INFO] Response:", r.text)

def test_predict_with_file():
    print("\n=== 🔍 TEST PREDICT DENGAN FILE ===")
    if not os.path.exists(IMAGE_PATH):
        print("[ERROR] Gambar tidak ditemukan:", IMAGE_PATH)
        return
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (os.path.basename(IMAGE_PATH), f)}
        r = requests.post(PREDICT_URL, files=files)
    print("[INFO] Status:", r.status_code)
    print("[INFO] Response:", r.text)

if __name__ == "__main__":
    test_root()
    test_predict_no_file()
    test_predict_with_file()
    