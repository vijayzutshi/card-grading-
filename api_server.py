from fastapi import FastAPI
from pydantic import BaseModel
import requests
import shutil
from card_centering import analyze_centering  # your centering logic

app = FastAPI()

# -----------------------------
# Request Model
# -----------------------------
class CenteringRequest(BaseModel):
    front_image_url: str
    back_image_url: str

# -----------------------------
# Helper: Download images
# -----------------------------
def download_image(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
    else:
        raise Exception(f"Failed to download image: {url}")

# -----------------------------
# Root Route (Fixes 404 + Swagger redirect)
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Card Grading API is live!"}

# -----------------------------
# Main API Endpoint
# -----------------------------
@app.post("/analyze_centering")
def analyze(req: CenteringRequest):
    # Download images
    download_image(req.front_image_url, "front_temp.png")
    download_image(req.back_image_url, "back_temp.png")

    # Run your centering logic
    result = analyze_centering("front_temp.png", "back_temp.png")

    return result
