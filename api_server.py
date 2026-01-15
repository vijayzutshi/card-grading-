from fastapi import FastAPI
from pydantic import BaseModel
import requests
import shutil
import uuid
from card_centering import analyze_centering
from fastapi.responses import JSONResponse

app = FastAPI()

class CenteringRequest(BaseModel):
    front_image_url: str
    back_image_url: str
    card_id: str  # Include this if FlutterFlow sends it

def download_image(url):
    filename = f"{uuid.uuid4().hex}.png"  # unique filename every time
    headers = {"User-Agent": "Mozilla/5.0"}  # helps with Firebase download
    response = requests.get(url, stream=True, headers=headers)
    if response.status_code == 200:
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return filename
    else:
        print("DOWNLOAD FAILED:", response.status_code, url)
        raise Exception(f"Failed to download image: {url}")

@app.get("/")
def read_root():
    return {"message": "Card Grading API is live!"}

@app.post("/analyze_centering")
def analyze(req: CenteringRequest):
    print("DOWNLOADING FRONT:", req.front_image_url)
    print("DOWNLOADING BACK:", req.back_image_url)

    # Download images first
    front_path = download_image(req.front_image_url)
    back_path = download_image(req.back_image_url)

    # Run centering analysis
    result = analyze_centering(front_path, back_path)

    print("RETURNING RESULT:", result)

    # Return clean JSON response
    return JSONResponse(content={
        "success": True,
        "card_id": req.card_id,
        "psa_grade": result.get("psa_grade"),
        "ratio": result.get("ratio"),
        "centered": result.get("centered")
    })

  


