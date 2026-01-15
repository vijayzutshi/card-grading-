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

def download_image(url):
    filename = f"{uuid.uuid4()}.png"   # unique filename every time
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return filename
    else:
        raise Exception(f"Failed to download image: {url}")

@app.get("/")
def read_root():
    return {"message": "Card Grading API is live!"}

@app.post("/analyze_centering")
def analyze(req: CenteringRequest):

    print("DOWNLOADING FRONT:", req.front_image_url)
    print("DOWNLOADING BACK:", req.back_image_url)
    result = analyze_centering(front_path, back_path)
    print("RETURNING RESULT:", result) # Add this line 
    return JSONResponse(content=result) # Ensure this is used

    # Download images with unique filenames
    front_path = download_image(req.front_image_url)
    back_path = download_image(req.back_image_url)

    # Run your centering logic
    result = analyze_centering(front_path, back_path)

    return result


