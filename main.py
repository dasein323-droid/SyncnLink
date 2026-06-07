from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import os
import google.generativeai as genai
import tempfile
import json
import traceback
import requests

# Firebase 초기화
if not firebase_admin._apps:
    try:
        raw_private_key = os.getenv("FIREBASE_PRIVATE_KEY", "")
        clean_private_key = raw_private_key.strip('"').replace('\\n', '\n')

        cred_json = {
            "type": "service_account",
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key": clean_private_key,
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        cred = credentials.Certificate(cred_json)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
    except Exception as e:
        print("Firebase Init Error:", e)
        db = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class STTRequest(BaseModel):
    videoId: str
    lang: str


@app.post("/api/stt")
async def process_stt(request: STTRequest):
    video_id = request.videoId

    if not db:
        raise HTTPException(status_code=500, detail="Firebase DB initialization failed")

    cache_ref = db.collection("video_stt_cache").document(f"{video_id}_{request.lang}")
    cache_doc = cache_ref.get()

    if cache_doc.exists:
        return {"status": "success", "data": cache_doc.to_dict().get("sttData")}

    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, f"{video_id}.mp3")
    formatted_data = None
    audio_file = None

    try:
        print(f"[STEP 1] Downloading audio from YouTube API: {video_id}")

        rapid_api_url = "https://youtube-mp36.p.rapidapi.com/dl"
        querystring = {"id": video_id} 

        headers = {
            "x-rapidapi-key": "4966da32e6msh7182c742dac2424p10afb7jsn0d01b22c96ff",
            "x-rapidapi-host": "youtube-mp36.p.rapidapi.com"
        }

        print("Sending API request...")
        response = requests.get(rapid_api_url, headers=headers, params=querystring)
        response_data = response.json()

        print("API response received")

        audio_url = response_data.get("link", "")

        if response.status_code != 200 or not audio_url.startswith("http"):
            error_msg = response_data.get("msg") or response_data.get("message") or "API failed to provide download link"
            print("API error:", response_data)
            raise Exception(f"YouTube API failed: {error_msg}")

        print("Downloading audio file...")

        audio_data = requests.get(audio_url).content
        with open(audio_path, 'wb') as f:
            f.write(audio_data)

        print("Audio file saved successfully")

        print("[STEP 2] Starting Gemini STT analysis")
        gemini_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_key)

        target_model = 'gemini-2.0-flash'
        print(f"Using model: {target_model}")

        model = genai.GenerativeModel(target_model)

        prompt = f"""Listen to this audio. Regardless of the original language, translate and summarize the content into natural {request.lang}.
Split the translated transcription into short, readable sentences. 
Estimate the 'start' and 'end' time (in seconds) for each sentence matching the audio timeline.
Return ONLY a valid JSON array format like this, nothing else:
[
  {{"start": 0.0, "end": 2.5, "original": "Sample text"}},
  {{"start": 2.5, "end": 5.0, "original": "More sample text"}}
]"""

        print("Uploading audio to Gemini API...")

        audio_file = genai.upload_file(audio_path, mime_type="audio/mpeg")
        print(f"File uploaded: {audio_file.name}")

        gemini_response = model.generate_content([
            prompt,
            audio_file
        ])

        result_text = gemini_response.text.strip()
        print("Gemini response received. Converting to JSON...")

        if result_text.startswith("```json"):
            result_text = result_text[7:-3]
        elif result_text.startswith("```"):
            result_text = result_text[3:-3]

        formatted_data = json.loads(result_text)
        print("JSON conversion successful")

    except Exception as e:
        print("Analysis failed:")
        print(str(e))
        print(traceback.format_exc())

    finally:
        if audio_file is not None:
            try:
                genai.delete_file(audio_file.name)
                print(f"Gemini file deleted: {audio_file.name}")
            except Exception as e:
                print(f"Warning - could not delete Gemini file: {e}")

        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"Local file deleted: {audio_path}")
            except Exception as e:
                print(f"Warning - could not delete local file: {e}")

        if formatted_data is None:
            raise HTTPException(status_code=500, detail="Audio analysis failed")

    if formatted_data:
        try:
            cache_ref.set({
                "sttData": formatted_data,
                "language": request.lang,
                "processedAt": firestore.SERVER_TIMESTAMP
            })
            print("Firestore cache saved")
        except Exception as e:
            print("Firestore cache error:", e)

    return {"status": "success", "data": formatted_data}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
