from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import os
from google import genai
import tempfile
import json
import traceback
import requests
import time
import re

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

RATE_LIMIT_DELAY = 65

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

    try:
        print(f"[STEP 1] Downloading audio: {video_id}")

        rapid_api_url = "https://youtube-mp36.p.rapidapi.com/dl"
        querystring = {"id": video_id} 

        headers = {
            "x-rapidapi-key": os.getenv("RAPIDAPI_KEY", ""),
            "x-rapidapi-host": "youtube-mp36.p.rapidapi.com"
        }

        # 1. API 폴링 (대기열 처리) 로직
        max_retries = 10
        audio_url = ""

        for attempt in range(max_retries):
            response = requests.get(rapid_api_url, headers=headers, params=querystring)
            response_data = response.json()
            audio_url = response_data.get("link", "")

            if response.status_code == 200 and audio_url.startswith("http"):
                break
            
            msg = response_data.get("msg", "")
            if msg == "in process":
                print(f"API processing video, waiting 3 seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(3)
                continue
            else:
                raise Exception(f"YouTube API failed: {msg}")

        if not audio_url.startswith("http"):
            raise Exception("YouTube API timeout: Audio extraction took too long.")

        print("Downloading audio...")
        audio_data = requests.get(audio_url).content
        with open(audio_path, 'wb') as f:
            f.write(audio_data)

        print("[STEP 2] Gemini analysis")
        gemini_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=gemini_key)

        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        lang_name = "Korean" if request.lang == "ko" else "English"
        prompt = (
            f"Listen to the attached audio. If the audio is in another language, TRANSLATE the meaning to {lang_name}. "
            f"If it is already in {lang_name}, TRANSCRIBE it. "
            f"CRITICAL: Base your output STRICTLY on the actual audio content. Do NOT hallucinate, guess, or create dummy/sample text (like '유라 씨...'). "
            f"If the audio contains no speech or is just music, return an empty array []. "
            f"Return strictly as a valid JSON array: [{{ \"start\": 0.0, \"end\": 1.5, \"text\": \"actual speech\" }}]"
        )

        # 2. 최신 모델 순차 호출 (Fallback) 로직
        models_to_try = [
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-3.1-flash-lite"
        ]

        response = None
        for model_name in models_to_try:
            try:
                print(f"Trying Gemini model: {model_name}...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        genai.types.Part(
                            inline_data=genai.types.Blob(
                                mime_type="audio/mpeg",
                                data=audio_bytes
                            )
                        )
                    ]
                )
                print(f"SUCCESS with {model_name}")
                break
            except Exception as e:
                print(f"WARNING: {model_name} failed. Error: {e}")
                continue

        if not response:
            raise Exception("All Gemini model attempts failed.")

        result_text = response.text.strip()

        # 3. 정규표현식을 활용한 안전한 JSON 추출 로직
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        
        if json_match:
            clean_json_str = json_match.group(0)
            formatted_data = json.loads(clean_json_str)
            print("SUCCESS: JSON conversion complete")
        else:
            raise Exception(f"Failed to extract JSON. Raw response: {result_text[:200]}")

        print(f"Rate limit: waiting {RATE_LIMIT_DELAY}s before next request...")
        time.sleep(RATE_LIMIT_DELAY)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

        if formatted_data is None:
            raise HTTPException(status_code=500, detail="Audio analysis failed")

    if formatted_data:
        try:
            cache_ref.set({
                "sttData": formatted_data,
                "language": request.lang,
                "processedAt": firestore.SERVER_TIMESTAMP
            })
            print("Cache saved")
        except Exception as e:
            print(f"Cache error: {e}")

    return {"status": "success", "data": formatted_data}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
