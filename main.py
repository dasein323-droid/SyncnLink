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
from mutagen import File

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
    duration: float  # 💡 프론트엔드에서 전달받는 정확한 원본 영상 길이

RATE_LIMIT_DELAY = 65

@app.post("/api/stt")
async def process_stt(request: STTRequest):
    video_id = request.videoId
    original_duration = request.duration # 프론트엔드에서 받은 길이 사용

    if not db:
        raise HTTPException(status_code=500, detail="Firebase DB initialization failed")

    cache_ref = db.collection("video_stt_cache").document(f"{video_id}_{request.lang}")
    cache_doc = cache_ref.get()

    if cache_doc.exists:
        return {"status": "success", "data": cache_doc.to_dict().get("sttData")}

    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, f"{video_id}.m4a")
    formatted_data = None

    try:
        print(f"📺 프론트엔드에서 전달받은 원본 영상 길이: {original_duration}초")

        print(f"[STEP 1] Downloading audio via RapidAPI: {video_id}")
        rapid_api_url = "https://youtube-convert-mp3-m4a.p.rapidapi.com/v1/social/youtube/audio"
        payload = {"id": video_id, "ext": "m4a"} 
        headers = {
            "x-rapidapi-key": os.getenv("RAPIDAPI_KEY", ""),
            "x-rapidapi-host": "youtube-convert-mp3-m4a.p.rapidapi.com",
            "Content-Type": "application/json"
        }

        max_ad_retries = 5 # 광고 추출 시 최대 재시도 횟수 증가
        valid_audio_downloaded = False

        for ad_attempt in range(max_ad_retries):
            print(f"🔄 오디오 추출 시도 {ad_attempt + 1}/{max_ad_retries}...")
            
            audio_url = ""
            max_api_retries = 10
            
            for attempt in range(max_api_retries):
                response = requests.post(rapid_api_url, headers=headers, json=payload)
                try:
                    response_data = response.json()
                except Exception:
                    raise Exception(f"API 응답 파싱 실패. HTTP 코드: {response.status_code}")

                audio_url = response_data.get("linkDownload", "")

                if response.status_code == 200 and audio_url.startswith("http"):
                    break
                
                if response_data.get("error") is True:
                    raise Exception(f"YouTube API failed: {response_data}")

                time.sleep(3)
                
            if not audio_url.startswith("http"):
                raise Exception("YouTube API timeout: Audio extraction took too long.")

            print("⬇️ Downloading audio file...")
            audio_data = requests.get(audio_url).content
            with open(audio_path, 'wb') as f:
                f.write(audio_data)

            # 다운로드된 음원 길이 검증
            try:
                audio_file_meta = File(audio_path)
                downloaded_duration = audio_file_meta.info.length if audio_file_meta else 0.0
                print(f"🎵 다운로드된 음원 길이: {downloaded_duration:.2f}초")
            except Exception as e:
                print(f"⚠️ 음원 길이 분석 실패: {e}")
                downloaded_duration = 0.0

            # 🚨 검증 로직: 다운로드된 음원이 원본보다 15초 이상 짧거나, 원본 길이의 80% 미만이면 광고로 간주
            if original_duration > 0 and downloaded_duration > 0:
                if downloaded_duration < (original_duration - 15) or downloaded_duration < (original_duration * 0.8):
                    print(f"🚫 [광고 감지] 원본({original_duration}초)에 비해 음원({downloaded_duration:.2f}초)이 너무 짧습니다. 광고로 간주하고 폐기합니다.")
                    os.remove(audio_path)
                    time.sleep(3) # 3초 지연 후 재요청
                    continue
            
            print("✅ 정상적인 본 영상 음원으로 판별되었습니다.")
            valid_audio_downloaded = True
            break

        if not valid_audio_downloaded:
            raise Exception("연속된 재시도에도 불구하고 계속해서 광고 음원이 추출되었습니다.")

        print("[STEP 2] Gemini analysis")
        gemini_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=gemini_key)

        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        lang_name = "Korean" if request.lang == "ko" else "English"
        
        # 💡 프롬프트 수정: JSON 키를 프론트엔드와 동일하게 "original"로 변경
        prompt = (
            f"Listen to the attached audio. If the audio is in another language, TRANSLATE the meaning to {lang_name}. "
            f"If it is already in {lang_name}, TRANSCRIBE it. "
            f"CRITICAL INSTRUCTIONS: "
            f"1. The audio might begin with a random YouTube pre-roll advertisement (e.g., fitness, product ads). IGNORE ALL ADVERTISEMENTS completely. "
            f"2. ONLY transcribe/translate the main content (e.g., movie scenes, music, actual content). "
            f"3. Base your output STRICTLY on the actual audio. Do NOT hallucinate, guess, or create dummy text. "
            f"4. If the audio is 100% advertisement or contains no main speech, return an empty array []. "
            f"Return strictly as a valid JSON array: [{{\"start\": 0.0, \"end\": 1.5, \"original\": \"actual speech\"}}]"
        )

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
                                mime_type="audio/mp4",
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
