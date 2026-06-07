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
import urllib.request
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


# ... (FastAPI 및 Firebase 초기화 코드는 동일) ...

@app.post("/api/stt")
async def process_stt(request: STTRequest):
    video_id = request.videoId
    
    if not db:
        raise HTTPException(status_code=500, detail="Firebase DB가 초기화되지 않았습니다.")

    cache_ref = db.collection("video_stt_cache").document(f"{video_id}_{request.lang}")
    cache_doc = cache_ref.get()
    
    if cache_doc.exists:
        return {"status": "success", "data": cache_doc.to_dict().get("sttData")}

    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, f"{video_id}.mp3")
    formatted_data = None

    try:
        # [STEP 1] 외부 API를 활용한 오디오 다운로드 (yt-dlp 대체)
        print(f"🔄 외부 API를 통해 오디오 확보 시작: {video_id}")
        
        # 검증된 Youtube MP36 API 엔드포인트
        rapid_api_url = "https://youtube-mp36.p.rapidapi.com/dl"
        
        # 중요: url 전체가 아닌 video_id만 전달해야 합니다.
        querystring = {"id": video_id} 
        
        headers = {
            "x-rapidapi-key": "4966da32e6msh7182c742dac2424p10afb7jsn0d01b22c96ff", 
            "x-rapidapi-host": "youtube-mp36.p.rapidapi.com"
        }

        print("API 요청 중...")
        response = requests.get(rapid_api_url, headers=headers, params=querystring)
        response_data = response.json()
        
        print("API 응답 데이터:", response_data) 

        # [수정된 부분] link 항목이 아예 없거나, 빈 문자열("")인 경우 모두 걸러냅니다.
        audio_url = response_data.get("link", "")
        
        if response.status_code != 200 or not audio_url.startswith("http"):
            # 실패 원인을 정확히 로그에 남깁니다.
            error_msg = response_data.get("msg") or response_data.get("message") or "API가 유효한 다운로드 링크를 제공하지 않았습니다."
            print("🚨 API 에러 전체 응답:", response_data)
            raise Exception(f"외부 API 실패: {error_msg}")

        # 오디오 파일 다운로드 및 서버 임시 저장
        print("오디오 다운로드 링크 확보 성공, 다운로드 중...")
        
        audio_data = requests.get(audio_url).content
        with open(audio_path, 'wb') as f:
            f.write(audio_data)
        
        print("오디오 파일 임시 저장 완료.")

        # [STEP 2] Gemini STT로 번역 및 타임라인 추출
        print(f"🎬 오디오 확보 성공. Gemini STT 분석을 시작합니다.")
        gemini_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_key)
        
        # 로그에서 확인된 사용 가능한 최신 모델로 고정합니다.
        target_model = 'gemini-3.5-flash'
        print(f"🚀 최종 선택된 모델: {target_model}")
        
        model = genai.GenerativeModel(target_model)
        
        prompt = f"""
        Listen to this audio. Regardless of the original language, translate and summarize the content into natural {request.lang} (Korean).
        Split the translated transcription into short, readable sentences. 
        Estimate the 'start' and 'end' time (in seconds) for each sentence matching the audio timeline.
        Return ONLY a valid JSON array format like this, nothing else:
        [
          {{"start": 0.0, "end": 2.5, "original": "안녕하세요, 오늘 살펴볼 주제는..."}},
          {{"start": 2.5, "end": 5.0, "original": "바로 이것입니다."}}
        ]
        """
        
        print("Gemini API로 오디오 전송 중...")
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            
        gemini_response = model.generate_content([
            prompt,
            {
                "mime_type": "audio/mp3",
                "data": audio_bytes
            }
        ])
        
        result_text = gemini_response.text.strip()
        print("Gemini 응답 완료. JSON 변환 시도...")
        
        if result_text.startswith("```json"):
            result_text = result_text[7:-3]
        elif result_text.startswith("```"):
            result_text = result_text[3:-3]
            
        formatted_data = json.loads(result_text)
        
        # 처리 완료 후 파일 삭제
        genai.delete_file(audio_file.name)
        if os.path.exists(audio_path):
            os.remove(audio_path)

    except Exception as e:
        print("🚨 분석 최종 실패:\n", str(e))
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(status_code=500, detail=f"오디오 확보 또는 분석에 실패했습니다: {str(e)}")

    # [STEP 3] 정상 추출된 경우 Firestore에 캐싱
    if formatted_data:
        try:
            cache_ref.set({
                "sttData": formatted_data,
                "language": request.lang,
                "processedAt": firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            print("Firestore Cache Save Error:", e)

    return {"status": "success", "data": formatted_data}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
