from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import firebase_admin
from firebase_admin import credentials, firestore
import os
import re

# Vercel 환경변수에서 Firebase 인증 정보 로드
if not firebase_admin._apps:
    cred_json = {
        "type": "service_account",
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    cred = credentials.Certificate(cred_json)
    firebase_admin.initialize_app(cred)

db = firestore.client()
app = FastAPI()

class STTRequest(BaseModel):
    url: str
    lang: str

@app.post("/api/stt")
async def process_stt(request: STTRequest):
    # 유튜브 URL에서 Video ID 추출
    video_id_match = re.search(r"(?:v=|youtu\.be\/)([^&]+)", request.url)
    if not video_id_match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    video_id = video_id_match.group(1)

    # 1. Firestore 캐시 확인 (비용 절감)
    cache_ref = db.collection("video_stt_cache").document(video_id)
    cache_doc = cache_ref.get()
    
    if cache_doc.exists:
        return {"status": "success", "data": cache_doc.to_dict().get("sttData")}

    try:
        # 2. 유튜브 자막 추출
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[request.lang])
        
        # 데이터 포맷팅
        formatted_data = []
        for item in transcript:
            formatted_data.append({
                "start": item["start"],
                "end": item["start"] + item["duration"],
                "original": item["text"]
            })

        # 3. Firestore에 캐시 저장
        cache_ref.set({
            "sttData": formatted_data,
            "language": request.lang,
            "processedAt": firestore.SERVER_TIMESTAMP
        })

        return {"status": "success", "data": formatted_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))