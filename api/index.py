from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import firebase_admin
from firebase_admin import credentials, firestore
import os
import re
import traceback # 상세 에러 확인용

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

    # 1. Firestore 캐시 확인 (비용 절감 및 속도 향상)
    cache_ref = db.collection("video_stt_cache").document(f"{video_id}_{request.lang}")
    cache_doc = cache_ref.get()
    
    if cache_doc.exists:
        return {"status": "success", "data": cache_doc.to_dict().get("sttData")}

    try:
        # 2. 유튜브 자막 "스마트" 추출 로직
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            # 시도 1: 사용자가 요청한 언어(예: 'ko')의 자막이 있는지 확인
            transcript = transcript_list.find_transcript([request.lang]).fetch()
        except:
            # 시도 2: 요청한 언어가 없다면, 이용 가능한 아무 자막(자동생성 포함)이나 가져와서 요청한 언어로 자동 번역!
            # 예: 영어 영상에 한국어 자막이 없으면 영어를 가져와서 한국어로 번역함
            for t in transcript_list:
                if t.is_translatable:
                    transcript = t.translate(request.lang).fetch()
                    break
            else:
                # 번역 가능한 자막조차 없는 경우
                raise Exception("No translatable subtitles found for this video.")

        # 데이터 포맷팅
        formatted_data = []
        for item in transcript:
            formatted_data.append({
                "start": item["start"],
                "end": item["start"] + item["duration"],
                "original": item["text"]
            })

        # 3. Firestore에 캐시 저장 (언어별로 따로 저장되도록 문서 ID에 언어코드 추가)
        cache_ref.set({
            "sttData": formatted_data,
            "language": request.lang,
            "processedAt": firestore.SERVER_TIMESTAMP
        })

        return {"status": "success", "data": formatted_data}

    except Exception as e:
        # 🚨 Vercel Logs에 정확한 파이썬 에러 원인을 출력합니다.
        error_msg = traceback.format_exc()
        print(f"STT Error for Video {video_id}: \n", error_msg)
        
        # 500 에러 대신 400 에러로 클라이언트에게 "자막이 없음"을 명확히 알림
        raise HTTPException(status_code=400, detail="해당 영상에서 자막을 추출할 수 없거나 자막이 비활성화되어 있습니다.")