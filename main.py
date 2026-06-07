from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

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
        raise HTTPException(status_code=500, detail="Firebase DB가 초기화되지 않았습니다.")

    cache_ref = db.collection("video_stt_cache").document(f"{video_id}_{request.lang}")
    cache_doc = cache_ref.get()
    
    if cache_doc.exists:
        return {"status": "success", "data": cache_doc.to_dict().get("sttData")}

    # 🚨 핵심: 쿠키 파일 경로 확인
    cookie_path = os.path.join(os.path.dirname(__file__), "cookies.txt")
    has_cookie = os.path.exists(cookie_path)

    if has_cookie:
        print(f"✅ [쿠키 확인] {cookie_path} 적용 완료")
    else:
        print("❌ [쿠키 누락] cookies.txt 파일이 없습니다. 봇 차단이 발생할 수 있습니다.")

    # [STEP 1] 유튜브 자막 추출 (쿠키 적용)
    try:
        # 쿠키가 있으면 쿠키를 포함해서 요청 (봇 차단 완벽 우회)
        if has_cookie:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookie_path)
        else:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
        try:
            # 1. 요청한 언어(ko 또는 en) 자막 찾기
            transcript = transcript_list.find_transcript([request.lang]).fetch()
        except:
            # 2. 없으면 자동 생성 자막이나 다른 언어를 번역해서 가져오기
            for t in transcript_list:
                if t.is_translatable:
                    transcript = t.translate(request.lang).fetch()
                    break
            else:
                raise HTTPException(status_code=404, detail="해당 언어로 번역할 수 있는 자막이 없습니다.")

        formatted_data = [{"start": i["start"], "end": i["start"] + i["duration"], "original": i["text"]} for i in transcript]

    except Exception as e:
        error_msg = str(e)
        print(f"🚨 자막 추출 실패: {error_msg}")
        
        if "Subtitles are disabled" in error_msg or "NoTranscriptFound" in error_msg:
            raise HTTPException(status_code=404, detail="자막을 가져올 수 없습니다. (실제 자막 없음)")
        elif "cookies provided are not valid" in error_msg:
            raise HTTPException(status_code=401, detail="서버의 유튜브 쿠키가 만료되었습니다. 관리자에게 문의하세요.")
        else:
            raise HTTPException(status_code=500, detail=f"서버 오류: {error_msg}")

    # 정상 추출된 경우 Firestore에 캐싱
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
