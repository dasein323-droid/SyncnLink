from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import firebase_admin
from firebase_admin import credentials, firestore
import os
import yt_dlp
import google.generativeai as genai
import tempfile
import json
import traceback

# Firebase 초기화 (환경변수 누락 시 에러 방지)
if not firebase_admin._apps:
    try:
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
    except Exception as e:
        print("Firebase Init Error:", e)
        db = None

app = FastAPI()

# CORS 설정 추가 (통신 오류 방지)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class STTRequest(BaseModel):
    videoId: str  # url 대신 videoId를 직접 받음
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

    # [STEP 1] 유튜브 자막 1차 시도
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript([request.lang]).fetch()
        except:
            for t in transcript_list:
                if t.is_translatable:
                    transcript = t.translate(request.lang).fetch()
                    break
            else:
                raise Exception("번역 가능한 자막 없음")

        formatted_data = [{"start": i["start"], "end": i["start"] + i["duration"], "original": i["text"]} for i in transcript]

    # [STEP 2] 유튜브 자막이 아예 없는 경우 -> Gemini로 전환
    except Exception as e:
        print(f"🎬 자막 없음 감지됨. Gemini STT로 분석을 시작합니다. (비디오: {video_id})")
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise HTTPException(status_code=400, detail="서버에 Gemini API Key가 설정되지 않았습니다.")
            
        try:
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, f"{video_id}.m4a")
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            
            ydl_opts = {
                'format': 'm4a/bestaudio',
                'outtmpl': audio_path,
                'noplaylist': True,
                'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
                
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('models/gemini-1.5-flash')
            audio_file = genai.upload_file(path=audio_path)
            
            prompt = f"""
            Listen to this audio and transcribe it in {request.lang} language. 
            Split the transcription into short sentences. 
            Estimate the 'start' and 'end' time (in seconds) for each sentence.
            Return ONLY a valid JSON array format like this, nothing else:
            [
              {{"start": 0.0, "end": 2.5, "original": "Hello"}},
              {{"start": 2.5, "end": 5.0, "original": "World"}}
            ]
            """
            
            response = model.generate_content([prompt, audio_file])
            result_text = response.text.strip()
            
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            elif result_text.startswith("```"):
                result_text = result_text[3:-3]
                
            formatted_data = json.loads(result_text)
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
            genai.delete_file(audio_file.name)

        except Exception as gemini_err:
            error_msg = traceback.format_exc()
            print("🚨 Gemini STT 처리 최종 실패:\n", error_msg)
            # Vercel 타임아웃(10초)에 걸릴 확률이 높으므로 명확한 에러 반환
            raise HTTPException(status_code=500, detail="자막 추출 실패 (영상이 너무 길거나 Vercel 타임아웃 발생)")

    # 3. 데이터베이스(Firestore) 저장
    try:
        cache_ref.set({
            "sttData": formatted_data,
            "language": request.lang,
            "processedAt": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print("Firestore Cache Save Error:", e)

    return {"status": "success", "data": formatted_data}
