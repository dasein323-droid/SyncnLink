from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import firebase_admin
from firebase_admin import credentials, firestore
import os
import re
import yt_dlp
import google.generativeai as genai
import tempfile
import json
import traceback

# Firebase 초기화
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
    video_id_match = re.search(r"(?:v=|youtu\.be\/)([^&]+)", request.url)
    if not video_id_match:
        raise HTTPException(status_code=400, detail="유효하지 않은 유튜브 URL입니다.")
    
    video_id = video_id_match.group(1)
    cache_ref = db.collection("video_stt_cache").document(f"{video_id}_{request.lang}")
    cache_doc = cache_ref.get()
    
    if cache_doc.exists:
        return {"status": "success", "data": cache_doc.to_dict().get("sttData")}

    try:
        # [STEP 1] 유튜브 자체 자막 추출 (1순위, 초고속)
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

    except Exception as e:
        print(f"유튜브 자막 없음. Gemini STT로 전환 (비디오: {video_id})")
        
        # [STEP 2] 자막이 없는 경우: 무료 Gemini 1.5 Flash STT 사용
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise HTTPException(status_code=400, detail="자막이 없는 영상입니다. (GEMINI_API_KEY가 없습니다.)")
            
        try:
            # 1. 오디오 다운로드
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, f"{video_id}.m4a")
            
            ydl_opts = {
                'format': 'm4a/bestaudio/best',
                'outtmpl': audio_path,
                'max_filesize': 20000000, # 약 20MB 제한
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([request.url])
                
            # 2. Gemini 설정 (빠른 응답을 위해 1.5-flash 모델 사용)
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('models/gemini-1.5-flash')
            
            # 3. Gemini 서버에 오디오 업로드
            audio_file = genai.upload_file(path=audio_path)
            
            # 4. 프롬프트 요청 (JSON 형태로 자막 요구)
            prompt = f"""
            Listen to this audio and transcribe it in {request.lang} language. 
            Split the transcription into short sentences. 
            Estimate the 'start' and 'end' time (in seconds) for each sentence.
            Return ONLY a valid JSON array format like this:
            [
              {{"start": 0.0, "end": 2.5, "original": "Hello world"}},
              {{"start": 2.5, "end": 5.0, "original": "Next sentence"}}
            ]
            Do not include any markdown tags (like ```json). Just the raw JSON.
            """
            
            response = model.generate_content([prompt, audio_file])
            
            # 5. 결과 파싱 및 정리
            result_text = response.text.strip()
            # markdown 코드가 섞여 올 경우를 대비한 제거 로직
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            elif result_text.startswith("```"):
                result_text = result_text[3:-3]
                
            formatted_data = json.loads(result_text)
            
            # 서버 용량 관리를 위해 임시 파일 삭제
            if os.path.exists(audio_path):
                os.remove(audio_path)
            genai.delete_file(audio_file.name) # 구글 서버에서도 삭제

        except Exception as gemini_err:
            error_msg = traceback.format_exc()
            print("Gemini STT 처리 실패:\n", error_msg)
            raise HTTPException(status_code=500, detail="영상이 너무 길거나 음성 분석에 실패했습니다.")

    # 3. Firestore 캐시 저장 (성공했을 때만)
    cache_ref.set({
        "sttData": formatted_data,
        "language": request.lang,
        "processedAt": firestore.SERVER_TIMESTAMP
    })

    return {"status": "success", "data": formatted_data}
