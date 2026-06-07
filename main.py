from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import os
import yt_dlp
import google.generativeai as genai
import tempfile
import json
import traceback
import urllib.request

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

    cookie_path = os.path.join(os.path.dirname(__file__), "cookies.txt")
    has_cookie = os.path.exists(cookie_path)
    formatted_data = None

    # [STEP 1] yt-dlp를 이용한 자막 직접 추출 (IP 차단 우회)
    try:
        print(f"🔍 yt-dlp로 자막 추출 시도: {video_id}")
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': [request.lang],
            'quiet': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['tv', 'mweb'], # TV 클라이언트로 위장
                    'player_skip': ['webpage', 'configs']
                }
            }
        }
        if has_cookie:
            ydl_opts['cookiefile'] = cookie_path

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

        subs = info.get('subtitles', {})
        auto_subs = info.get('automatic_captions', {})

        # 요청한 언어의 자막 찾기
        target_sub_list = subs.get(request.lang) or auto_subs.get(request.lang)

        if target_sub_list:
            # json3 포맷의 자막 URL 추출
            json3_url = next((sub['url'] for sub in target_sub_list if sub['ext'] == 'json3'), None)

            if json3_url:
                # 자막 데이터 다운로드 및 파싱
                req = urllib.request.Request(json3_url)
                with urllib.request.urlopen(req) as response:
                    sub_data = json.loads(response.read().decode())

                formatted_data = []
                for event in sub_data.get('events', []):
                    if 'segs' in event and 'tStartMs' in event:
                        start = event['tStartMs'] / 1000.0
                        duration = event.get('dDurationMs', 0) / 1000.0
                        text = "".join([seg.get('utf8', '') for seg in event['segs']])
                        if text.strip() and text.strip() != '\n':
                            formatted_data.append({
                                "start": start,
                                "end": start + duration,
                                "original": text.strip()
                            })
        
        if not formatted_data:
            raise Exception("해당 언어의 자막이 존재하지 않습니다.")

    except Exception as e:
        print(f"⚠️ yt-dlp 자막 추출 실패: {e}")
        formatted_data = None

    # [STEP 2] 자막 추출 실패 시 Gemini STT로 오디오 분석 (Fallback)
    if not formatted_data:
        print(f"🎬 자막 없음 감지됨. Gemini STT로 분석을 시작합니다. (비디오: {video_id})")
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise HTTPException(status_code=400, detail="서버에 Gemini API Key가 설정되지 않았습니다.")
            
        try:
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, f"{video_id}.m4a")
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            
            ydl_opts_audio = {
                'format': 'm4a/bestaudio/best',
                'outtmpl': audio_path,
                'noplaylist': True,
                'quiet': True,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['tv', 'mweb'],
                        'player_skip': ['webpage', 'configs']
                    }
                }
            }
            if has_cookie:
                ydl_opts_audio['cookiefile'] = cookie_path
            
            with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
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
            raise HTTPException(status_code=500, detail="자막 추출 및 STT 분석에 모두 실패했습니다. (유튜브 봇 차단)")

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
