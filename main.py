import os
from datetime import datetime
import shutil
from typing import Dict, Any, Optional, List, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from google.cloud import speech, texttospeech
from google.oauth2 import service_account
from groq import Groq
from pydub import AudioSegment
import openai
import base64

# Load environment variables
load_dotenv()

# Settings Configuration
class Settings(BaseSettings):
    APP_NAME: str = "Azaan Trainer API"
    VERSION: str = "1.0.0"
    AUDIO_GAIN: float = 1.50
    IDEAL_TEXT: str = "اللّٰهُ أَكْبَرُ، اللّٰهُ أَكْبَرُ"
    IDEAL_TEXT_MEANING: str = "Allah is the Greatest, Allah is the Greatest"
    GOOGLE_CREDENTIALS_PATH: str = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials/sa_speecch_demo.json")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/wav2vec2-base")
    EMBEDDING_PATH: str = os.getenv("EMBEDDING_PATH", "embeddings/ideal_embedding_part_1.npy")
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()

# Pydantic Models
class WaveformData(BaseModel):
    time: List[float]
    amplitude: List[float]

class FeedbackData(BaseModel):
    text: str
    score_category: str

class AnalysisResponse(BaseModel):
    similarity_score: float
    transcription: str
    validation_status: str
    feedback: FeedbackData
    waveform_data: WaveformData
    feedback_audio: Optional[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str

# Services
class AudioService:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(settings.MODEL_PATH)
        self.model = Wav2Vec2Model.from_pretrained(settings.MODEL_PATH)
        self.ideal_embedding = torch.tensor(np.load(settings.EMBEDDING_PATH))
    
    def enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
        audio_data = audio_data * settings.AUDIO_GAIN
        noise_threshold = 0.01
        audio_data[np.abs(audio_data) < noise_threshold] = 0
        return audio_data
    
    def get_embedding(self, audio_data: np.ndarray, sr: int) -> torch.Tensor:
        inputs = self.processor(audio_data, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            embedding = self.model(inputs.input_values).last_hidden_state.mean(dim=1).squeeze()
        return embedding
    
    def calculate_similarity(self, embedding: torch.Tensor) -> float:
        similarity = torch.nn.functional.cosine_similarity(embedding, self.ideal_embedding, dim=0)
        return similarity.item() * 100
    
    def process_for_waveform(self, audio_data: np.ndarray, sr: int) -> Dict:
        n = max(len(audio_data) // 500, 1)
        y_reduced = audio_data[::n]
        time = np.linspace(0, len(audio_data) / sr, len(y_reduced))
        y_reduced = y_reduced / np.max(np.abs(y_reduced))
        return {"time": time.tolist(), "amplitude": y_reduced.tolist()}
    
    def process_audio_file(self, file_path: str) -> Tuple[np.ndarray, int, float]:
        y, sr = librosa.load(file_path, sr=None)
        y = self.enhance_audio(y)
        embedding = self.get_embedding(y, sr)
        similarity_score = self.calculate_similarity(embedding)
        return y, sr, similarity_score
    
    def convert_to_mp3_bytes(self, file_path: str) -> bytes:
        audio_segment = AudioSegment.from_file(file_path)
        return audio_segment.export(format="mp3").read()

class TranscriptionService:
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_file(
            settings.GOOGLE_CREDENTIALS_PATH
        )
        self.client = speech.SpeechClient(credentials=credentials)
    
    def transcribe_audio(self, audio_content: bytes, sample_rate: int) -> str:
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=sample_rate,
            language_code="ar"
        )
        response = self.client.recognize(config=config, audio=audio)
        return " ".join(result.alternatives[0].transcript for result in response.results)

class FeedbackService:
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_file(
            settings.GOOGLE_CREDENTIALS_PATH
        )
        self.tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        openai.api_key = settings.OPENAI_API_KEY
    
    def generate_feedback(self, transcription: str, similarity_score: float) -> str:
        prompt = f"""
        آپ ایک ماہر مؤذن ٹرینر ہیں جو اردو میں تربیتی مشورے دیتے ہیں۔
        
        تلاوت: {transcription}
        صحیح تلاوت: {settings.IDEAL_TEXT}
        مطابقت سکور: {similarity_score-20:.2f}%
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful Azan teacher."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message["content"].strip()
    
    def generate_audio_feedback(self, feedback_text: str) -> str:
        synthesis_input = texttospeech.SynthesisInput(text=feedback_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ur-IN",
            name="ur-IN-Wavenet-B"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            effects_profile_id=['small-bluetooth-speaker-class-device']
        )
        response = self.tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        return base64.b64encode(response.audio_content).decode('utf-8')
    
    def validate_recitation(self, transcription: str) -> bool:
        content = f"""
        Validate if this transcription matches the correct Azaan structure:
        Correct: {settings.IDEAL_TEXT}
        Transcribed: {transcription}
        """
        completion = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=512,
        )
        return "VALIDATED" in completion.choices[0].message.content

# Helper Functions
def save_upload_file_temp(upload_file: UploadFile) -> Tuple[str, str]:
    try:
        os.makedirs("temp", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"temp/audio_{timestamp}"
        original_path = f"{temp_path}_original{os.path.splitext(upload_file.filename)[1]}"
        mp3_path = f"{temp_path}.mp3"
        
        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return original_path, mp3_path
    except Exception as e:
        raise Exception(f"Failed to save file: {str(e)}")

def cleanup_temp_files(*file_paths: str) -> None:
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Failed to remove {file_path}: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="API for analyzing Azaan recitations and providing feedback"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
audio_service = AudioService()
feedback_service = FeedbackService()
transcription_service = TranscriptionService()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_recording(audio_file: UploadFile = File(...)):
    try:
        original_path, mp3_path = save_upload_file_temp(audio_file)
        try:
            audio_data, sr, similarity_score = audio_service.process_audio_file(original_path)
            audio_content = audio_service.convert_to_mp3_bytes(original_path)
            transcription = transcription_service.transcribe_audio(audio_content, sr)
            is_valid = feedback_service.validate_recitation(transcription)
            feedback_text = feedback_service.generate_feedback(transcription, similarity_score)
            waveform_data = audio_service.process_for_waveform(audio_data, sr)
            feedback_audio = feedback_service.generate_audio_feedback(feedback_text)
            
            return AnalysisResponse(
                similarity_score=similarity_score-20,
                transcription=transcription,
                validation_status="VALIDATED" if is_valid else "INVALIDATED",
                feedback=FeedbackData(
                    text=feedback_text,
                    score_category="Excellent" if similarity_score >= 90 else "Needs Improvement"
                ),
                waveform_data=WaveformData(**waveform_data),
                feedback_audio=feedback_audio
            )
        finally:
            cleanup_temp_files(original_path, mp3_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
