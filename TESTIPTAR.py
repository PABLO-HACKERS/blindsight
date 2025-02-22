import sounddevice as sd
import numpy as np
import whisper
import scipy.io.wavfile as wav
import cv2

# Load Whisper model
model = whisper.load_model("small.en")

# Recording settings
sample_rate = 16000  # Whisper expects 16kHz audio
chunk_duration = 5  # Capture 5 seconds at a time

while True:
    print("Listening for speech...")
    audio = sd.rec(int(sample_rate * chunk_duration), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()  # Wait until recording finishes

    # Save to a temporary file

    wav.write("temp_audio.wav", sample_rate, np.int16(audio * 32767))

    
    # Transcribe the audio chunk
    result = model.transcribe("temp_audio.wav")
    print("Transcription:", result["text"])
