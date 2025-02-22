import time
import threading
import sounddevice as sd
import numpy as np
import whisper
import scipy.io.wavfile as wav
import cv2

model = whisper.load_model("small.en")

sample_rate = 16000
chunk_duration = 5
filepath = "temp_audio.wav"

def record_voice():
    while True:
        audio = sd.rec(int(sample_rate * chunk_duration), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        wav.write(filepath, sample_rate, np.int16(audio * 32767))
        print("done recording")
        
def transcribe_audio():
    result = model.transcribe(filepath)
    print("Transcription:", result["text"])
        
threading.Thread(target=record_voice).start()
while True:
    result = model.transcribe(filepath)
    print("Transcription:", result["text"])
