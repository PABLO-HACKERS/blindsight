import threading
import sounddevice as sd
import wave
import numpy as np
import whisper

SAMPLE_RATE = 16000
DURATION = 5
FILENAME = "audio.wav"

def record_audio():
    """Records 5 seconds of audio and saves it as a WAV file."""
    print("Recording for 5 seconds...")
    audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()  # Wait for recording to finish

    # Save to WAV file
    with wave.open(FILENAME, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    print(f"Recording saved as {FILENAME}")

def transcribe_audio():
    """Waits for recording to finish and then transcribes the saved WAV file."""
    print("Waiting for recording to finish...")
    
    # Wait for the recording to complete
    recording_thread.join()

    print("Transcribing...")
    model = whisper.load_model("small")
    result = model.transcribe(FILENAME)
    
    print("Transcription:", result["text"])

# Start recording thread first
recording_thread = threading.Thread(target=record_audio)
recording_thread.start()

# Start transcription thread (will wait for recording_thread to finish)
transcription_thread = threading.Thread(target=transcribe_audio)
transcription_thread.start()

# Wait for both threads to finish before exiting
recording_thread.join()
transcription_thread.join()

print("Done.")
