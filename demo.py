import os
import time
import numpy as np
import torch
import cv2
from deepface import DeepFace
from blazeface import BlazeFace
from blazebase import resize_pad, denormalize_detections
from visualization import draw_detections

import threading
import sounddevice as sd
import whisper
import scipy.io.wavfile as wav

from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama
import keyboard

llama_model = "llama3.1:8b"


os.makedirs('faces', exist_ok=True)

# Setup Torch / BlazeFace
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

back_detector = False
face_detector = BlazeFace(back_model=back_detector).to(gpu)
face_detector.load_weights("blazeface.pth")
face_detector.load_anchors("anchors_face.npy")

name = "Unknown"

known_face_encodings = []
known_face_names = []
can_record = False
time_during_unknown = 0
unknown_time_threshold = 500

cv2.namedWindow("test")
capture = cv2.VideoCapture(0)
mirror_img = True

# load previously stored faces
image_files = [os.path.join("faces", f) for f in os.listdir("faces")]

for img in image_files:      
    new_enc = DeepFace.represent(cv2.imread(img), model_name="Facenet", enforce_detection=False)
    if len(new_enc) > 0:
        known_face_encodings.append(np.array(new_enc[0]["embedding"]))
    
    img = os.path.basename(img).split("_")[0]
    known_face_names.append(img)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

model = whisper.load_model("small.en")

sample_rate = 16000  # Whisper expects 16kHz audio
chunk_duration = 5  # Capture 5 seconds at a time

frame_count = 0
recording = False

detect_faces = False
detect_speech = True
get_llama_response = False

#LLAMA

def record_voice():
    while True:
        print("record start")
        audio = sd.rec(int(sample_rate * chunk_duration), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        wav.write("temp_audio.wav", sample_rate, np.int16(audio * 32767))
        print("record end")
        
def transcribe_audio():
    global detect_faces
    while True:
        result = model.transcribe("temp_audio.wav")
        
        with open("transcription.txt", "w") as file:
            file.write(result["text"])
            
def get_llama():
    global detect_faces
    
    with open("transcription.txt", "r") as file:
        text = file.read()
    
    input_text = "You are an AI assistant for a blind person. The blind person just said the following: " + text + " . One of your features is to identify who the blind person is talking to. Is this user trying to identify another person with their command? Some example commands could be \"Who are you?\" or \"Who is this person?\". Please answer with only one word: either Yes or No"

    response = ollama.chat(model=llama_model, messages=[
        {
            'role':'user',
            'content': input_text
        }
    ])
    output_text = response['message']['content']
    print("=======================")
    print(input_text)
    print("=======================")
    print(output_text)

    if "yes" in output_text.lower():
        detect_faces = True
    
threading.Thread(target=get_llama).start()
threading.Thread(target=record_voice).start()
threading.Thread(target=transcribe_audio).start()

while hasFrame:    
    frame_count += 1
    
    if mirror_img:
        frame = np.ascontiguousarray(frame[:, ::-1, ::-1])
    else:
        frame = np.ascontiguousarray(frame[:, :, ::-1])
    
    
    
    img1, img2, scale, pad = resize_pad(frame)

    normalized_face_detections = face_detector.predict_on_image(img2)
    face_detections = denormalize_detections(normalized_face_detections, scale, pad)
    if not recording:
        draw_detections(frame, face_detections)

    for idx, det in enumerate(face_detections):
        y1, x1, y2, x2 = map(int, det[:4])
        h, w, _ = frame.shape
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))

        if x2 <= x1 or y2 <= y1:
            continue

        face_crop_bgr = frame[y1:y2, x1:x2]
        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        
        encodings = []
        if frame_count % 100 == 0 and detect_faces:
            encodings = DeepFace.represent(face_crop_rgb, model_name="Facenet", enforce_detection=False)
                
        if time_during_unknown  >= unknown_time_threshold:
            can_record = True
      
        
        if len(known_face_encodings) > 0:
            if len(encodings) > 0:
                new_enc = np.array(encodings[0]["embedding"])

                # Compare to known faces using cosine similarity
                similarities = [np.dot(new_enc, known_enc) / (np.linalg.norm(new_enc) * np.linalg.norm(known_enc))
                                for known_enc in known_face_encodings]
                
                best_idx = np.argmax(similarities)
                best_sim = similarities[best_idx]

                threshold = 0.7
                if best_sim > threshold:
                    name = known_face_names[best_idx]
                    time_during_unknown = 0
                    recording = False


                else:
                    time_during_unknown += 0.1
                    name = "Unknown"         
                    
            else:
                time_during_unknown += 0.1
               
        else:
            time_during_unknown += 0.1

        if name == "Unknown" and not recording and can_record:
            new_name = input("Enter Name: ")
            record_start_time = time.time()
            recording = True

        if recording:
            elapsed_time = time.time() - record_start_time
            cropped_face = frame[y1:y2, x1:x2]
            filename = f"faces/{new_name}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            
            success = cv2.imwrite(filename, cropped_face)
            if not success:
                print(f"Error saving image at {filename}")
            
            time.sleep(1)
            

            new_enc = DeepFace.represent(cropped_face, model_name="Facenet", enforce_detection=False)
            if len(new_enc) > 0:
                known_face_encodings.append(np.array(new_enc[0]["embedding"]))
                known_face_names.append(new_name)
                time_during_unknown = 0
            else:
                print(f"Failed to load image: {filename}")


            if elapsed_time >= 5:
                recording = False
                can_record = False
                time_when_video_over = time.time()
                name = new_name
                
                
        # Draw bounding box and label
        color = (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("test", frame[:, :, ::-1])

    hasFrame, frame = capture.read()
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
