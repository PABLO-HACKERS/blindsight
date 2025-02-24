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

from langchain_ollama import OllamaLLM
import edge_tts
import asyncio
import pygame
import json

pygame.mixer.init()

llama_model = OllamaLLM(model="llama3.1:8b", base_url="http://localhost:11434")

CACHE_FILE = "cache.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        try:
            cache = json.load(f)
        except json.JSONDecodeError:
            cache = {}
else:
    cache = {}


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
unknown_time_threshold = 30

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

sample_rate = 16000
chunk_duration = 5

frame_count = 0
recording = False

detect_faces = False
detect_speech = True
detect_objects = False
get_llama_response = False
can_record_transcribe = True
already_played_name = False

create_new_name = False
new_name = ""

recording_done_event = threading.Event() 
recording_done_event.clear()

create_new_name_event = threading.Event()

recording_new_name_event = threading.Event()

"""
YOLOV3 MODEL SETUP
"""

net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


"""FUNCTIONS FOR THREADING"""
def record_voice():
    
    global can_record_transcribe
    
    while True:
        global create_new_name
        
        print("record start")
        audio = sd.rec(int(sample_rate * chunk_duration), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        if create_new_name:
            wav.write("name_audio.wav", sample_rate, np.int16(audio * 32767))
            recording_new_name_event.set()
        else:
            wav.write("temp_audio.wav", sample_rate, np.int16(audio * 32767))
            recording_done_event.set()

        print("record end")
        
        
def transcribe_audio():
    
    global detect_faces
    global detect_objects
    global create_new_name
    global new_name
    
    while True:           
        
        recording_done_event.wait()

        if create_new_name:
            
            recording_new_name_event.wait()
            recording_new_name_event.clear()
            
            result = model.transcribe("name_audio.wav")
            user_command = result["text"]            
            
                        
            extracted_name = llama_model.invoke(input=f"Extract the valid English name from the following text: '{user_command}'. If there is a name, respond with just the name. If there is no name, respond with 'No'.")
        
            if "No" not in extracted_name:
                print(f"new person saved under name: {extracted_name}")
                create_new_name = False
                new_name = extracted_name
                create_new_name_event.set()
            else:
                print("Invalid name")
        else:
                       
            result = model.transcribe("temp_audio.wav")
            user_command = result["text"]
            
            
            if len(user_command) < 5 or "thank you" in user_command.lower():
                print("jepti")
                continue
            
            output_text = ""
            if user_command in cache:
                output_text = cache[user_command]
            else:
                input_text = (
                    f"You are an AI assistant for a blind person. The user said: \"{user_command}\". "
                    "Your task is to determine if they are specifically asking for facial recognition or object identification. "
                    "If the user asks about identifying a person, such as \"Who is this person?\" or \"Who am I talking to?\", respond with \"face\". "
                    "If the user asks about objects in their surroundings, such as \"What is in front of me?\" or \"What am I holding?\", respond with \"object\". "
                    "If the user request does not fall into these categories, respond with \"no\". "
                    "Respond with only one word: face, object, or no."
                )

                output_text = llama_model.invoke(input=input_text)
                
                cache[user_command] = output_text
                if "no" not in output_text.lower():
                    with open(CACHE_FILE, "w") as f:
                        json.dump(cache, f, indent=4)
                    
                print("=======================")
                print(input_text)
                print("=======================")
            print(output_text)

            if "face" in output_text.lower():
                detect_faces = True
            elif "object" in output_text.lower():
                detect_objects = True
      
"""OTHER FUNCTIONS"""      
async def speak(text_to_say):
    communicate = edge_tts.Communicate(text_to_say, "en-US-JennyNeural")
    await communicate.save("output.wav")    
    
def run_object_detection():
    image = cv2.imread("input.jpg")
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    outputs = net.forward(output_layers)

    conf_threshold = 0.5
    nms_threshold = 0.4
    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype(int)
                x, y = center_x - w // 2, center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    detected_objects = []
    
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        detected_objects.append(classes[class_ids[i]])
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if label == "person":
            label = "stud"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("YOLOv3 Detection", image)
    print(detected_objects)
    output_string = ""
    for text in detected_objects:
        output_string += text + ", "
    asyncio.run(speak(f"{output_string}"))
    sound = pygame.mixer.Sound("output.wav")
    sound.play()
    pygame.time.delay(int(sound.get_length() * 1000))  
        
    
threading.Thread(target=record_voice).start()
threading.Thread(target=transcribe_audio).start()

while hasFrame:    

    if name != "Unknown" or detect_faces == False:
        time_during_unknown = 0
                
    frame_count += 1
    
    
    
    if mirror_img:
        frame = np.ascontiguousarray(frame[:, ::-1, ::-1])
    else:
        frame = np.ascontiguousarray(frame[:, :, ::-1])
    
    if detect_objects:
        detect_objects = False
        cv2.imwrite("input.jpg", frame)
        run_object_detection()
    
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
        if (frame_count % 100 == 0):
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
                    recording = False
                    if not already_played_name and detect_faces:
                        asyncio.run(speak(f"{name} is in front of you"))
                        sound = pygame.mixer.Sound("output.wav")
                        sound.play()
                        pygame.time.delay(int(sound.get_length() * 1000))              
                        already_played_name = True
                else:
                    time_during_unknown += 0.1
                    name = "Unknown"      
                    already_played_name = False   
                    
            else:
                time_during_unknown += 0.1
               
        else:
            time_during_unknown += 0.1

        if name == "Unknown" and not recording and can_record and detect_faces:
            
            asyncio.run(speak(f"new person detected, what is their name?"))
            sound = pygame.mixer.Sound("output.wav")
            sound.play()
            pygame.time.delay(int(sound.get_length() * 1000 + 1000))    
            
            create_new_name = True
            create_new_name_event.wait()
            create_new_name_event.clear()
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
