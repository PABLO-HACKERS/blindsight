import os
import sys
import time
import json
import threading
import asyncio
import numpy as np
import torch
import cv2
import pygame
import sounddevice as sd
import whisper
import scipy.io.wavfile as wav
from deepface import DeepFace
from blazeface import BlazeFace
from blazebase import resize_pad, denormalize_detections
from visualization import draw_detections
from langchain_ollama import OllamaLLM
import edge_tts

# PyQt imports and custom GUI files
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QLabel, QLineEdit, QPushButton, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt, QObject, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from interface import Ui_MainWindow  
from Custom_Widgets import *

pygame.mixer.init()

class ChatBubble(QLabel):
    def __init__(self, text, is_sender=False):
        super().__init__(text)
        self.setWordWrap(True)
        self.setMaximumWidth(250)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        # Set different background colors for sender and receiver
        if is_sender:
            self.setStyleSheet("""
                QLabel {
                    background-color: #FFFFFF;
                    color: #000000;
                    border-radius: 10px;
                    padding: 8px;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    background-color: #FFA500;  /* Orange background */
                    color: #000000;
                    border-radius: 10px;
                    padding: 8px;
                }
            """)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        loadJsonStyle(self, self.ui)

        # --- Setup Camera & Timer ---
        self.capture = cv2.VideoCapture(1)
        if not self.capture.isOpened():
            print("Error: Cannot open camera")
            sys.exit()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 fps

        # --- Initialize Variables (from the original script) ---
        self.llama_model = OllamaLLM(model="llama3.1:8b", base_url="http://localhost:11434")
        self.CACHE_FILE = "cache.json"
        if os.path.exists(self.CACHE_FILE):
            with open(self.CACHE_FILE, "r") as f:
                try:
                    self.cache = json.load(f)
                except json.JSONDecodeError:
                    self.cache = {}
        else:
            self.cache = {}

        os.makedirs('faces', exist_ok=True)
        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        self.back_detector = False
        self.face_detector = BlazeFace(back_model=self.back_detector).to(self.gpu)
        self.face_detector.load_weights("blazeface.pth")
        self.face_detector.load_anchors("anchors_face.npy")

        self.name = "Unknown"
        self.known_face_encodings = []
        self.known_face_names = []
        self.can_record = False
        self.time_during_unknown = 0
        self.unknown_time_threshold = 30
        self.mirror_img = True
        self.frame_count = 0
        self.recording = False
        self.detect_faces = False
        self.detect_speech = True
        self.detect_objects = False
        self.can_record_transcribe = True
        self.already_played_name = False
        self.create_new_name = False
        self.new_name = ""
        self.record_start_time = 0
        
        self.message_to_SR = ""
        self.s_or_r = 0

        # Threading events
        self.recording_done_event = threading.Event()
        self.recording_done_event.clear()
        self.create_new_name_event = threading.Event()
        self.recording_new_name_event = threading.Event()
        
        self.ui.settingsBtn.clicked.connect(lambda:self.ui.centerMenuContainer.expandMenu())
        self.ui.informationBtn.clicked.connect(lambda:self.ui.centerMenuContainer.expandMenu())
        self.ui.helpBtn.clicked.connect(lambda:self.ui.centerMenuContainer.expandMenu())

        self.ui.closeCenterMenuBtn.clicked.connect(lambda:self.ui.centerMenuContainer.collapseMenu())

        self.ui.moreBtn.clicked.connect(lambda:self.ui.rightMenuContainer.expandMenu())
        self.ui.profileBtn.clicked.connect(lambda:self.ui.rightMenuContainer.expandMenu())

        self.ui.closeRightMenuBtn.clicked.connect(lambda:self.ui.rightMenuContainer.collapseMenu())
        
        # After self.ui.setupUi(self) and any style loading, etc.
        # Hide or remove the existing label if it’s not needed:
        self.ui.label_9.hide()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        # Get the existing layout, if any
        layout = self.ui.page_2.layout()
        if layout is None:
            layout = QVBoxLayout(self.ui.page_2)
            self.ui.page_2.setLayout(layout)
            
        # Now add your scroll area to the layout
        layout.addWidget(self.scroll_area)

        
        
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10)
        self.scroll_area.setWidget(self.chat_container)

        # --- Load previously stored faces ---
        image_files = [os.path.join("faces", f) for f in os.listdir("faces")]
        for img_path in image_files:
            img = cv2.imread(img_path)
            new_enc = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)
            if len(new_enc) > 0:
                self.known_face_encodings.append(np.array(new_enc[0]["embedding"]))
            base = os.path.basename(img_path)
            face_name = base.split("_")[0]
            self.known_face_names.append(face_name)

        # --- Load whisper model ---
        self.whisper_model = whisper.load_model("small.en")
        self.sample_rate = 16000
        self.chunk_duration = 5

        # --- YOLOv3 Setup ---
        self.net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        with open("yolo/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # --- Start Audio Threads ---
        threading.Thread(target=self.record_voice, daemon=True).start()
        threading.Thread(target=self.transcribe_audio, daemon=True).start()

        self.show()

    def add_chat_bubble(self, text, is_sender=False):
        """Add a chat bubble to the chat container."""
        bubble = ChatBubble(text, is_sender)
        
        # Create a horizontal layout to align bubble left or right
        bubble_layout = QHBoxLayout()
        if is_sender:
            bubble_layout.addStretch()
            bubble_layout.addWidget(bubble)
        else:
            bubble_layout.addWidget(bubble)
            bubble_layout.addStretch()

        self.chat_layout.addLayout(bubble_layout)
        # Auto-scroll to the bottom
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def send_message(self, message):
        """Handle sending a message."""
        self.add_chat_bubble("You: " + message, is_sender=True)

    def receive_message(self, message):
        """Handle receiving a message (left-aligned)."""
        self.add_chat_bubble("Pablo.AI: " + message, is_sender=False)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            print("Failed to capture frame")
            return
        
        if self.s_or_r != 0:
            self.s_or_r = 0
            self.send_message(self.message_to_SR)
            self.message_to_SR = ""

        self.frame_count += 1
        if self.mirror_img:
            frame = np.ascontiguousarray(frame[:, ::-1, :])
        else:
            frame = np.ascontiguousarray(frame[:, :, ::-1])

        # --- Object Detection (if flagged) ---
        if self.detect_objects:
            self.detect_objects = False
            cv2.imwrite("input.jpg", frame)
            self.run_object_detection()

        # --- Face Detection & Recognition ---
        img1, img2, scale, pad = resize_pad(frame)
        normalized_face_detections = self.face_detector.predict_on_image(img2)
        face_detections = denormalize_detections(normalized_face_detections, scale, pad)
        if not self.recording:
            draw_detections(frame, face_detections)

        for det in face_detections:
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
            if self.frame_count % 100 == 0:
                encodings = DeepFace.represent(face_crop_rgb, model_name="Facenet", enforce_detection=False)
                

            if self.time_during_unknown >= self.unknown_time_threshold:
                self.can_record = True

            if self.known_face_encodings:
                if encodings:
                    new_enc = np.array(encodings[0]["embedding"])
                    similarities = [
                        np.dot(new_enc, known_enc) / (np.linalg.norm(new_enc) * np.linalg.norm(known_enc))
                        for known_enc in self.known_face_encodings
                    ]
                    best_idx = np.argmax(similarities)
                    best_sim = similarities[best_idx]
                    threshold = 0.7
                    if best_sim > threshold:
                        self.name = self.known_face_names[best_idx]
                        self.recording = False
                        if not self.already_played_name and self.detect_faces:
                            asyncio.run(self.speak(f"{self.name} is in front of you"))
                            self.receive_message(f"{self.name} is in front of you")
                            sound = pygame.mixer.Sound("output.wav")
                            sound.play()
                            pygame.time.delay(int(sound.get_length() * 1000))
                            self.already_played_name = True
                    else:
                        self.time_during_unknown += 0.1
                        self.name = "Unknown"
                        self.already_played_name = False
                else:
                    self.time_during_unknown += 0.1
            else:
                self.time_during_unknown += 0.1

            if self.name == "Unknown" and self.detect_faces:
                asyncio.run(self.speak("new person detected, what is their name?"))
                self.receive_message("new person detected, what is their name?")
                sound = pygame.mixer.Sound("output.wav")
                sound.play()
                pygame.time.delay(int(sound.get_length() * 1000 + 1000))
                self.create_new_name = True
                self.create_new_name_event.wait()
                self.create_new_name_event.clear()
                self.record_start_time = time.time()
                self.recording = True

            if self.recording:
                elapsed_time = time.time() - self.record_start_time
                cropped_face = frame[y1:y2, x1:x2]
                filename = f"faces/{self.new_name}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                success = cv2.imwrite(filename, cropped_face)
                if not success:
                    print(f"Error saving image at {filename}")
                time.sleep(1)
                new_enc = DeepFace.represent(cropped_face, model_name="Facenet", enforce_detection=False)
                if new_enc:
                    self.known_face_encodings.append(np.array(new_enc[0]["embedding"]))
                    self.known_face_names.append(self.new_name)
                    self.time_during_unknown = 0
                else:
                    print(f"Failed to load image: {filename}")
                if elapsed_time >= 5:
                    self.recording = False
                    self.can_record = False
                    self.name = self.new_name

            # Draw the bounding box and label on the frame
            color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, self.name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.ui.label_10.setPixmap(QPixmap.fromImage(q_img))

    def record_voice(self):

        while True:
            print("record start")
            audio = sd.rec(int(self.sample_rate * self.chunk_duration),
                           samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            if self.create_new_name:
                wav.write("name_audio.wav", self.sample_rate, np.int16(audio * 32767))
                self.recording_new_name_event.set()
                self.recording_done_event.set()
            else:
                wav.write("temp_audio.wav", self.sample_rate, np.int16(audio * 32767))
                self.recording_done_event.set()
            print("record end")

    def transcribe_audio(self):
        while True:
            self.recording_done_event.wait()
            if self.create_new_name:
                self.recording_new_name_event.wait()
                self.recording_new_name_event.clear()
                result = self.whisper_model.transcribe("name_audio.wav")
                user_command = result["text"]
                print(user_command)
                extracted_name = self.llama_model.invoke(
                    input=f"Extract the valid English name from the following text: '{user_command}'. If there is a name, respond with just the name. If there is no name, respond with 'No'.")
                if "No" not in extracted_name:
                    self.message_to_SR = extracted_name
                    self.s_or_r = 1

                    self.create_new_name = False
                    self.new_name = extracted_name
                    self.create_new_name_event.set()
                else:
                    print("Invalid name")
            else:
                result = self.whisper_model.transcribe("temp_audio.wav")
                user_command = result["text"]
                if len(user_command) < 5 or "thank you" in user_command.lower():
                    print("jepti")
                    self.recording_done_event.clear()
                    continue
                output_text = ""
                if user_command in self.cache:
                    output_text = self.cache[user_command]
                else:
                    input_text = (
                        f"You are an AI assistant for a blind person. The user said: \"{user_command}\". "
                        "Your task is to determine if they are specifically asking for facial recognition or object identification. "
                        "If the user asks about identifying a person, such as \"Who is this person?\" or \"Who am I talking to?\", respond with \"face\". "
                        "If the user asks about objects in their surroundings, such as \"What is in front of me?\" or \"What am I holding?\", respond with \"object\". "
                        "If the user request does not fall into these categories, respond with \"no\". "
                        "Respond with only one word: face, object, or no."
                    )
                    output_text = self.llama_model.invoke(input=input_text)
                    self.cache[user_command] = output_text
                    if "no" not in output_text.lower():
                        with open(self.CACHE_FILE, "w") as f:
                            json.dump(self.cache, f, indent=4)
                    print(input_text)
                print(output_text)
                if "face" in output_text.lower():
                    self.detect_faces = True
                    self.message_to_SR = user_command
                    self.s_or_r = 1
                elif "object" in output_text.lower():
                    self.detect_objects = True
                    self.message_to_SR = user_command
                    self.s_or_r = 1

            self.recording_done_event.clear()

    async def speak(self, text_to_say):
        communicate = edge_tts.Communicate(text_to_say, "en-US-JennyNeural")
        await communicate.save("output.wav")

    def run_object_detection(self):
        image = cv2.imread("input.jpg")
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        conf_threshold = 0.5
        nms_threshold = 0.4
        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x, center_y, w_box, h_box = (detection[:4] * np.array([width, height, width, height])).astype(int)
                    x = center_x - w_box // 2
                    y = center_y - h_box // 2
                    boxes.append([x, y, w_box, h_box])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        detected_objects = []
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            label = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
            detected_objects.append(self.classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("YOLOv3 Detection", image)
        output_string = ", ".join(detected_objects)
        asyncio.run(self.speak(f"There is a {output_string} in front of you"))
        self.receive_message(f"There is a {output_string} in front of you")
        sound = pygame.mixer.Sound("output.wav")
        sound.play()
        pygame.time.delay(int(sound.get_length() * 1000))

    def closeEvent(self, event):
        if self.capture.isOpened():
            self.capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
