import os
import time
import numpy as np
import torch
import cv2
from deepface import DeepFace
from blazeface import BlazeFace
from blazebase import resize_pad, denormalize_detections
from visualization import draw_detections

# Ensure faces directory exists
os.makedirs('faces', exist_ok=True)
known_face_encodings = []
known_face_names = []
# Setup Torch / BlazeFace
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

back_detector = False
face_detector = BlazeFace(back_model=back_detector).to(gpu)
face_detector.load_weights("blazeface.pth")
face_detector.load_anchors("anchors_face.npy")

canRecord = False
time_during_unknown = 0

# OpenCV Capture
cv2.namedWindow("test")
capture = cv2.VideoCapture(0)
mirror_img = True

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

frame_count = 0
name = "Unknown"
recording = False
video_writer = None
record_start_time = None

while hasFrame:
    frame_count += 1
    if mirror_img:
        frame = np.ascontiguousarray(frame[:, ::-1, ::-1])
    else:
        frame = np.ascontiguousarray(frame[:, :, ::-1])
    
    # Resize / Pad for BlazeFace
    img1, img2, scale, pad = resize_pad(frame)

    # Face Detection
    normalized_face_detections = face_detector.predict_on_image(img2)
    face_detections = denormalize_detections(normalized_face_detections, scale, pad)
    if not recording:
        draw_detections(frame, face_detections)
  
    # Recognize Faces
    for idx, det in enumerate(face_detections):
        y1, x1, y2, x2 = map(int, det[:4])
        h, w, _ = frame.shape
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))

        if x2 <= x1 or y2 <= y1:
            continue

        # Crop face
        face_crop_bgr = frame[y1:y2, x1:x2]
        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

        # Encode using FaceNet
        encodings = DeepFace.represent(face_crop_rgb, model_name="Facenet", enforce_detection=False)
        
        if time_during_unknown  >= 5:
            canRecord = True
        
        if frame_count % 20 == 0 and len(known_face_encodings) > 0:
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
                    print(time_during_unknown)
                    recording = False
                else:
                    time_during_unknown += 0.05
                    name = "Unknown"
                    
                    
                    
            else:
                time_during_unknown += 0.05
                name = "Unknown"
        else:
            time_during_unknown += 0.05
        if name == "Unknown" and not recording and canRecord:
            name = input("Please Enter your Name: ")
            random_filename = f"videos/{name}.mp4"
            record_start_time = time.time()
            recording = True
            print(f"Recording unknown face: {random_filename}")

        if recording:
            elapsed_time = time.time() - record_start_time
            cropped_face = frame[y1:y2, x1:x2]
            filename = f"faces/{name}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            
            # Save the face as JPEG
            success = cv2.imwrite(filename, cropped_face)
            if not success:
                print(f"Error saving image at {filename}")
            
            # Add delay to ensure the file is written
            time.sleep(1)
            
            # Load and process saved image
            new_img = cv2.imread(filename)
            if new_img is not None:
                new_enc = DeepFace.represent(new_img, model_name="Facenet", enforce_detection=False)
                if len(new_enc) > 0:
                    known_face_encodings.append(np.array(new_enc[0]["embedding"]))
                    known_face_names.append(name)
                    time_during_unknown = 0
            else:
                print(f"Failed to load image: {filename}")

            print(f"Elapsed Time: {elapsed_time:.2f} seconds")

            if elapsed_time >= 5:
                recording = False
                canRecord = False
                time_when_video_over = time.time()
                
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
