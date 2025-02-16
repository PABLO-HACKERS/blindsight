import os
import time
import numpy as np
import torch
import cv2
from deepface import DeepFace
from blazeface import BlazeFace
from blazebase import resize_pad, denormalize_detections
from visualization import draw_detections

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
unknown_time_threshold = 5

cv2.namedWindow("test")
capture = cv2.VideoCapture(0)
mirror_img = True

# load previously stored faces
image_files = [os.path.join("faces", f) for f in os.listdir("faces")]

for img in image_files:  
    print(img)
    cv2.imshow("prev images", cv2.imread(img))
    cv2.waitKey(500)
    
    new_enc = DeepFace.represent(cv2.imread(img), model_name="Facenet", enforce_detection=False)
    if len(new_enc) > 0:
        known_face_encodings.append(np.array(new_enc[0]["embedding"]))
    
    img = os.path.basename(img).split("_")[0]
    known_face_names.append(img)
    print(img)
    
if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

frame_count = 0
recording = False

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

        encodings = DeepFace.represent(face_crop_rgb, model_name="Facenet", enforce_detection=False)
                
        if time_during_unknown  >= unknown_time_threshold:
            can_record = True
        
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
                    recording = False

                    # print(time_during_unknown)

                else:
                    time_during_unknown += 0.1
                    name = "Unknown"                    
                    
            else:
                time_during_unknown += 0.1
                name = "Unknown"
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
            print(f"Recording new face at: {filename}")
            
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

            print(f"Elapsed Time: {elapsed_time:.2f} seconds")

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
