import numpy as np
import torch
import cv2
import sys
import face_recognition  # <-- new import

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from visualization import draw_detections

# -----------------------------
# Known Faces
# -----------------------------
known_face_encodings = []
known_face_names = []

# Example: Alice
alice_img = face_recognition.load_image_file("alice.jpg")
alice_enc = face_recognition.face_encodings(alice_img)
if len(alice_enc) > 0:
    known_face_encodings.append(alice_enc[0])
    known_face_names.append("Enrique")



# -----------------------------
# Torch / BlazeFace Setup
# -----------------------------
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

back_detector = False
face_detector = BlazeFace(back_model=back_detector).to(gpu)
face_detector.load_weights("blazeface.pth")
face_detector.load_anchors("anchors_face.npy")

# -----------------------------
# OpenCV Capture
# -----------------------------
cv2.namedWindow("test")
capture = cv2.VideoCapture(0)  # or a video file
mirror_img = True


if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False
frame_count = 0
name = "Unknown"
while hasFrame:
    frame_count += 1
    # Mirror + channel flip (if that's what you want)
    if mirror_img:
        frame = np.ascontiguousarray(frame[:, ::-1, ::-1])
    else:
        frame = np.ascontiguousarray(frame[:, :, ::-1])
    
    # 1) Resize / Pad for BlazeFace
    img1, img2, scale, pad = resize_pad(frame)

    # 2) Face Detection (using front detector => use img2)
    normalized_face_detections = face_detector.predict_on_image(img2)
    face_detections = denormalize_detections(normalized_face_detections, scale, pad)

    # 3) Draw bounding boxes
    draw_detections(frame, face_detections)
  
    # 4) For each bounding box => Crop & Recognize
    for idx, det in enumerate(face_detections):
        # [y1, x1, y2, x2, score]
        y1, x1, y2, x2 = map(int, det[:4])

        # Clamp coords
        h, w, _ = frame.shape
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))

        if x2 <= x1 or y2 <= y1:
            continue

        # Crop the face from 'frame' (currently BGR or reversed?)
        face_crop_bgr = frame[y1:y2, x1:x2]

        # face_recognition expects RGB
        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(face_crop_rgb)
        if frame_count % 20 == 0:
            if len(encodings) > 0:
                new_enc = encodings[0]

                # Compare to known faces
                # Option A: face_distance
                distances = face_recognition.face_distance(known_face_encodings, new_enc)
                best_idx = np.argmin(distances)
                best_dist = distances[best_idx]

                threshold = 0.6  # typical default
                if best_dist < threshold:
                    name = known_face_names[best_idx]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

        # Put text on bounding box in 'frame'
        # NOTE: Because frame might be channel-reversed, if you see weird colors,
        # you might revert it later with frame[:,:,::-1].
        color = (0, 255, 255)  # for demonstration
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 5) Show the final
    cv2.imshow("test", frame[:, :, ::-1])  # revert channels for display if needed

    # 6) Next frame
    hasFrame, frame = capture.read()
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
