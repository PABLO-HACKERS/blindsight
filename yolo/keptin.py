import cv2
import numpy as np

net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Read input image
image = cv2.imread("input.jpg")
height, width, _ = image.shape

# Convert image to blob and feed to model
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Run forward pass
outputs = net.forward(output_layers)

# Parse detections
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

# Apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw bounding boxes
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    color = (0, 255, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
cv2.imshow("YOLOv3 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
