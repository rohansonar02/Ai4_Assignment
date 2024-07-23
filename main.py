import cv2
import numpy as np
import os
#RohanSonar
#AI4SEE Assignment
def load_yolo():
    try:
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return net, output_layers, classes
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        exit(1)

def detect_objects(img, net, output_layers):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Filter weak detections
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], class_ids[i], confidences[i]) for i in indices]

def save_cropped_objects(img, objects, output_dir, frame_id):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    height, width = img.shape[:2]
    for i, (box, class_id, confidence) in enumerate(objects):
        x, y, w, h = box
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        if w > 0 and h > 0:
            crop_img = img[y:y+h, x:x+w]
            crop_img_path = os.path.join(output_dir, f"frame{frame_id}_obj{i}.jpg")
            cv2.imwrite(crop_img_path, crop_img)
            print(f"Saved cropped object to {crop_img_path}")
        else:
            print(f"Invalid bounding box for object {i} in frame {frame_id}: {box}")

def process_video(input_video_path, output_dir):
    net, output_layers, classes = load_yolo()
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Processing frame {frame_id}")
        objects = detect_objects(frame, net, output_layers)
        if objects:
            save_cropped_objects(frame, objects, output_dir, frame_id)
        else:
            print(f"No objects detected in frame {frame_id}")
        frame_id += 1
    cap.release()
    print(f"Finished processing video. Total frames: {frame_id}")


input_video_path = '/Users/rohansonar/Desktop/Ai4See Assignment/input.mp4'
output_dir = '/Users/rohansonar/Desktop/Ai4See Assignment/Output'
process_video(input_video_path, output_dir)