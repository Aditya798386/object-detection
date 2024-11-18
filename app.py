import torch
import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model (replace 'yolov8n.pt' with your model path if needed)
model = YOLO('yolov8n.pt')  # Adjust this if you're using a custom YOLO model

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop for real-time object detection
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO object detection on the frame
    results = model(frame)

    # Initialize counters for humans and objects
    human_count = 0
    total_objects = 0

    # Process each detection result
    for result in results:
        # Access each bounding box in the result
        for box in result.boxes:
            # Get the bounding box coordinates, confidence score, and class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert to integers
            confidence = box.conf[0].item()                  # Convert confidence to float
            class_id = int(box.cls[0].item())                # Convert class ID to integer
            
            # Get the class label and format label text
            label = model.names[class_id]
            label_text = f"{label} {confidence:.2f}"

            # Check if the detected object is a human
            if label.lower() == 'person':  # Assuming 'person' is the class name for humans
                human_count += 1
            total_objects += 1

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

            # Draw the label above the bounding box
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)  # Green label text

    # Display the total counts on the frame
    cv2.putText(frame, f'Total Humans: {human_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(frame, f'Total Objects: {total_objects}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
