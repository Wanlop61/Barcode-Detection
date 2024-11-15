import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("runs/detect/train3/weights/best.pt")

# Initialize the webcam
cap = cv2.VideoCapture(2)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame, stream=True)

    # Processing results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])

            # Filter out weak detections
            if confidence > 0.7:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                # Put class name and confidence on the image
                cv2.putText(frame, f'Barcode {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Webcam", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
