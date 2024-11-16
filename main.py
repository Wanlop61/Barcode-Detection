import cv2
from pyzbar.pyzbar import decode
from ultralytics import YOLO
from kraken import binarization
from PIL import Image
import numpy as np

BARCODE_DETECTED_NUM = 0
BARCODE_DECODED_NUM = 0

def preprocessing_image(image):
    """
    This function preprocess the image before passing it to PyZbar for decoding.
    We convert it to grayscale, apply binarization, and enchance contrast.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image (use Kraken's binarization method)
    # Convert to PIL image before applying Kraken binarization
    pil_image = Image.fromarray(gray)
    bw_im = binarization.nlbin(pil_image)
    bw_im = np.array(bw_im)

    # Apply some additional thresholding for better results
    _, thresh = cv2.threshold(bw_im, 127, 255, cv2.THRESH_BINARY)

    return thresh

def decodeBarcode(image):
    global BARCODE_DECODED_NUM
    barcodes = decode(preprocessing_image(image))
    if not barcodes:
        print("No barcode decode")
    else:
        print("Barcodes = ", barcodes)
        BARCODE_DECODED_NUM += 1

def videoCapture(index):
    # Initialize the webcam
    cap = cv2.VideoCapture(index)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If couldn't access webcam, break the loop
        if not ret:
            print("Failed to capture image from webcam.")
            break

        # Call detect barcode function
        frame = detectBarcode(frame)

        # Display the resulting frame
        displayImage(frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def detectBarcode(frame):
    # Declare for count the number of detected barcode
    barcodes_num = 0
    # Load the YOLO model
    model = YOLO("runs/detect/train3/weights/best.pt")

    # Perform object detection
    results = model(frame, stream=True)

    # Processing results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]

            # Filter out weak detections
            if confidence > 0.7:
                # Count the number of detected barcode
                barcodes_num += 1
                # Crop image
                crop_image = frame[y1:y2, x1:x2]
                # Decode barcode
                decodeBarcode(crop_image)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                # Put class name and confidence on the image
                cv2.putText(frame, f'Barcode {barcodes_num} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Set the BARCODE_NUM
    global BARCODE_DETECTED_NUM
    BARCODE_DETECTED_NUM = barcodes_num

    # Return result of detection
    return frame

def loadImage(image_path):
    # Read the image
    frame = cv2.imread(image_path)

    # If couldn't access image, return
    if frame is None:
        print(f"Failed to load image from {image_path}.")
        return
    
    # Perform object detection by call detect barcode function
    frame = detectBarcode(frame)

    while True:
        # Display the resulting frame
        displayImage(frame)

        # Break the loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def displayImage(frame):
    # Display the resulting frame
    cv2.imshow("Barcode Detection", frame)


if __name__ == "__main__":
    # videoCapture(2)
    loadImage('image/barcode4.png') 
    print(f"Decoded {BARCODE_DECODED_NUM} of {BARCODE_DETECTED_NUM}")