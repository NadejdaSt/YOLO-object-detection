from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8m.pt')

# Path to image file
image_path = r'C:\Users\Nadejda\Desktop\yolo_test_toothbrush.jpg'

# Perform object detection
results = model(image_path)

# Process and display results
for result in results:
    boxes = result.boxes  # Bounding box
    #masks = result.masks  # Masks (if applicable, e.g., for segmentation models)
    #keypoints = result.keypoints  # Keypoints (if applicable, e.g., for pose estimation)
    probs = result.probs  # Class probabilities (for classification tasks)

    # Print the detected objects and their bounding boxes
    #print(boxes)

    # To visualize the results with bounding boxes on the image:
    annotated_image = result.plot() # This returns a NumPy array with detections plotted

    # Display the image
    cv2.imshow("YOLOv8 test", annotated_image)
    cv2.waitKey(0) # Wait for a key press to close the image window
    cv2.destroyAllWindows()

    # Save the annotated image:
    #cv2.imwrite('path/to/save/annotated_image.jpg', annotated_image)