import cv2
import torch
import numpy as np
from pathlib import Path
import sys
from ultralytics import YOLO

def load_model(weights_path):
    """Load the YOLO model."""
    model = YOLO(weights_path)  # Load the model using provided weights path
    return model

def draw_predictions(image, predictions, class_names):
    """Draw bounding boxes and labels on the image."""
    img = image.copy()
    num_classes = len(class_names)
    
    # Colors for visualization (one color per class)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    
    # Process results from ultralytics format
    if len(predictions) > 0:
        for r in predictions:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                # Convert to integers for drawing
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_id = int(cls)
                
                # Skip if class_id is invalid
                if class_id >= num_classes:
                    print(f"Warning: Detected invalid class ID {class_id}, skipping...")
                    continue
                
                # Get color for this class
                color = colors[class_id].tolist()
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{class_names[class_id]} ({conf:.2f})"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def predict_image(model, image_path, class_names, conf_threshold=0.25):
    """Run prediction on a single image."""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None
    
    # Run inference
    predictions = model.predict(img, conf=conf_threshold)
    
    # Draw predictions
    result_img = draw_predictions(img, predictions, class_names)
    
    return result_img, predictions

def main():
    # Configuration
    weights_path = "runs/detect/yolo_custom/weights/best.pt"  # Path to your trained weights
    test_image_path = "94.jpg"  # Single test image
    output_dir = "predictions"  # Directory to save results
    conf_threshold = 0.25  # Confidence threshold for detections
    
    # Class names (same as in training)
    class_names = ['buyer_name_thai', 
                  'buyer_name_eng',
                  'seller_name_thai',
                  'seller_name_eng',
                  'buyer_vat_number',
                  'seller_vat_number',
                  'total_due_amount']
    
    num_classes = len(class_names)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(weights_path)
    
    # Process single test image
    test_path = Path(test_image_path)
    if not test_path.exists():
        print(f"Error: Test image {test_image_path} not found")
        return
    
    print(f"Processing {test_path.name}...")
    
    # Run prediction
    result_img, predictions = predict_image(model, test_path, class_names, conf_threshold)
    
    if result_img is not None:
        # Save result
        output_file = output_path / f"pred_{test_path.name}"
        cv2.imwrite(str(output_file), result_img)
        print(f"Saved prediction to {output_file}")
        
        # Print detections
        print("Detections:")
        if len(predictions) > 0:
            for r in predictions:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    class_id = int(cls)
                    if class_id < num_classes:
                        print(f"- {class_names[class_id]}: {conf:.2f}")
                    else:
                        print(f"- Unknown class {class_id}: {conf:.2f}")
        else:
            print("No detections found")

if __name__ == "__main__":
    main() 