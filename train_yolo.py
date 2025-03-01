from ultralytics import YOLO
import os

def train_yolo(data_yaml_path, epochs=100, imgsz=640, batch_size=16):
    """
    Train YOLO model on the converted dataset
    
    Args:
        data_yaml_path: Path to the data.yaml file
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
    """
    # Load a YOLOv8 model
    model = YOLO('yolo11s.pt')  # load pretrained
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name='yolo_custom',  # save results to runs/detect/yolo_custom
        device='0',  # use GPU if available
        patience=50,  # early stopping patience
        save=True,  # save checkpoints
        plots=True  # save training plots
    )
    
    # Validate the model
    results = model.val()
    
    return model, results

if __name__ == '__main__':
    # Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, 'yolo_dataset', 'data.yaml')
    epochs = 100
    imgsz = 640
    batch_size = 16
    
    # Train the model
    model, results = train_yolo(data_yaml_path, epochs, imgsz, batch_size)
    print("Training completed successfully!") 