from ultralytics import YOLO
import os

def train_yolo(data_yaml_path, epochs=100, imgsz=640, batch_size=16):
    """
    Train YOLO model on the converted dataset with enhanced augmentation and hyperparameters
    
    Args:
        data_yaml_path: Path to the data.yaml file
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
    """
    # Load a YOLOv8 model
    model = YOLO('yolo11s.pt')  # load pretrained
    
    # Train the model with enhanced parameters
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name='yolo_custom',  # save results to runs/detect/yolo_custom
        device='0',  # use GPU if available
        patience=50,  # early stopping patience
        save=True,  # save checkpoints
        plots=True,  # save training plots
        
        # Optimizer settings
        lr0=0.01,  # initial learning rate
        lrf=0.01,  # final learning rate (lr0 * lrf)
        momentum=0.937,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # optimizer weight decay
        warmup_epochs=3.0,  # warmup epochs
        warmup_momentum=0.8,  # warmup initial momentum
        warmup_bias_lr=0.1,  # warmup initial bias lr
        
        # Data augmentation settings
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        degrees=10.0,  # rotation (+/- deg)
        translate=0.1,  # translation (+/- fraction)
        scale=0.5,  # scale (+/- gain)
        shear=2.0,  # shear (+/- deg)
        perspective=0.0,  # perspective (+/- fraction), range 0-0.001
        flipud=0.5,  # probability of flip up-down
        fliplr=0.5,  # probability of flip left-right
        mosaic=1.0,  # mosaic augmentation probability
        mixup=0.1,  # mixup augmentation probability
        copy_paste=0.1,  # segment copy-paste probability
        
        # Additional training configurations
        cos_lr=True,  # use cosine learning rate scheduler
        close_mosaic=10,  # disable mosaic augmentation for final epochs
        label_smoothing=0.1,  # label smoothing epsilon
        overlap_mask=True,  # masks should overlap during training
        mask_ratio=4,  # mask downsample ratio
        single_cls=False,  # train multi-class data as single-class
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