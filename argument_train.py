from ultralytics import YOLO
import os

def train_chess_model():
    model = YOLO("yolo11s.pt")

    custom_save_path = r"D:\Project\Lab2\25-04-2026"
    experiment_name = "Chess_Project"

    results = model.train(
        data= r"D:\Project\Lab2\25-04-2026\Dataset\data.yaml",
      
        project=custom_save_path,  
        name=experiment_name,  
        
        exist_ok=True,            
        epochs=300,
        imgsz=640,
        batch=16,
        device="0",
        plots=True,
        save=True,
        patience=30,

    # --- Data Augmentation Settings ---
        degrees=20.0,      # Rotate image within +/- 20 degrees
        translate=0.1,     # Translate image horizontally/vertically by 10%
        scale=0.5,         # Scale image by +/- 50% (crucial for robot height changes)
        shear=2.0,         # Shear image by +/- 2 degrees
        perspective=0.001, # Perspective transformation for angled views
        hsv_h=0.015,       # HSV Hue variation
        hsv_s=0.7,         # HSV Saturation variation
        hsv_v=0.4,         # HSV Brightness variation (important for lab lighting)
        flipud=0.0,        # Vertical flip (keep at 0.0 unless camera is strictly top-down)
        fliplr=0.5,        # Horizontal flip (50% probability)
        mosaic=1.0,        # Mosaic augmentation (combines 4 images into one)
        mixup=0.1,         # Mixup (blends two images to increase complexity)
        blur=0.1           # Simulate camera motion blur during robot movement
    )

    print(f"--- ĐÃ LƯU MODEL TẠI: {results.save_dir} ---")

if __name__ == '__main__':
    train_chess_model()