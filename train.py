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
        patience=30
    )

    print(f"--- ĐÃ LƯU MODEL TẠI: {results.save_dir} ---")

if __name__ == '__main__':
    train_chess_model()