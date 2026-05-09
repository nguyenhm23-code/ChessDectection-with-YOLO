from ultralytics import YOLO
import os

class ChessDetector:
    def __init__(self, model_path, conf_threshold=0.6, iou_threshold=0.45, imgsz=640):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.model = self._load_model()

    def _load_model(self):
        print(f"Đang load model từ: {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("LỖI: Không tìm thấy file model! Hãy kiểm tra lại đường dẫn.")
        return YOLO(self.model_path)

    def predict(self, frame, stream=False):
        """Thực hiện dự đoán trên một frame hoặc ảnh tĩnh"""
        return self.model.predict(
            frame, 
            conf=self.conf_threshold, 
            agnostic_nms=True, 
            iou=self.iou_threshold, 
            imgsz=self.imgsz, 
            stream=stream, 
            verbose=False
        )

    def annotate(self, result):
        """Vẽ bounding box lên ảnh từ kết quả predict"""
        return result.plot()