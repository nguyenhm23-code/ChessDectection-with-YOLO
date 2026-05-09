import cv2
import os
from core_detector import ChessDetector

class ImageBatchProcessor:
    def __init__(self, detector: ChessDetector, input_dir, output_dir):
        self.detector = detector
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_folder(self):
        if not os.path.exists(self.input_dir):
            print(f"LỖI: Không tìm thấy thư mục: {self.input_dir}")
            return

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(valid_extensions)]

        if not image_files:
            print("CẢNH BÁO: Không có ảnh nào để xử lý.")
            return

        print(f"--- BẮT ĐẦU DỰ ĐOÁN ({len(image_files)} ảnh) ---")

        for idx, img_name in enumerate(image_files):
            self._process_single_image(img_name, idx, len(image_files))

    def _process_single_image(self, img_name, idx, total):
        img_path = os.path.join(self.input_dir, img_name)
        save_path = os.path.join(self.output_dir, f"result_{img_name}")
        
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[{idx+1}/{total}] ❌ Lỗi đọc ảnh: {img_name}")
            return
            
        results = self.detector.predict(frame)
        for result in results:
            annotated_frame = self.detector.annotate(result)
            cv2.imwrite(save_path, annotated_frame)
            print(f"[{idx+1}/{total}] ✅ Đã lưu: {save_path}")

if __name__ == '__main__':
    model_path = r"D:\Project\Lab2\25-04-2026 ( argument )\No cutout\Chess_Project (No cutout)\weights\best.pt"
    input_dir = r"D:\Project\Lab2\Test_image"
    output_dir = r"D:\Project\Lab2\Test_image_output"

    chess_ai = ChessDetector(model_path=model_path, conf_threshold=0.6)
    processor = ImageBatchProcessor(detector=chess_ai, input_dir=input_dir, output_dir=output_dir)
    processor.process_folder()