import cv2
import os
import time
from datetime import datetime
import numpy as np
from core_detector import ChessDetector

class ModelComparator:
    def __init__(self, model_1_path, model_2_path, name_1="Model 1", name_2="Model 2"):
        print("⏳ Đang tải 2 models vào bộ nhớ...")
        self.detector1 = ChessDetector(model_path=model_1_path, conf_threshold=0.6)
        self.detector2 = ChessDetector(model_path=model_2_path, conf_threshold=0.6)
        self.name_1 = name_1
        self.name_2 = name_2

    def _process_single_frame(self, frame):
        """Xử lý 1 frame qua cả 2 model và ghép lại"""
        # Copy frame để 2 model không vẽ đè box lên nhau
        frame1 = frame.copy()
        frame2 = frame.copy()

        # Model 1 xử lý
        results1 = self.detector1.predict(frame1, stream=False)
        for r in results1:
            frame1 = self.detector1.annotate(r)
            
        # Model 2 xử lý
        results2 = self.detector2.predict(frame2, stream=False)
        for r in results2:
            frame2 = self.detector2.annotate(r)

        # Vẽ tên Model lên góc trái của mỗi bên để dễ phân biệt
        cv2.putText(frame1, f"[{self.name_1}]", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame2, f"[{self.name_2}]", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Ghép 2 ảnh theo chiều ngang
        combined_frame = cv2.hconcat([frame1, frame2])
        return combined_frame

    def run_realtime(self, camera_index=0, flip_mode = 0):
        """Chạy so sánh trực tiếp trên Camera"""
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Hạ độ phân giải xuống để ghép 2 ảnh không bị tràn màn hình
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print(f"❌ LỖI: Không thể mở Camera số {camera_index}.")
            return

        print("--- REAL-TIME COMPARE ---")
        prev_frame_time = 0

        while True:
            success, frame = cap.read()
            if not success:
                continue
            
            if not flip_mode:
                frame = cv2.flip(frame, flip_mode)

            # Xử lý và ghép ảnh
            combined_frame = self._process_single_frame(frame)

            # Tính toán FPS tổng
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0            
            prev_frame_time = new_frame_time

            cv2.putText(combined_frame, f"Compare (2 Models): {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Compare Models ( Q to ESC )", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_image_batch(self, input_dir, output_dir):
        """Chạy so sánh trên một thư mục ảnh có sẵn"""
        os.makedirs(output_dir, exist_ok=True)
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        
        if not os.path.exists(input_dir):
            print(f"❌ LỖI: Không tìm thấy thư mục: {input_dir}")
            return
            
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
        print(f"--- BẮT ĐẦU SO SÁNH TRÊN {len(image_files)} ẢNH ---")

        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(input_dir, img_name)
            frame = cv2.imread(img_path)
            
            if frame is None:
                continue
                
            combined_frame = self._process_single_frame(frame)
            
            save_path = os.path.join(output_dir, f"compare_{img_name}")
            cv2.imwrite(save_path, combined_frame)
            print(f"[{idx+1}/{len(image_files)}] ✅ Đã lưu kết quả so sánh: {save_path}")


if __name__ == '__main__':
    # ================= CẤU HÌNH ĐƯỜNG DẪN =================
    # Đường dẫn tới 2 file weights bạn muốn so sánh
    PATH_MODEL_1 = r"D:\Project\Lab2\workspace2\Version\12-5-2026\Chess_Project(yolo26s)\weights\best.pt"
    PATH_MODEL_2 = r"D:\Project\Lab2\workspace2\Version\25-04-2026\Chess_Project\weights\best.pt"
    
    # Khởi tạo công cụ so sánh
    comparator = ModelComparator(
        model_1_path=PATH_MODEL_1, 
        model_2_path=PATH_MODEL_2,
        name_1="Version 11-5", 
        name_2="Version 25-4"
    )

    # ================= CHỌN CHẾ ĐỘ CHẠY =================
    # Đổi thành "image" nếu muốn test trên ảnh, hoặc "realtime" nếu dùng cam
    MODE = "realtime" 
    #MODE = "image" 

    if MODE == "realtime":
        comparator.run_realtime(camera_index=0, flip_mode=0)
    elif MODE == "image":
        INPUT_FOLDER = r"D:\Project\Lab2\workspace2\Test_image"
        OUTPUT_FOLDER = r"D:\Project\Lab2\workspace2\Compare\11-5-2026"
        comparator.run_image_batch(input_dir=INPUT_FOLDER, output_dir=OUTPUT_FOLDER)