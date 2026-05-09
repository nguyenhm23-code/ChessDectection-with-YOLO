import cv2
import time
import os
from datetime import datetime
from core_detector import ChessDetector

class RealtimeCameraApp:
    def __init__(self, detector: ChessDetector, camera_index=1, flip_mode=0):
        self.detector = detector
        self.camera_index = camera_index
        self.flip_mode = flip_mode
        
        # Thư mục lưu trữ
        self.save_result_dir = r"D:\Project\Lab2\25-04-2026 ( argument )\No cutout\Preview"
        self.save_data_dir = r"D:\Project\Lab2\25-04-2026 ( argument )\No cutout\Bonus_Data"
        os.makedirs(self.save_result_dir, exist_ok=True)
        os.makedirs(self.save_data_dir, exist_ok=True)

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print(f"LỖI: Không thể mở Camera số {self.camera_index}.")
            return

        print("--- ĐANG CHẠY REAL-TIME ---")
        fail_count, prev_frame_time = 0, 0

        while True:
            success, frame = cap.read()
            if not success:
                fail_count += 1
                if fail_count > 10:
                    print("❌ Mất kết nối Camera. Đang dừng...")
                    break
                continue
            
            fail_count = 0
            if self.flip_mode is not None:
                frame = cv2.flip(frame, self.flip_mode)

            raw_frame = frame.copy()
            annotated_frame = frame

            # Gọi class Detector để xử lý AI
            results = self.detector.predict(frame, stream=True)

            for result in results:
                annotated_frame = self.detector.annotate(result)
                
                # Tính FPS
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0            
                prev_frame_time = new_frame_time
                
                self._draw_hud(annotated_frame, fps)
                cv2.imshow("YOLO Chess Detection", annotated_frame)

            if self._handle_keystrokes(raw_frame, annotated_frame):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _draw_hud(self, frame, fps):
        cv2.putText(frame, f"Cam: {self.camera_index} | FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Exit: Q | Save Raw: S | Save Box: A", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _handle_keystrokes(self, raw_frame, annotated_frame):
        key = cv2.waitKey(1) & 0xFF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if key == ord('q') or key == 27:
            return True # Signal to break
        elif key == ord('s'):
            filepath = os.path.join(self.save_data_dir, f"raw_{timestamp}.jpg")
            cv2.imwrite(filepath, raw_frame)
            print(f"📸 Đã lưu ảnh GỐC: {filepath}")
        elif key == ord('a'):
            filepath = os.path.join(self.save_result_dir, f"box_{timestamp}.jpg")
            cv2.imwrite(filepath, annotated_frame)
            print(f"📸 Đã lưu ảnh KẾT QUẢ: {filepath}")
        return False

if __name__ == '__main__':
    model_path = r"D:\Project\Lab2\25-04-2026 ( argument )\No cutout\Chess_Project (No cutout)\weights\best.pt"
    # Khởi tạo đối tượng Core, tiêm (inject) nó vào ứng dụng Camera
    chess_ai = ChessDetector(model_path=model_path, conf_threshold=0.6)
    app = RealtimeCameraApp(detector=chess_ai, camera_index=1, flip_mode=0)
    app.run()