import sys
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QFileDialog, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal

import cv2
from ultralytics import YOLO
import numpy as np
import torch

CLASS_NAMES = {0: "knife", 1: "handgun"}

# ----------------------------
# Detection Worker Thread
# ----------------------------
class DetectionWorker(QThread):
    update_image = pyqtSignal(QImage)
    update_log = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, source, model):
        super().__init__()
        self.source = source
        self.model = model
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        # Check if source is image or video or webcam
        if isinstance(self.source, str) and Path(self.source).suffix.lower() in ['.jpg', '.jpeg', '.png']:
            self._process_image(self.source)
        else:
            self._process_video_or_webcam(self.source)

        self.finished_signal.emit()

    def _process_image(self, img_path):
        img = cv2.imread(img_path)
        start_time = time.time()
        results = self.model(img)
        end_time = time.time()

        img = self._draw_results(img, results)
        qimg = self._convert_cv_qt(img)
        self.update_image.emit(qimg)

        log_text = f"Processed image: {img_path}\nTime taken: {(end_time - start_time)*1000:.2f} ms\n"
        for r in results:
            for box in r.boxes:
                conf = box.conf.item()
                cls = int(box.cls.item())
                log_text += f"Class {cls}, Confidence: {conf:.2f}\n"
        self.update_log.emit(log_text)

    def _process_video_or_webcam(self, source):
        cap = cv2.VideoCapture(0 if source == "webcam" else source)
        frame_count = 0

        # --- Check video length (skip if webcam) ---
        if source != "webcam":
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = total_frames / fps if fps > 0 else 0

            if duration > 30:
                self.update_log.emit(
                    f"Video too long ({duration:.2f}s). Please select a video ≤ 30s.\n"
                )
                cap.release()
                return

        start_time_total = time.time()  # For FPS calculation

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            results = self.model(frame)
            end_time = time.time()

            frame_count += 1
            frame = self._draw_results(frame, results)
            qimg = self._convert_cv_qt(frame)
            self.update_image.emit(qimg)

            log_text = f"Frame {frame_count}, Time: {(end_time - start_time)*1000:.2f} ms\n"
            for r in results:
                for box in r.boxes:
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    log_text += f"Detected {CLASS_NAMES.get(cls, cls)} with Confidence: {conf:.2f}\n"

            self.update_log.emit(log_text)

        # --- After detection, show FPS ---
        if frame_count > 0:
            total_time = time.time() - start_time_total
            avg_fps = frame_count / total_time
            self.update_log.emit(
                f"\n--- Detection Finished ---\n"
                f"Processed {frame_count} frames\n"
                f"Average FPS: {avg_fps:.2f}\n"
            )

        cap.release()

    def _draw_results(self, frame, results):
        for r in results:
            for box in r.boxes:
                conf = box.conf.item()
                # if conf < 0.4:   # <-- filter out low confidence detections
                #     continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls.item())
                label = f"{CLASS_NAMES.get(cls, cls)} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def _convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = qt_img.scaled(640, 480)
        return scaled_img


# ----------------------------
# Main GUI
# ----------------------------
class WeaponGUI(QWidget):
    def __init__(self, model_path):
        super().__init__()
        self.setWindowTitle("Handgun & Knife Detection")
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        if torch.backends.mps.is_available():
            self.model.to("mps")
            print("✅ Using Apple MPS backend for faster inference")
        else:
            self.model.to("cpu")
            print("⚠️ Running on CPU (slow)")
        self.init_ui()
        self.worker = None

    def init_ui(self):
        # Buttons
        self.btn_video = QPushButton("Input Video (≤30s)")
        self.btn_image = QPushButton("Input Image")
        self.btn_webcam = QPushButton("Webcam")
        self.btn_back = QPushButton("Back")
        self.btn_exit = QPushButton("Exit")

        self.btn_video.clicked.connect(self.select_video)
        self.btn_image.clicked.connect(self.select_image)
        self.btn_webcam.clicked.connect(self.start_webcam)
        self.btn_back.clicked.connect(self.back)
        self.btn_exit.clicked.connect(self.close_app)

        # Image display
        self.label_image = QLabel()
        self.label_image.setFixedSize(640, 480)

        # Log display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedWidth(300)

        # Layouts
        hbox = QHBoxLayout()
        hbox.addWidget(self.log_text)
        hbox.addWidget(self.label_image)

        btn_hbox = QHBoxLayout()
        btn_hbox.addWidget(self.btn_video)
        btn_hbox.addWidget(self.btn_image)
        btn_hbox.addWidget(self.btn_webcam)
        btn_hbox.addWidget(self.btn_back)
        btn_hbox.addWidget(self.btn_exit)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(btn_hbox)

        self.setLayout(vbox)

    # ----------------------------
    # Button functions
    # ----------------------------
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.start_detection(path)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.png *.jpeg)")
        if path:
            self.start_detection(path)

    def start_webcam(self):
        self.start_detection("webcam")

    def back(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self.label_image.clear()
        self.log_text.clear()

    def close_app(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self.close()

    # ----------------------------
    # Start Detection
    # ----------------------------
    def start_detection(self, source):
        self.worker = DetectionWorker(source, self.model)
        self.worker.update_image.connect(self.update_image)
        self.worker.update_log.connect(self.update_log)
        self.worker.start()

    def update_image(self, qimg):
        self.label_image.setPixmap(QPixmap.fromImage(qimg))

    def update_log(self, text):

        self.log_text.append(text)


# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    model_path = "/Users/mac/Documents/4-Sep-2025Model/last (11).pt"
    gui = WeaponGUI(model_path)
    gui.show()
    sys.exit(app.exec_())
