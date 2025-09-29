# Real-Time Knife and Gun Detection Using YOLO

**Collaborators:** Aye Mie Mie Soe, Paing Soe Khant  
**Course/Project:** Image Processing Project  

---

## Project Overview

This project implements a **real-time harmful object detection system** to detect **knives and guns** in images, video files, and live webcam streams. The system leverages **YOLOv8m**, **OpenCV**, and a **PyQt GUI** to provide an interactive, accurate, and user-friendly security monitoring tool.

The system is designed to enhance **public safety and surveillance efficiency** by automatically detecting harmful objects and providing visual and log-based information in real time.

---

## Features

- **Real-Time Detection:** Detect knives and guns in images, videos, and live webcam feeds.  
- **Interactive GUI:**  
  - **Input Video:** Upload and detect objects in video files.  
  - **Input Image:** Detect objects in a single image.  
  - **Webcam:** Enable real-time webcam detection.  
  - **Back:** Return to the previous state.  
  - **Finish:** Terminate the program.  
- **Information Panel:** Displays the number of frames processed, detected objects with confidence scores, and frame processing time (ms).  
- **Visual Detection Panel:** Shows detection results with bounding boxes and labels.  

---

## Dataset

- **Sources:** https://www.kaggle.com/datasets/sithutungraki/weapon-detection5000  
- **Classes:** Knife (Class 0), Gun (Class 1)  
- **Dataset Split:**  
  - Training: 70% (7,000 images)  
  - Validation: 15% (1,500 images)  
  - Testing: 15% (1,500 images)  
- **Augmentation:** Rotation, scaling, cropping, motion/gaussian blur, partial occlusion, lighting, and color variations.  

---

## Model Performance

- **YOLOv8m** was selected for deployment due to its **best balance between accuracy and speed**.

| Model     | mAP50-95 | Knife AP (%) | Gun AP (%) |  
|-----------|-----------|--------------|------------|  
| YOLOv5s   | 66.5%     | 57.7%        | 75.3%      |  
| YOLOv5m   | 69.5%     | 59.5%        | 79.5%      |  
| YOLOv8s   | 69.1%     | 59.7%        | 78.4%      |  
| YOLOv8m   | 70.2%     | 59.6%        | 80.7%      |  

---

## Tech Stack

- **Programming Language:** Python  
- **Object Detection:** YOLOv8m  
- **Image Processing:** OpenCV, Albumentations  
- **GUI:** PyQt5  
- **Visualization:** Matplotlib (for data augmentation visualization)  
- **Hardware Used for Training:** Kaggle Notebook with NVIDIA T4 GPU  

---

## Installation

1. Clone this repository:  
```bash
git clone https://github.com/SiThuTun22/GunAndKnifeDetection

