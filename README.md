# 🟢 SmartVision

### Real-Time AI Video Analytics, Object Detection & Tracking System

SmartVision is an AI-powered real-time video analytics system that detects, tracks, counts, and monitors objects from live camera input.

It combines **YOLO** object detection with **ByteTrack** multi-object tracking to generate real-time insights such as object counts, entry/exit statistics, restricted-zone monitoring, and security alerts.

---

## 🚀 Live Demo

🌐 **Live Application:**  
https://smartvision-8dp7.onrender.com

> Note: The application is hosted on Render's free tier. The service may take some time to wake up after a period of inactivity.

---

## 🎯 Project Objective

Traditional CCTV systems mainly record video and require humans to continuously monitor the footage.

SmartVision transforms a camera feed into an intelligent monitoring system by automatically:

- Detecting objects
- Assigning tracking IDs
- Tracking objects across frames
- Counting objects
- Detecting entry and exit movements
- Monitoring restricted zones
- Generating security alerts
- Displaying live analytics

---

## ✨ Key Features

### 🎯 Real-Time Object Detection

Uses a pretrained **YOLO26** model to identify objects from video frames.

Supported objects include common COCO classes such as:

- Person
- Car
- Bus
- Truck
- Bicycle
- Motorcycle
- Dog
- Cat
- And many more

---

### 🆔 Multi-Object Tracking

SmartVision uses **ByteTrack** to maintain the identity of detected objects across consecutive frames.

Each tracked object receives a unique tracking ID.


Person #18
Car #7
