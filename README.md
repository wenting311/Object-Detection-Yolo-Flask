# Object Detection Flask Application

This repository contains a Flask-based web application for object detection using the YOLO (You Only Look Once) model. The application can process images and video streams, detecting objects and displaying the results.

<div>
    <img src="./img1.jpg" width="60%" hight="50%">
</div>

## Features

- Upload an image and get the detected objects highlighted.
- Real-time object detection from a video stream.

## Requirements

- Python 3.6+
- Flask
- ultralytics
- numpy
- pillow
- opencv-python
- werkzeug

## Setup

1. Clone the repository:

```bash
git clone https://github.com/givkashi/Object-Detection-Yolo-Flask.git
cd Object-Detection-Yolo-Flask
```

2. Download the YOLO weights from [[here](https://github.com/ultralytics/ultralytics)] and place them in the object_detection directory.

3. Running the Application
To start the Flask application, run:
```bash
python app.py
```
The application will be available at http://localhost:8000.

Usage
Image Upload
1. Open your browser and navigate to http://localhost:8000.
2. Upload an image file.
3. The application will process the image and display the detected objects.

Real-time Video Detection
1. Open your browser and navigate to http://localhost:8000/video.
2. The application will start the webcam and display the real-time object detection.

Real-time Person Tracking (新增功能)

1. Open your browser and navigate to http://localhost:8000/track.
   - This page streams a live MJPEG feed from the webcam and runs person detection + tracking.
   - Controls:
     - 「顯示 ID」: 切換是否在畫面上顯示每位被追蹤者的 ID 標籤。
     - 「啟用日誌」: 啟用後伺服器會把每幀的追蹤資料附加寫入 `logs/track_log.csv`。

2. Endpoints useful for debugging and automation:
   - `/track_feed?draw_ids=1` — MJPEG stream; set `draw_ids=0` to hide IDs.
   - `/track_count` — returns current number of tracked persons as plain text.
   - `/track_logging?enable=1` or `/track_logging?enable=0` — enable/disable logging via HTTP.
   - `/track_logging_status` — returns `1` if logging enabled, `0` otherwise.

3. Log file format and location:
   - Logs are appended to `logs/track_log.csv` in CSV format: `timestamp,objectID,startX,startY,endX,endY`.
   - Example line:
     `2026-01-06T07:59:12.345678,0,120,80,200,300`

Notes & tips
- Only open **one** page that accesses the webcam at a time (either `/video` or `/track`) to avoid camera conflicts.
- If performance is slow, consider resizing frames or running detection every N frames.
- The tracker uses a simple centroid algorithm suitable for small-to-moderate crowds; consider upgrading to a Hungarian matcher or adding motion modelling for better results.

