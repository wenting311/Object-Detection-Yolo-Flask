from flask import Flask, request, render_template, send_file, Response
from werkzeug.utils import secure_filename
import io
import shutil
import tempfile
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
from datetime import datetime
import torch
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
print(torch.cuda.is_available())

class Detection:
    def __init__(self):
        #download weights from here:https://github.com/ultralytics/ultralytics and change the path
        self.model = YOLO(r"yolo11n.pt")

    def predict(self, img, classes=[], conf=0.5):
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)

        return results

    def predict_and_detect(self, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.predict(img, classes, conf=conf)
        for result in results:
            for box in result.boxes:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
        return img, results

    def detect_from_image(self, image):
        result_img, _ = self.predict_and_detect(image, classes=[], conf=0.5)
        return result_img


detection = Detection()


# --- Simple centroid-based tracker ---
class CentroidTracker:
    def __init__(self, maxDisappeared=40, maxDistance=100):
        self.nextObjectID = 0
        self.objects = {}  # objectID -> centroid (x, y)
        self.bboxes = {}   # objectID -> bbox (startX, startY, endX, endY)
        self.disappeared = {}  # objectID -> frames disappeared
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.bboxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.bboxes:
            del self.bboxes[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]

    def update(self, rects):
        # rects: list of (startX, startY, endX, endY)
        if len(rects) == 0:
            # mark all existing objects as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return dict(self.bboxes)

        inputCentroids = []
        for (startX, startY, endX, endY) in rects:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids.append((cX, cY))

        # if no existing objects, register all
        if len(self.objects) == 0:
            for i, centroid in enumerate(inputCentroids):
                self.register(centroid, rects[i])
            return dict(self.bboxes)

        # match existing objects to input centroids
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        # compute distance matrix
        D = []
        for oc in objectCentroids:
            row = []
            for ic in inputCentroids:
                d = ((oc[0] - ic[0]) ** 2 + (oc[1] - ic[1]) ** 2) ** 0.5
                row.append(d)
            D.append(row)

        # greedy assignment based on minimal distance
        rows = list(range(len(D)))
        cols = list(range(len(D[0])))
        assignedRows = set()
        assignedCols = set()
        pairs = []

        while True:
            minDist = None
            minRow = None
            minCol = None
            for r in rows:
                if r in assignedRows:
                    continue
                for c in cols:
                    if c in assignedCols:
                        continue
                    d = D[r][c]
                    if minDist is None or d < minDist:
                        minDist = d
                        minRow = r
                        minCol = c
            if minDist is None:
                break
            if minDist > self.maxDistance:
                break
            assignedRows.add(minRow)
            assignedCols.add(minCol)
            pairs.append((minRow, minCol))

        # update matched
        for (r, c) in pairs:
            objectID = objectIDs[r]
            self.objects[objectID] = inputCentroids[c]
            self.bboxes[objectID] = rects[c]
            self.disappeared[objectID] = 0

        # handle disappeared
        for r in rows:
            if r not in assignedRows:
                objectID = objectIDs[r]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

        # register new inputs
        for c in cols:
            if c not in assignedCols:
                self.register(inputCentroids[c], rects[c])

        return dict(self.bboxes)


# tracker instance
tracker = CentroidTracker()

# Logging configuration
track_log_enabled = False
track_log_path = 'logs/track_log.csv'
# rotation configuration
TRACK_LOG_ROTATE_DAILY = True
TRACK_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB default
TRACK_LOG_BACKUP_COUNT = 7  # keep last N rotated logs

# ensure logs directory exists and header initialized
os.makedirs(os.path.dirname(track_log_path) or 'logs', exist_ok=True)
if not os.path.exists(track_log_path):
    with open(track_log_path, 'w', encoding='utf-8') as f:
        f.write('timestamp,objectID,startX,startY,endX,endY\n')


def prune_old_backups():
    """Remove old rotated log files while keeping the most recent TRACK_LOG_BACKUP_COUNT."""
    try:
        log_dir = os.path.dirname(track_log_path) or '.'
        base = os.path.splitext(os.path.basename(track_log_path))[0] + '_'
        files = []
        for fn in os.listdir(log_dir):
            if fn.startswith(base) and fn.endswith('.csv'):
                full = os.path.join(log_dir, fn)
                files.append((os.path.getmtime(full), full))
        files.sort(reverse=True)
        # keep the newest TRACK_LOG_BACKUP_COUNT
        for _, fpath in files[TRACK_LOG_BACKUP_COUNT:]:
            try:
                os.remove(fpath)
            except Exception:
                pass
    except Exception as e:
        print('Prune old backups error:', e)


def rotate_log_if_needed():
    """Rotate the log file based on daily or size limits.

    - If TRACK_LOG_ROTATE_DAILY and the file's modification date is before today, move it to a dated file.
    - If file size exceeds TRACK_LOG_MAX_BYTES, rotate with a timestamp suffix.
    """
    try:
        if not os.path.exists(track_log_path):
            return

        log_dir = os.path.dirname(track_log_path) or '.'

        # Daily rotation
        if TRACK_LOG_ROTATE_DAILY:
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(track_log_path))
                file_date = mtime.strftime('%Y%m%d')
                today = datetime.now().strftime('%Y%m%d')
                if file_date != today:
                    dest = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(track_log_path))[0]}_{file_date}.csv")
                    os.rename(track_log_path, dest)
                    with open(track_log_path, 'w', encoding='utf-8') as f:
                        f.write('timestamp,objectID,startX,startY,endX,endY\n')
                    prune_old_backups()
                    return
            except Exception as e:
                print('Log rotation (daily) check error:', e)

        # Size-based rotation
        try:
            size = os.path.getsize(track_log_path)
            if TRACK_LOG_MAX_BYTES and size >= TRACK_LOG_MAX_BYTES:
                ts = datetime.now().strftime('%Y%m%dT%H%M%S')
                dest = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(track_log_path))[0]}_{ts}.csv")
                os.rename(track_log_path, dest)
                with open(track_log_path, 'w', encoding='utf-8') as f:
                    f.write('timestamp,objectID,startX,startY,endX,endY\n')
                prune_old_backups()
        except Exception as e:
            print('Log rotation (size) check error:', e)
    except Exception as e:
        print('Log rotation error:', e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = Image.open(file_path).convert("RGB")
        img = np.array(img)
        img = cv2.resize(img, (512, 512))
        img = detection.detect_from_image(img)
        output = Image.fromarray(img)

        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)

        os.remove(file_path)
        return send_file(buf, mimetype='image/png')


@app.route('/video')
def index_video():
    return render_template('video.html')


def gen_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (512, 512))
        if frame is None:
            break
        frame = detection.detect_from_image(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Tracking routes and generators ---
@app.route('/track')
def track_page():
    return render_template('track.html')


@app.route('/track_feed')
def track_feed():
    # allow client to control whether IDs are drawn via query param
    draw_ids = request.args.get('draw_ids', '1') == '1'
    return Response(gen_frames_track(draw_ids), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/track_count')
def track_count():
    # return current active count
    return str(len(tracker.objects))


@app.route('/track_logging')
def track_logging():
    global track_log_enabled
    enable = request.args.get('enable')
    if enable is None:
        return '1' if track_log_enabled else '0'
    track_log_enabled = enable in ('1', 'true', 'True')
    return '1' if track_log_enabled else '0'


@app.route('/track_logging_status')
def track_logging_status():
    return '1' if track_log_enabled else '0'


@app.route('/download_track_log', methods=['GET', 'HEAD'])
def download_track_log():
    # allow HEAD to check for existence
    if not os.path.exists(track_log_path):
        return ('', 404)
    # HEAD should quickly report existence
    if request.method == 'HEAD':
        return ('', 200)

    # Create an atomic snapshot copy, read into memory, then serve from BytesIO.
    tmp_name = None
    try:
        fd, tmp_name = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        shutil.copy2(track_log_path, tmp_name)
        with open(tmp_name, 'rb') as f:
            data = f.read()
    except Exception as e:
        print('Download snapshot error:', e)
        return ('', 500)
    finally:
        try:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass

    return send_file(io.BytesIO(data), mimetype='text/csv', as_attachment=True, download_name='track_log.csv')


def gen_frames_track(draw_ids=True):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (512, 512))

        # perform detection
        try:
            results = detection.predict(frame, conf=0.5)
        except Exception as e:
            # fallback to original frame if detection fails
            print('Detection error:', e)
            results = []

        rects = []
        for res in results:
            for box in res.boxes:
                cls = int(box.cls[0])
                name = res.names[cls]
                # filter only persons
                if name != 'person':
                    continue
                x1 = max(0, int(box.xyxy[0][0]))
                y1 = max(0, int(box.xyxy[0][1]))
                x2 = min(frame.shape[1] - 1, int(box.xyxy[0][2]))
                y2 = min(frame.shape[0] - 1, int(box.xyxy[0][3]))
                rects.append((x1, y1, x2, y2))

        # update tracker and get labeled bboxes
        labeled_bboxes = tracker.update(rects)

        # optionally write logs
        if track_log_enabled and len(labeled_bboxes) > 0:
            try:
                # check rotation policy before appending
                rotate_log_if_needed()
                t = datetime.now().isoformat()
                with open(track_log_path, 'a', encoding='utf-8') as logf:
                    for objectID, bbox in labeled_bboxes.items():
                        (startX, startY, endX, endY) = bbox
                        logf.write(f"{t},{objectID},{startX},{startY},{endX},{endY}\n")
            except Exception as e:
                print('Logging error:', e)

        # draw detections and IDs
        for objectID, bbox in labeled_bboxes.items():
            (startX, startY, endX, endY) = bbox
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            if draw_ids:
                cv2.putText(frame, f"ID {objectID}", (startX, max(15, startY - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # overlay count
        count = len(labeled_bboxes)
        cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
    #http://localhost:8000/video for video source
    #http://localhost:8000 for image source
