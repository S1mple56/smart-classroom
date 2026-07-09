from base64 import b64encode
import base64
import numpy as np
from glob import glob
from io import BytesIO
import mimetypes
import os
import shutil
import time

from flask import Flask, jsonify, request, send_from_directory, Response
import json
import threading
import queue
from datetime import datetime
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO
try:
    import torch
except Exception:
    torch = None


WEB_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(WEB_DIR, ".."))
import sys
sys.path.insert(0, BASE_DIR)
import inference_student_pose
TRY_ORIGIN = os.path.join(BASE_DIR, "TRY", "origin")
TRY_TARGET = os.path.join(BASE_DIR, "TRY", "target")
os.makedirs(TRY_ORIGIN, exist_ok=True)
os.makedirs(TRY_TARGET, exist_ok=True)
CAPTURES_DIR = os.path.join(WEB_DIR, 'captures')
CAPTURES_ORIGINALS = os.path.join(CAPTURES_DIR, 'originals')
CAPTURES_ANNOTATED = os.path.join(CAPTURES_DIR, 'annotated')
CAPTURES_STATS = os.path.join(CAPTURES_DIR, 'stats')
CAPTURES_ALERTS = os.path.join(CAPTURES_DIR, 'alerts')
os.makedirs(CAPTURES_ORIGINALS, exist_ok=True)
os.makedirs(CAPTURES_ANNOTATED, exist_ok=True)
os.makedirs(CAPTURES_STATS, exist_ok=True)
os.makedirs(CAPTURES_ALERTS, exist_ok=True)

# Alerts storage
ALERTS_FILE = os.path.join(WEB_DIR, "alerts.json")
alerts_lock = threading.Lock()
alerts_queue = queue.Queue()
try:
    if os.path.isfile(ALERTS_FILE):
        with open(ALERTS_FILE, 'r', encoding='utf-8') as f:
            alerts_store = json.load(f)
    else:
        alerts_store = []
except Exception:
    alerts_store = []

def save_alerts():
    try:
        with alerts_lock:
            with open(ALERTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(alerts_store, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def push_alert(alert):
    # alert: dict
    with alerts_lock:
        alerts_store.insert(0, alert)
        # keep recent 200
        del alerts_store[200:]
        save_alerts()
    try:
        alerts_queue.put(alert, block=False)
    except Exception:
        pass

# Attendance records storage
ATTENDANCE_FILE = os.path.join(WEB_DIR, "attendance_records.json")
attendance_lock = threading.Lock()
try:
    if os.path.isfile(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r', encoding='utf-8') as f:
            attendance_store = json.load(f)
    else:
        attendance_store = []
except Exception:
    attendance_store = []

def save_attendance():
    try:
        with attendance_lock:
            with open(ATTENDANCE_FILE, 'w', encoding='utf-8') as f:
                json.dump(attendance_store, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# TRACKED_CLASSES will be initialized after loading the model to match model.names
TRACKED_CLASSES = None

# expected tracked keys used by frontend (keep stable)
EXPECTED_TRACKED = ("hand_up", "sleep", "phone", "abnormal")


def newest_file(pattern):
    matches = [path for path in glob(pattern) if os.path.isfile(path)]
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def find_model():
    candidates = [
        newest_file(os.path.join(BASE_DIR, "runs", "detect", "*", "weights", "last.pt")),
        newest_file(os.path.join(BASE_DIR, "runs", "detect", "*", "weights", "best.pt")),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("No YOLO model weights were found.")


def ensure_origin_path(filename):
    safe_name = os.path.basename(filename)
    path = os.path.abspath(os.path.join(TRY_ORIGIN, safe_name))
    # 验证路径确实在 TRY_ORIGIN 目录下（basename 已防注入，这里做双重检查）
    if os.path.dirname(path) != os.path.abspath(TRY_ORIGIN):
        raise ValueError("invalid file path")
    return path


def find_latest_output(directory, expected_prefix=None):
    newest_path = None
    newest_mtime = 0.0
    for root, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            if expected_prefix:
                media_type, _ = mimetypes.guess_type(path)
                if not media_type or not media_type.startswith(expected_prefix):
                    continue
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_path = path
    return newest_path


def summarize_results(results, names):
    """
    Summarize detection results into counts and tracked metrics.
    Returns a dict with frame_count, total_detections, counts (per-class),
    tracked_totals, tracked_max_per_frame, display_counts, display_mode, display_mode_label.
    """
    # build class id -> name map
    class_map = {}
    if isinstance(names, dict):
        try:
            class_map = {int(k): v for k, v in names.items()}
        except Exception:
            class_map = {int(k): v for k, v in names.items() if k.isdigit()}
    elif isinstance(names, (list, tuple)):
        class_map = {i: n for i, n in enumerate(names)}

    total_counts = {}
    max_per_frame = {}

    for r in results:
        frame_counts = {}
        boxes = getattr(r, 'boxes', None)
        cls_ids = []
        try:
            if boxes is not None:
                # boxes.cls may be a tensor
                cls_ids = getattr(boxes, 'cls', None)
                if hasattr(cls_ids, 'cpu'):
                    cls_ids = cls_ids.cpu().numpy().tolist()
                elif hasattr(cls_ids, 'numpy'):
                    cls_ids = cls_ids.numpy().tolist()
                elif isinstance(cls_ids, (list, tuple, np.ndarray)):
                    cls_ids = list(cls_ids)
                else:
                    cls_ids = []
        except Exception:
            cls_ids = []

        for cid in cls_ids:
            try:
                cid_int = int(cid)
            except Exception:
                continue
            class_name = class_map.get(cid_int, f'class_{cid_int}')
            total_counts[class_name] = total_counts.get(class_name, 0) + 1
            frame_counts[class_name] = frame_counts.get(class_name, 0) + 1

        for class_name, count in frame_counts.items():
            max_per_frame[class_name] = max(max_per_frame.get(class_name, 0), int(count))

    tracked_keys = TRACKED_CLASSES if TRACKED_CLASSES else EXPECTED_TRACKED
    tracked_totals = {name: int(total_counts.get(name, 0)) for name in tracked_keys}
    tracked_max = {name: int(max_per_frame.get(name, 0)) for name in tracked_keys}
    is_video = len(results) > 1
    display_counts = tracked_max if is_video else tracked_totals

    return {
        "frame_count": int(len(results)),
        "total_detections": int(sum(total_counts.values())),
        "counts": {key: int(value) for key, value in total_counts.items()},
        "tracked_totals": tracked_totals,
        "tracked_max_per_frame": tracked_max,
        "display_counts": display_counts,
        "display_mode": "max_per_frame" if is_video else "image_total",
        "display_mode_label": "视频逐帧峰值" if is_video else "图片检测数量",
    }


def ensure_tracked_keys(stats):
    """Ensure stats contains the expected tracked keys (fill missing with 0)."""
    if not isinstance(stats, dict):
        return stats
    for key in (EXPECTED_TRACKED or []):
        # counts
        if 'counts' not in stats:
            stats['counts'] = {}
        if key not in stats['counts']:
            stats['counts'][key] = 0
        # tracked_totals
        if 'tracked_totals' not in stats:
            stats['tracked_totals'] = {}
        if key not in stats['tracked_totals']:
            stats['tracked_totals'][key] = 0
        # tracked_max_per_frame or tracked_max
        if 'tracked_max_per_frame' not in stats:
            stats['tracked_max_per_frame'] = {}
        if key not in stats['tracked_max_per_frame']:
            stats['tracked_max_per_frame'][key] = 0
        # display_counts
        if 'display_counts' not in stats:
            stats['display_counts'] = {}
        if key not in stats['display_counts']:
            stats['display_counts'][key] = 0
    return stats


app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
CORS(app)

# 优先使用 train-final 的 best.pt，不存在则自动查找
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train-final", "weights", "best.pt")
if not os.path.isfile(MODEL_PATH):
    MODEL_PATH = find_model()

print("Using model:", MODEL_PATH)
model = YOLO(MODEL_PATH)
# choose device: prefer CUDA if available
if torch is not None and torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
print('Using device:', DEVICE)

# derive tracked classes from the loaded model names when possible
try:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        TRACKED_CLASSES = tuple([str(v) for k, v in sorted(names.items(), key=lambda x: int(x[0]))])
    elif isinstance(names, (list, tuple)):
        TRACKED_CLASSES = tuple([str(v) for v in names])
    else:
        TRACKED_CLASSES = tuple()
except Exception:
    TRACKED_CLASSES = tuple()


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/model-info")
def model_info():
    return jsonify(
        {
            "model_path": MODEL_PATH,
            "model_name": os.path.basename(MODEL_PATH),
            "model_classes": getattr(model, "names", None),
        }
    )


@app.route('/alerts', methods=['GET', 'POST'])
def alerts():
    if request.method == 'GET':
        # return recent alerts
        with alerts_lock:
            return jsonify({'alerts': alerts_store})

    # POST: create alert
    payload = request.get_json(silent=True) or {}
    alert = {
        'id': payload.get('id') or int(time.time() * 1000),
        'timestamp': payload.get('timestamp') or datetime.utcnow().isoformat() + 'Z',
        'level': payload.get('level') or 'warning',
        'type': payload.get('type') or 'abnormal',
        'count': int(payload.get('count') or 1),
        'meta': payload.get('meta') or {},
    }
    # optional image_base64
    if payload.get('image_base64'):
        alert['image_base64'] = payload.get('image_base64')

    push_alert(alert)
    return jsonify({'ok': True, 'alert': alert}), 201


@app.route('/alerts/stream')
def alerts_stream():
    def gen():
        # send current backlog first? we stream only new alerts
        while True:
            try:
                alert = alerts_queue.get(timeout=30)
            except queue.Empty:
                # 发送心跳保持连接
                yield ": heartbeat\n\n"
                continue
            try:
                data = json.dumps(alert, ensure_ascii=False)
            except Exception:
                data = json.dumps({'id': alert.get('id')})
            yield f"data: {data}\n\n"

    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route("/list")
def list_files():
    images = []
    videos = []
    for name in sorted(os.listdir(TRY_ORIGIN)):
        path = os.path.join(TRY_ORIGIN, name)
        if not os.path.isfile(path):
            continue
        media_type, _ = mimetypes.guess_type(path)
        if media_type and media_type.startswith("image"):
            images.append(name)
        elif media_type and media_type.startswith("video"):
            videos.append(name)
    return jsonify({"images": images, "videos": videos})


@app.route("/origin/<path:filename>")
def serve_origin(filename):
    return send_from_directory(TRY_ORIGIN, filename)


@app.route("/target/<path:filename>")
def serve_target(filename):
    return send_from_directory(TRY_TARGET, filename)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    filename = payload.get("file")
    media_type = payload.get("type", "image")

    if not filename:
        return jsonify({"error": "file missing"}), 400

    try:
        src_path = ensure_origin_path(filename)
    except ValueError:
        return jsonify({"error": "invalid file path"}), 400

    if not os.path.isfile(src_path):
        return jsonify({"error": "file not found"}), 404

    timestamp = int(time.time())

    if media_type == "image":
        results = inference_student_pose.run_inference(src_path, TRY_TARGET, model_path=MODEL_PATH, imgsz=640, conf=0.25, iou=0.45, device=DEVICE, save=False)
        if not results:
            return jsonify({"error": "no results"}), 500
        # derive names from the cached model instance
        model_for_names = inference_student_pose._MODEL_CACHE.get(MODEL_PATH)
        stats = summarize_results(results, getattr(model_for_names, 'names', None))
        stats = ensure_tracked_keys(stats)

        try:
            annotated = results[0].plot()
            image = Image.fromarray(annotated)
        except Exception:
            image = Image.open(src_path).convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = b64encode(buffer.getvalue()).decode("utf-8")

        # build normalized boxes for frontend drawing
        boxes_out = []
        try:
            r = results[0]
            b = getattr(r, 'boxes', None)
            if b is not None:
                try:
                    xyxy = b.xyxy.cpu().numpy()
                    confs = b.conf.cpu().numpy()
                    clsids = b.cls.cpu().numpy()
                except Exception:
                    xyxy, confs, clsids = [], [], []

                try:
                    img_w, img_h = image.size
                except Exception:
                    img_h = 1
                    img_w = 1

                class_names = getattr(model_for_names, 'names', {})
                for bb, cf, cid in zip(xyxy, confs, clsids):
                    x1, y1, x2, y2 = map(float, bb)
                    cid = int(cid)
                    # model.names 是 dict[int, str]，用 int(cid) 查找
                    if isinstance(class_names, (list, tuple)):
                        cname = class_names[cid] if cid < len(class_names) else str(cid)
                    elif isinstance(class_names, dict):
                        cname = class_names.get(cid, str(cid))
                    else:
                        cname = str(cid)
                    nx1 = max(0.0, min(1.0, x1 / (img_w or 1)))
                    ny1 = max(0.0, min(1.0, y1 / (img_h or 1)))
                    nx2 = max(0.0, min(1.0, x2 / (img_w or 1)))
                    ny2 = max(0.0, min(1.0, y2 / (img_h or 1)))
                    boxes_out.append({
                        'xyxy': [nx1, ny1, nx2, ny2],
                        'conf': float(cf),
                        'class_id': cid,
                        'class_name': str(cname)
                    })
        except Exception:
            boxes_out = []

        # if abnormal detected, create alert with annotated image
        try:
            if stats.get('display_counts', {}).get('abnormal', 0) > 0:
                alert = {
                    'id': int(time.time() * 1000),
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'level': 'warning',
                    'type': 'abnormal',
                    'count': int(stats.get('display_counts', {}).get('abnormal', 0)),
                    'meta': {'source': 'predict', 'file': os.path.basename(src_path)},
                    'image_base64': image_b64,
                }
                push_alert(alert)
        except Exception:
            pass

        return jsonify({
            "image_base64": image_b64,
            "stats": stats,
            "boxes": boxes_out,
        })
    
    # handle video files: run inference and return annotated video path + stats
    if media_type == "video":
        try:
            before_ts = time.time()
            results = inference_student_pose.run_inference(src_path, TRY_TARGET, model_path=MODEL_PATH, imgsz=640, conf=0.25, iou=0.45, device=DEVICE, save=True)
            if not results:
                return jsonify({"error": "no results"}), 500

            model_for_names = inference_student_pose._MODEL_CACHE.get(MODEL_PATH)
            stats = summarize_results(results, getattr(model_for_names, 'names', None))
            stats = ensure_tracked_keys(stats)

            # find most recently written annotated file under TRY_TARGET (recursive), prefer video files
            annot_fname = None
            try:
                candidates = []
                video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
                for root, dirs, files in os.walk(TRY_TARGET):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        try:
                            m = os.path.getmtime(fp)
                        except Exception:
                            continue
                        if m >= (before_ts - 5):
                            rel = os.path.relpath(fp, TRY_TARGET)
                            candidates.append((m, rel))
                # prefer video extension candidates
                vid_cands = [c for c in candidates if c[1].lower().endswith(video_exts)]
                pick = None
                if vid_cands:
                    pick = sorted(vid_cands)[-1][1]
                elif candidates:
                    pick = sorted(candidates)[-1][1]
                if pick:
                    annot_fname = pick.replace('\\', '/')
            except Exception:
                annot_fname = None

            if not annot_fname:
                annot_fname = os.path.basename(src_path)

            return jsonify({
                "video_url": f"/target/{annot_fname}",
                "stats": stats,
            })
        except Exception as e:
            app.logger.exception('/predict (video) failed')
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": f"unsupported media_type: {media_type}"}), 400


@app.route("/predict-frame", methods=["POST"])
def predict_frame():
    payload = request.get_json(silent=True) or {}
    start_ts = time.time()
    app.logger.info("/predict-frame: received request")
    image_data = payload.get("image")
    if not image_data:
        app.logger.warning("/predict-frame: missing image in payload")
        return jsonify({"error": "image missing"}), 400

    # accept data URLs or raw base64
    if isinstance(image_data, str) and image_data.startswith("data:"):
        try:
            image_b64 = image_data.split(",", 1)[1]
        except Exception:
            return jsonify({"error": "invalid data url"}), 400
    else:
        image_b64 = image_data

    try:
        decoded = base64.b64decode(image_b64)
        img = Image.open(BytesIO(decoded)).convert("RGB")
    except Exception as e:
        app.logger.exception("/predict-frame: invalid image data")
        return jsonify({"error": "invalid image data", "detail": str(e)}), 400

    # convert to numpy array for model.predict
    try:
        arr = np.asarray(img)
    except Exception:
        return jsonify({"error": "cannot convert image"}), 500

    results = inference_student_pose.run_inference(arr, TRY_TARGET, model_path=MODEL_PATH, imgsz=640, conf=0.25, iou=0.45, device=DEVICE, save=False)
    if not results:
        return jsonify({"error": "no results"}), 500

    model_for_names = inference_student_pose._MODEL_CACHE.get(MODEL_PATH)
    stats = summarize_results(results, getattr(model_for_names, 'names', None))
    stats = ensure_tracked_keys(stats)

    try:
        try:
            annotated = results[0].plot()
            out_img = Image.fromarray(annotated)
        except Exception:
            out_img = img

        buffer = BytesIO()
        out_img.save(buffer, format="PNG")

        # build normalized boxes for frontend drawing
        boxes_out = []
        try:
            r = results[0]
            b = getattr(r, 'boxes', None)
            if b is not None:
                try:
                    xyxy = b.xyxy.cpu().numpy()
                    confs = b.conf.cpu().numpy()
                    clsids = b.cls.cpu().numpy()
                except Exception:
                    xyxy, confs, clsids = [], [], []

                try:
                    img_w, img_h = out_img.size
                except Exception:
                    img_h = arr.shape[0] if hasattr(arr, 'shape') else 1
                    img_w = arr.shape[1] if hasattr(arr, 'shape') else 1

                class_names = getattr(model_for_names, 'names', {})
                for bb, cf, cid in zip(xyxy, confs, clsids):
                    x1, y1, x2, y2 = map(float, bb)
                    cid = int(cid)
                    # model.names 是 dict[int, str]，用 int(cid) 查找
                    if isinstance(class_names, (list, tuple)):
                        cname = class_names[cid] if cid < len(class_names) else str(cid)
                    elif isinstance(class_names, dict):
                        cname = class_names.get(cid, str(cid))
                    else:
                        cname = str(cid)
                    nx1 = max(0.0, min(1.0, x1 / (img_w or 1)))
                    ny1 = max(0.0, min(1.0, y1 / (img_h or 1)))
                    nx2 = max(0.0, min(1.0, x2 / (img_w or 1)))
                    ny2 = max(0.0, min(1.0, y2 / (img_h or 1)))
                    boxes_out.append({
                        'xyxy': [nx1, ny1, nx2, ny2],
                        'conf': float(cf),
                        'class_id': cid,
                        'class_name': str(cname)
                    })
        except Exception:
            boxes_out = []

        # 如果检测到异常行为，触发告警
        try:
            if stats.get('display_counts', {}).get('abnormal', 0) > 0:
                alert = {
                    'id': int(time.time() * 1000),
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'level': 'warning',
                    'type': 'abnormal',
                    'count': int(stats.get('display_counts', {}).get('abnormal', 0)),
                    'meta': {'source': 'predict-frame'},
                }
                push_alert(alert)
        except Exception:
            pass

        resp = jsonify({
            "image_base64": b64encode(buffer.getvalue()).decode("utf-8"),
            "stats": stats,
            "boxes": boxes_out,
        })
        elapsed = time.time() - start_ts
        app.logger.info(f"/predict-frame: done, elapsed={elapsed:.3f}s, detections={stats.get('total_detections',0)}")
        return resp
    except Exception as e:
        app.logger.exception("/predict-frame: unexpected error")
        return jsonify({"error": "internal error", "detail": str(e)}), 500


@app.route('/capture', methods=['POST', 'OPTIONS'])
def capture():
    try:
        start_ts = time.time()
        app.logger.info('/capture: received request')
        # handle preflight
        if request.method == 'OPTIONS':
            return ('', 200)

        """Accept a data URL or base64 image, save original + annotated image and stats under web/captures.
        Supports two request formats:
         - application/json with {"image": "data:..."}
         - text/plain body containing the data URL string (avoids preflight)
        """
        image_data = None
        ct = (request.headers.get('Content-Type') or '').lower()
        if ct.startswith('application/json'):
            payload = request.get_json(silent=True) or {}
            image_data = payload.get('image')
        elif ct.startswith('text/plain') or ct == '':
            # treat raw body as dataURL (useful to avoid preflight)
            raw = request.get_data(as_text=True)
            image_data = raw or None
        else:
            # fallback: try json
            payload = request.get_json(silent=True) or {}
            image_data = payload.get('image')

        if not image_data:
            return jsonify({'error': 'image missing'}), 400

        # accept data URLs or raw base64
        if isinstance(image_data, str) and image_data.startswith('data:'):
            try:
                image_b64 = image_data.split(',', 1)[1]
            except Exception:
                return jsonify({'error': 'invalid data url'}), 400
        else:
            image_b64 = image_data

        try:
            decoded = base64.b64decode(image_b64)
            img = Image.open(BytesIO(decoded)).convert('RGB')
        except Exception:
            return jsonify({'error': 'invalid image data'}), 400

        # save into structured capture subfolders
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
        orig_name = f'capture_{timestamp}.png'
        orig_path = os.path.join(CAPTURES_ORIGINALS, orig_name)
        try:
            img.save(orig_path)
        except Exception:
            pass

        # run inference on numpy array (do not auto-save to TRY_TARGET here)
        try:
            arr = np.asarray(img)
        except Exception:
            app.logger.exception('/capture: cannot convert image to array')
            return jsonify({'error': 'cannot convert image'}), 500

        results = inference_student_pose.run_inference(arr, TRY_TARGET, model_path=MODEL_PATH, imgsz=640, conf=0.25, iou=0.45, device=DEVICE, save=False)
        if not results:
            return jsonify({'error': 'no results'}), 500

        model_for_names = inference_student_pose._MODEL_CACHE.get(MODEL_PATH)
        stats = summarize_results(results, getattr(model_for_names, 'names', None))
        stats = ensure_tracked_keys(stats)

        try:
            annotated = results[0].plot()
            out_img = Image.fromarray(annotated)
        except Exception:
            out_img = img

        annot_name = f'capture_annot_{timestamp}.png'
        annot_path = os.path.join(CAPTURES_ANNOTATED, annot_name)
        try:
            out_img.save(annot_path)
        except Exception:
            pass

        # save stats json
        try:
            stats_name = f'capture_stats_{timestamp}.json'
            stats_path = os.path.join(CAPTURES_STATS, stats_name)
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # if abnormal detected, push alert
        try:
            if stats.get('display_counts', {}).get('abnormal', 0) > 0:
                alert = {
                    'id': int(time.time() * 1000),
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'level': 'warning',
                    'type': 'abnormal',
                    'count': int(stats.get('display_counts', {}).get('abnormal', 0)),
                    'meta': {'source': 'capture', 'file': orig_name},
                }
                # copy annotated into alerts folder for easy access and attach base64
                try:
                    alert_fname = f'alert_annot_{timestamp}.png'
                    alert_path = os.path.join(CAPTURES_ALERTS, alert_fname)
                    shutil.copy2(annot_path, alert_path)
                    with open(alert_path, 'rb') as f:
                        alert['image_base64'] = b64encode(f.read()).decode('utf-8')
                    alert['meta']['file'] = alert_fname
                except Exception:
                    pass
                push_alert(alert)
        except Exception:
            pass

        # return relative URLs (static folder is WEB_DIR)
        elapsed = time.time() - start_ts
        app.logger.info(f"/capture: done, elapsed={elapsed:.3f}s, detections={stats.get('total_detections',0)}")
        return jsonify({
            'ok': True,
            'original': f'/captures/originals/{orig_name}',
            'annotated': f'/captures/annotated/{annot_name}',
            'stats': stats,
            'stats_url': f'/captures/stats/{stats_name}'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/attendance/mark', methods=['POST'])
def attendance_mark():
    """Receive an attendance mark payload and save to attendance_records.json
    Expected JSON: { original: str, annotated: str, stats: dict, meta?: dict }
    """
    try:
        payload = request.get_json(silent=True) or {}
        if not payload:
            return jsonify({'error': 'missing json payload'}), 400

        record = {
            'id': int(time.time() * 1000),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'original': payload.get('original'),
            'annotated': payload.get('annotated'),
            'stats': payload.get('stats') or {},
            'meta': payload.get('meta') or {}
        }

        with attendance_lock:
            attendance_store.insert(0, record)
            # keep recent 2000
            del attendance_store[2000:]
            save_attendance()

        return jsonify({'ok': True, 'record': record}), 201
    except Exception as e:
        app.logger.exception('/attendance/mark failed')
        return jsonify({'error': str(e)}), 500


@app.route('/attendance/records', methods=['GET'])
def attendance_records():
    try:
        with attendance_lock:
            return jsonify({'records': attendance_store})
    except Exception as e:
        app.logger.exception('/attendance/records failed')
        return jsonify({'error': str(e)}), 500
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True, threaded=True)
