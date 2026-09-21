import base64
import io
import os
import shutil
import time
import wave
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import psutil
import streamlit as st

# Must be the very first Streamlit command executed
st.set_page_config(
    page_title="CyberVision",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    YOLO_AVAILABLE = False

try:
    import yara
    YARA_AVAILABLE = True
except Exception:
    yara = None
    YARA_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    ollama = None
    OLLAMA_AVAILABLE = False

try:
    from google.cloud import storage
    GCP_AVAILABLE = True
except Exception:
    storage = None
    GCP_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG: Dict[str, Any] = {
    "APP_NAME": "CyberVision",
    "APP_SUBTITLE": "Adaptive Context Optimization at the Edge",
    "VLM_MODEL": "moondream",
    "CAMERA_INDEX": 0,
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 300,
    "JPEG_QUALITY": 70,
    "ANALYZE_EVERY_N_FRAMES": 30, 
    "CPU_THRESHOLD": 90.0,
    "RAM_THRESHOLD": 85.0,
    "ALERT_HISTORY_MAX": 100,
    "METRICS_HISTORY_MAX": 150,
    "LATENCY_HISTORY_MAX": 100,
    "YARA_RULE_PATH": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "hazard_rules.yar",
    ),
    "GCP_BUCKET": "your-gcp-bucket-name",
    "YOLO_MODEL_PATH": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "YOLO_dataset.pt",
    ),
    "YOLO_CONFIDENCE": 0.35,
    "YOLO_IMAGE_SIZE": 320,
    "INFERENCE_TIMEOUT": 2.0,
    "WARMUP_FRAMES": 8,
    "LIVE_LOOP_SLEEP_SECONDS": 0.03,
}

SEVERITY_STYLE = {
    "NORMAL": {"score": 0, "icon": "✅", "color": "#30d158"},
    "MEDIUM": {"score": 1, "icon": "🟡", "color": "#ffd60a"},
    "HIGH": {"score": 2, "icon": "⚠️", "color": "#ff9500"},
    "CRITICAL": {"score": 3, "icon": "🚨", "color": "#ff3b30"},
}


# =============================================================================
# SESSION STATE & THEME
# =============================================================================
def init_session_state() -> None:
    defaults = {
        "dark_mode": False,
        "sound_enabled": True,
        "camera_running": False,
        "last_frame": None,
        "last_frame_rgb": None,
        "frames_processed": 0,
        "last_analyzed_frame": 0,
        "inference_count": 0,
        "alert_history": deque(maxlen=CONFIG["ALERT_HISTORY_MAX"]),
        "metrics_history": deque(maxlen=CONFIG["METRICS_HISTORY_MAX"]),
        "latency_history": deque(maxlen=CONFIG["LATENCY_HISTORY_MAX"]),
        "system_status": "STANDBY",
        "vlm_context": "READY",
        "last_error": "",
        "last_public_notice": "",
        "cloud_sync_enabled": False,
        "fps": 0,
        "last_frame_time": time.time(),
        "latest_detection": None,
        "yolo_detector": None,
        "yolo_detection": None,
        "last_vlm_candidate_frame": None,
        "latest_vlm_text": "",
        "alarm_active": False,
        "current_cpu": 0.0,
        "current_ram": 0.0,
        "last_severity": "NORMAL",
        "camera_fail_streak": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_theme() -> None:
    bg = "#fff7ed"
    panel_warm = "#fff3e6"
    text = "#1f2937"
    muted = "#6b7280"
    accent = "#f97316"
    accent_dark = "#c2410c"
    border = "#fed7aa"

    st.markdown(
        f"""
        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        div[data-baseweb="select"] {{
            border-radius: 14px !important;
        }}

        div[data-baseweb="select"] > div {{
            background: rgba(255,255,255,0.95) !important;
            border: 1px solid #fed7aa !important;
            color: #c2410c !important;
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(249,115,22,0.18), transparent 32%),
                linear-gradient(135deg, {bg} 0%, #ffffff 45%, #fff1e6 100%);
            color: {text};
        }}

        [data-testid="stAppViewContainer"] {{
            background: transparent;
        }}

        .main .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1400px;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #ffffff 0%, {panel_warm} 100%);
            border-right: 1px solid {border};
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: {accent_dark} !important;
            font-weight: 900 !important;
        }}

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {{
            color: {text};
        }}

        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: {text};
        }}

        .muted {{
            color: {muted};
        }}

        .hero {{
            position: relative;
            padding: 0.8rem 1.2rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,237,213,0.96)),
                radial-gradient(circle at top right, rgba(249,115,22,0.28), transparent 35%);
            border: 1px solid {border};
            border-radius: 26px;
            margin-bottom: 1.2rem;
            box-shadow: 0 14px 36px rgba(249,115,22,0.13);
            overflow: hidden;
        }}

        .hero-title {{
            font-size: 2.25rem;
            font-weight: 900;
            margin: 0.35rem 0 0 0;
            color: {accent_dark};
            letter-spacing: -0.04em;
        }}

        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(249,115,22,0.12);
            border: 1px solid rgba(249,115,22,0.35);
            color: {accent_dark};
            font-weight: 900;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        .status-card {{
            padding: 1.3rem;
            background: rgba(255,255,255,0.94);
            border: 1px solid {border};
            border-radius: 24px;
            box-shadow: 0 12px 30px rgba(249,115,22,0.12);
            margin-bottom: 1rem;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.94);
            border: 1px solid {border};
            border-radius: 14px;
            padding: 0.25rem 0.5rem;
            min-height: 65px;
            box-shadow: 0 8px 18px rgba(249,115,22,0.08);
        }}

        div[data-testid="stMetricLabel"] {{
            font-size: 0.70rem !important;
        }}

        div[data-testid="stMetric"] label {{
            color: {muted};
            font-weight: 700;
        }}

        div[data-testid="stMetricValue"] {{
            color: {accent_dark};
            font-weight: 900;
            font-size: 1.25rem !important;
        }}  

        .stButton > button {{
            border-radius: 14px;
            font-weight: 800;
            border: 1px solid #fb923c;
            background: linear-gradient(135deg, #fff7ed, #ffedd5);
            color: {accent_dark};
            box-shadow: 0 8px 18px rgba(249,115,22,0.12);
            transition: all 0.18s ease-in-out;
        }}

        .stButton > button:hover {{
            background: linear-gradient(135deg, #fed7aa, #fdba74);
            color: #7c2d12;
            border-color: {accent};
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(249,115,22,0.22);
        }}

        button[data-baseweb="tab"] {{
            border-radius: 14px 14px 0 0;
            font-weight: 800;
            color: {muted};
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {accent_dark};
            background: rgba(249,115,22,0.10);
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {border};
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(249,115,22,0.08);
        }}

        div[data-testid="stAlert"] {{
            border-radius: 18px;
            border: 1px solid {border};
        }}

        [data-testid="stImage"] img {{
            border-radius: 22px;
            border: 1px solid {border};
            box-shadow: 0 12px 32px rgba(249,115,22,0.16);
        }}

        .stDownloadButton > button {{
            border-radius: 14px;
            font-weight: 800;
            background: #fff7ed;
            color: {accent_dark};
            border: 1px solid #fb923c;
        }}

        .stDownloadButton > button:hover {{
            background: #fed7aa;
            color: #7c2d12;
        }}

        hr {{
            border-color: {border};
        }}

        ::selection {{
            background: rgba(249,115,22,0.25);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# ALERT SOUND
# =============================================================================
def build_beep_wav_base64(
    duration: float = 0.35,
    freq: int = 950,
    sample_rate: int = 44100
) -> str:
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(freq * t * 2 * np.pi)
    envelope = np.linspace(1, 0.15, tone.size)
    audio = (tone * envelope * 32767).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def play_alert_sound(loop: bool = False, stop: bool = False) -> None:
    if stop:
        st.components.v1.html(
            """
            <script>
            try {
                const players = window.parent.cybervisionAudioPlayers || [];
                players.forEach(audio => {
                    audio.pause();
                    audio.currentTime = 0;
                });
                window.parent.cybervisionAudioPlayers = [];
            } catch (e) {}
            </script>
            """,
            height=0,
        )
        return

    if not st.session_state.sound_enabled:
        return

    audio_b64 = build_beep_wav_base64(
        duration=0.45 if loop else 0.35,
        freq=1100 if loop else 950,
    )

    st.components.v1.html(
        f"""
        <script>
        try {{
            window.parent.cybervisionAudioPlayers = window.parent.cybervisionAudioPlayers || [];
            window.parent.cybervisionAudioPlayers.forEach(a => {{
                a.pause();
                a.currentTime = 0;
            }});
            window.parent.cybervisionAudioPlayers = [];
            const audio = new Audio("data:audio/wav;base64,{audio_b64}");
            audio.volume = 1.0;
            audio.loop = {"true" if loop else "false"};
            window.parent.cybervisionAudioPlayers.push(audio);
            audio.play().catch(() => {{}});
        }} catch (e) {{}}
        </script>
        """,
        height=0,
    )


# =============================================================================
# SYSTEM MONITORING
# =============================================================================
class SystemMonitor:
    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        try:
            cpu = float(psutil.cpu_percent(interval=None))
            ram = float(psutil.virtual_memory().percent)

            st.session_state.current_cpu = cpu
            st.session_state.current_ram = ram

            metrics = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "cpu": cpu,
                "ram": ram,
            }

            st.session_state.metrics_history.append(metrics)
            return metrics

        except Exception:
            return {
                "time": datetime.now().strftime("%H:%M:%S"),
                "cpu": 0.0,
                "ram": 0.0,
            }


class ResourceGovernor:
    @staticmethod
    def check_resource_pressure() -> None:
        metrics = SystemMonitor.get_metrics()
        cpu = metrics["cpu"]
        ram = metrics["ram"]

        if cpu >= CONFIG["CPU_THRESHOLD"] or ram >= CONFIG["RAM_THRESHOLD"]:
            st.session_state.system_status = "RESOURCE_PRESSURE"
            st.session_state.vlm_context = "REDUCED"
            st.session_state.last_public_notice = (
                f"Resource pressure detected. CPU={cpu:.1f}%, RAM={ram:.1f}%. "
                "Using reduced context."
            )
        else:
            st.session_state.system_status = "HEALTHY"
            st.session_state.vlm_context = "ACTIVE"
            st.session_state.last_public_notice = ""

    @staticmethod
    def recommended_token_budget() -> int:
        cpu = st.session_state.get("current_cpu", 0.0)
        ram = st.session_state.get("current_ram", 0.0)
        worst = max(cpu, ram)

        if worst >= 85:
            return 128
        if worst >= 70:
            return 256
        if worst >= 50:
            return 512

        return 1024


# =============================================================================
# CAMERA HANDLING
# =============================================================================
@st.cache_resource(show_spinner="Initializing camera hardware...")
def get_camera() -> cv2.VideoCapture:
    backend_candidates = [cv2.CAP_ANY]
    if os.name == "nt":
        backend_candidates = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]

    cap = None
    for backend in backend_candidates:
        candidate = cv2.VideoCapture(CONFIG["CAMERA_INDEX"], backend)
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()

    if cap is None:
        cap = cv2.VideoCapture(CONFIG["CAMERA_INDEX"])

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["FRAME_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def pause_camera() -> None:
    st.session_state.camera_running = False


def release_camera() -> None:
    try:
        InferenceWorker.stop() if "InferenceWorker" in globals() else None
        VideoCaptureManager.stop_capture_thread() if "VideoCaptureManager" in globals() else None
        cap = get_camera()
        if cap and cap.isOpened():
            cap.release()
        get_camera.clear()
    except Exception:
        pass


class VideoCaptureManager:
    _thread = None
    _stop_event = None
    _lock = None
    _latest_frame = None
    _started = False

    @classmethod
    def _ensure_thread(cls):
        if cls._thread is not None and cls._thread.is_alive():
            return
        import threading
        cls._stop_event = threading.Event()
        cls._lock = threading.Lock()
        cls._started = True
        cls._thread = threading.Thread(
            target=cls._capture_loop,
            name="CyberVisionCameraCapture",
            daemon=True,
        )
        cls._thread.start()

    @classmethod
    def _capture_loop(cls):
        import time as _time
        while cls._stop_event is not None and not cls._stop_event.is_set():
            try:
                cap = get_camera()
                if not cap.isOpened():
                    _time.sleep(0.2)
                    continue
                ret, frame = cap.read()
                if not ret or frame is None:
                    _time.sleep(0.02)
                    continue
                frame = cv2.resize(frame, (CONFIG["FRAME_WIDTH"], CONFIG["FRAME_HEIGHT"]))
                with cls._lock:
                    cls._latest_frame = frame
            except Exception:
                _time.sleep(0.1)

    @classmethod
    def capture_frame(cls) -> Optional[np.ndarray]:
        cls._ensure_thread()
        if cls._lock is None:
            return None
        with cls._lock:
            if cls._latest_frame is None:
                return None
            return cls._latest_frame.copy()

    @classmethod
    def stop_capture_thread(cls):
        if cls._stop_event is not None:
            cls._stop_event.set()
        if cls._thread is not None and cls._thread.is_alive():
            cls._thread.join(timeout=1.0)
        cls._thread = None
        cls._stop_event = None
        cls._started = False
        if cls._lock is not None:
            with cls._lock:
                cls._latest_frame = None

    @staticmethod
    def placeholder_frame(message: str, subtext: str = "") -> np.ndarray:
        frame = np.zeros((CONFIG["FRAME_HEIGHT"], CONFIG["FRAME_WIDTH"], 3), dtype=np.uint8)
        frame[:] = (22, 28, 55)
        cv2.putText(frame, message, (55, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 170), 2)
        if subtext:
            cv2.putText(frame, subtext, (55, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# =============================================================================
# IMAGE ENCODING
# =============================================================================
def frame_to_base64_jpeg(frame: np.ndarray) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG["JPEG_QUALITY"]]
    ok, buffer = cv2.imencode(".jpg", frame, encode_param)

    if not ok:
        raise ValueError("Failed to encode frame.")

    return base64.b64encode(buffer).decode("utf-8")


# =============================================================================
# YOLO FIRE & SMOKE DETECTION
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    if not YOLO_AVAILABLE:
        return None

    model_path = CONFIG["YOLO_MODEL_PATH"]
    if not os.path.exists(model_path):
        return None

    try:
        return YOLO(model_path)
    except Exception:
        return None


def is_yolo_model_available() -> bool:
    if not YOLO_AVAILABLE:
        return False

    model_path = CONFIG["YOLO_MODEL_PATH"]
    if not os.path.isfile(model_path):
        return False

    return load_yolo_model() is not None


class YOLOFireSmokeDetector:
    def __init__(self) -> None:
        self.model = load_yolo_model()

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        empty = {
            "detected": False,
            "detections": [],
            "highest_confidence": 0.0,
            "classes": [],
            "annotated_frame": frame,
            "error": None,
        }

        if frame is None:
            return empty
        if self.model is None:
            empty_no_model = dict(empty)
            empty_no_model["error"] = "YOLO model is not loaded (self.model is None)."
            return empty_no_model

        try:
            results = self.model(
                frame,
                imgsz=CONFIG["YOLO_IMAGE_SIZE"],
                conf=CONFIG["YOLO_CONFIDENCE"],
                verbose=False,
            )

            if not results:
                return empty

            result = results[0]
            detections = []
            names = result.names if hasattr(result, "names") else self.model.names

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    if isinstance(names, dict):
                        class_name = str(names.get(cls_id, cls_id)).lower()
                    else:
                        class_name = str(names[cls_id]).lower()

                    xyxy = box.xyxy[0].tolist()

                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "box": [int(v) for v in xyxy],
                    })

            annotated = result.plot() if detections else frame

            return {
                "detected": bool(detections),
                "detections": detections,
                "highest_confidence": (
                    max(d["confidence"] for d in detections)
                    if detections else 0.0
                ),
                "classes": sorted(set(d["class"] for d in detections)),
                "annotated_frame": annotated,
                "error": None,
            }

        except Exception as exc:
            errored = dict(empty)
            errored["error"] = f"{type(exc).__name__}: {exc}"
            return errored

    @staticmethod
    def hazard_severity(yolo_result: Dict[str, Any]) -> str:
        if not yolo_result.get("detected"):
            return "NORMAL"

        highest = "NORMAL"

        for detection in yolo_result.get("detections", []):
            class_name = detection["class"].lower()
            confidence = detection["confidence"]

            if "fire" in class_name or "flame" in class_name:
                candidate = "CRITICAL" if confidence >= 0.80 else "HIGH"
            elif "smoke" in class_name:
                candidate = "HIGH" if confidence >= 0.80 else "MEDIUM"
            else:
                candidate = "MEDIUM" if confidence >= 0.50 else "NORMAL"

            if SEVERITY_STYLE[candidate]["score"] > SEVERITY_STYLE[highest]["score"]:
                highest = candidate

        return highest


# =============================================================================
# VISUAL FALLBACK DETECTION
# =============================================================================
class VisualFireSmokeDetector:
    @staticmethod
    def detect(frame: np.ndarray) -> Dict[str, Any]:
        if frame is None:
            return {
                "visual_severity": "NORMAL",
                "visual_keyword": "",
                "fire_ratio": 0.0,
                "smoke_ratio": 0.0,
            }

        bgr = frame
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

        b, g, r = cv2.split(bgr)
        h, s, v = cv2.split(hsv)

        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        red_or_orange_hue = ((h <= 25) | (h >= 170))
        bright_saturated = (s >= 120) & (v >= 160)
        rgb_dominance = (r > 160) & (r > g * 1.15) & (r > b * 1.65) & (g > 45)
        non_skin = skin_mask == 0

        fire_pixels = red_or_orange_hue & bright_saturated & rgb_dominance & non_skin
        fire_mask = fire_pixels.astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_DILATE, kernel)

        low_saturation = s < 45
        mid_brightness = (v > 90) & (v < 210)
        gray_dominance = (
            (np.abs(r.astype(int) - g.astype(int)) < 25)
            & (np.abs(g.astype(int) - b.astype(int)) < 25)
        )

        smoke_pixels = low_saturation & mid_brightness & gray_dominance
        smoke_mask = smoke_pixels.astype(np.uint8) * 255

        total_pixels = frame.shape[0] * frame.shape[1]

        fire_ratio = float(cv2.countNonZero(fire_mask)) / total_pixels
        smoke_ratio = float(cv2.countNonZero(smoke_mask)) / total_pixels

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            fire_mask,
            connectivity=8,
        )

        largest_fire_area = 0
        if num_labels > 1:
            largest_fire_area = int(stats[1:, cv2.CC_STAT_AREA].max())

        largest_fire_ratio = largest_fire_area / total_pixels

        if fire_ratio >= 0.015 and largest_fire_ratio >= 0.018:
            severity = "CRITICAL"
            keyword = "visual_fire_critical"
        elif fire_ratio >= 0.008 and largest_fire_ratio >= 0.010:
            severity = "HIGH"
            keyword = "visual_fire_high"
        elif fire_ratio >= 0.003:
            severity = "MEDIUM"
            keyword = "visual_fire_medium"
        elif smoke_ratio >= 0.18:
            severity = "CRITICAL"
            keyword = "visual_smoke_critical"
        elif smoke_ratio >= 0.10:
            severity = "HIGH"
            keyword = "visual_smoke_high"
        elif smoke_ratio >= 0.05:
            severity = "MEDIUM"
            keyword = "visual_smoke_medium"
        else:
            severity = "NORMAL"
            keyword = ""

        return {
            "visual_severity": severity,
            "visual_keyword": keyword,
            "fire_ratio": fire_ratio,
            "smoke_ratio": smoke_ratio,
        }


# =============================================================================
# OLLAMA VLM INFERENCE
# =============================================================================
class VLMInference:
    @staticmethod
    def run_inference(frame: np.ndarray, token_budget: Optional[int] = None) -> Tuple[str, bool]:
        if not OLLAMA_AVAILABLE:
            return "NORMAL: VLM unavailable. No text-based hazard confirmed.", False
        try:
            if token_budget is None:
                token_budget = 128
            image_b64 = frame_to_base64_jpeg(frame)
            response = ollama.chat(
                model=CONFIG["VLM_MODEL"],
                messages=[{
                    "role": "user",
                    "content": (
                        "Analyze this camera frame for fire and smoke only. "
                        "Return exactly one category: NORMAL, MEDIUM, HIGH, or CRITICAL. "
                        "Use NORMAL when there is no visible fire or smoke. "
                        "Use MEDIUM only for light smoke or unclear early warning. "
                        "Use HIGH for visible flame, burning object, or clear smoke. "
                        "Use CRITICAL only for active fire, heavy smoke, visible flames, explosion, or immediate danger. "
                        "Do not classify people, skin, red clothing, walls, posters, or warm lighting as fire. "
                        "Start with the category, then one short reason."
                    ),
                    "images": [image_b64],
                }],
                options={"num_ctx": token_budget, "num_predict": 80, "temperature": 0.0},
            )
            text = response.get("message", {}).get("content", "").strip()
            if not text:
                return "NORMAL: VLM returned no confirmed fire or smoke.", False
            return text, True
        except Exception:
            return "NORMAL: VLM temporarily unavailable. No text-based fire or smoke confirmed.", False


# =============================================================================
# YARA VERIFICATION
# =============================================================================
@st.cache_resource(show_spinner=False)
def compile_yara_rules():
    if not YARA_AVAILABLE:
        return None

    if not os.path.exists(CONFIG["YARA_RULE_PATH"]):
        st.session_state.last_error = (
            f"hazard_rules.yar not found at expected path: {CONFIG['YARA_RULE_PATH']}"
        )
        return None

    try:
        return yara.compile(filepath=CONFIG["YARA_RULE_PATH"])
    except Exception as exc:
        st.session_state.last_error = f"YARA failed to compile hazard_rules.yar: {exc}"
        return None


class YARAVerifier:
    @staticmethod
    def verify(vlm_text: str, visual_result: Dict[str, Any]) -> Dict[str, Any]:
        yara_severity = "NORMAL"
        matched_rule = "no_yara_match"
        matched_source = "YARA_TEXT"
        yara_error = None

        if YARA_AVAILABLE:
            try:
                rules = compile_yara_rules()
                if rules is None:
                    matched_rule = "yara_rules_not_compiled"
                    yara_error = st.session_state.get("last_error") or "YARA rules failed to compile (see sidebar)."
                else:
                    matches = rules.match(data=vlm_text.encode("utf-8", errors="ignore"))
                    if matches:
                        highest_score = -1
                        for match in matches:
                            severity = match.meta.get("severity", "NORMAL")
                            score = SEVERITY_STYLE.get(severity, SEVERITY_STYLE["NORMAL"])["score"]

                            if score > highest_score:
                                highest_score = score
                                yara_severity = severity
                                matched_rule = match.rule
            except Exception as exc:
                yara_severity = "NORMAL"
                matched_rule = "yara_verification_error"
                yara_error = f"{type(exc).__name__}: {exc}"
        else:
            matched_rule = "yara_not_installed"

        visual_severity = visual_result.get("visual_severity", "NORMAL")
        yara_score = SEVERITY_STYLE[yara_severity]["score"]
        visual_score = SEVERITY_STYLE[visual_severity]["score"]

        if visual_score > yara_score:
            final_severity = visual_severity
            matched_source = "OPENCV_VISUAL_FALLBACK"
            matched_keyword = visual_result.get("visual_keyword", "visual_detection")
        else:
            final_severity = yara_severity
            matched_keyword = matched_rule

        is_hazard = final_severity != "NORMAL"
        confidence = {
            "NORMAL": 0.10,
            "MEDIUM": 0.64,
            "HIGH": 0.82,
            "CRITICAL": 0.92,
        }[final_severity]

        return {
            "severity": final_severity,
            "is_hazard": is_hazard,
            "confidence": confidence,
            "matched_keyword": matched_keyword,
            "matched_source": matched_source,
            "yara_severity": yara_severity,
            "visual_severity": visual_severity,
            "fire_ratio": visual_result.get("fire_ratio", 0.0),
            "smoke_ratio": visual_result.get("smoke_ratio", 0.0),
            "yara_available": YARA_AVAILABLE,
            "yara_error": yara_error,
        }


# =============================================================================
# GOOGLE CLOUD FORENSIC BACKUP
# =============================================================================
def upload_to_google_cloud(frame: np.ndarray, alert: Dict[str, Any], enabled: Optional[bool] = None) -> None:
    if enabled is None:
        enabled = st.session_state.cloud_sync_enabled
    if not enabled or not GCP_AVAILABLE:
        return

    if CONFIG["GCP_BUCKET"] == "your-gcp-bucket-name":
        return

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        severity = alert.get("severity", "UNKNOWN")

        image_name = f"cybervision_alerts/{severity}_{timestamp}.jpg"
        log_name = f"cybervision_alerts/{severity}_{timestamp}.json"

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return

        client = storage.Client()
        bucket = client.bucket(CONFIG["GCP_BUCKET"])

        image_blob = bucket.blob(image_name)
        image_blob.upload_from_string(buffer.tobytes(), content_type="image/jpeg")

        log_blob = bucket.blob(log_name)
        log_blob.upload_from_string(str(alert), content_type="application/json")

    except Exception:
        pass


# =============================================================================
# BACKGROUND INFERENCE WORKER
# =============================================================================
class InferenceWorker:
    """Runs all expensive detection work away from Streamlit's render loop.

    The camera/UI never waits for YOLO, Ollama, YARA, or cloud upload. Only the
    newest camera frame is queued, so slow inference can never create a backlog.
    """
    _thread = None
    _stop_event = None
    _input = None
    _output = None
    _lock = None
    _frame_counter = 0
    _last_vlm_candidate_frame = None
    _latest_vlm_text = ""
    _inference_count = 0

    @classmethod
    def start(cls, yolo_detector=None, cloud_sync=False):
        import threading, queue
        if cls._thread is not None and cls._thread.is_alive():
            return
        cls._stop_event = threading.Event()
        cls._input = queue.Queue(maxsize=1)
        cls._output = queue.Queue(maxsize=2)
        cls._lock = threading.Lock()
        cls._frame_counter = 0
        cls._last_vlm_candidate_frame = None
        cls._latest_vlm_text = ""
        cls._inference_count = 0
        cls._thread = threading.Thread(
            target=cls._loop,
            args=(yolo_detector, cloud_sync),
            name="CyberVisionInferenceWorker",
            daemon=True,
        )
        cls._thread.start()

    @classmethod
    def submit(cls, frame):
        if cls._input is None or cls._thread is None or not cls._thread.is_alive():
            return
        try:
            while True:
                try:
                    cls._input.get_nowait()
                except Exception:
                    break
            cls._input.put_nowait(frame.copy())
        except Exception:
            pass

    @classmethod
    def poll(cls):
        if cls._output is None:
            return None
        latest = None
        try:
            while True:
                latest = cls._output.get_nowait()
        except Exception:
            pass
        return latest

    @classmethod
    def stop(cls):
        if cls._stop_event is not None:
            cls._stop_event.set()
        if cls._thread is not None and cls._thread.is_alive():
            cls._thread.join(timeout=1.5)
        cls._thread = None
        cls._stop_event = None
        cls._input = None
        cls._output = None

    @staticmethod
    def _token_budget():
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            worst = max(cpu, ram)
            if worst >= 85:
                return 128
            if worst >= 70:
                return 256
            if worst >= 50:
                return 512
            return 1024
        except Exception:
            return 256

    @classmethod
    def _loop(cls, yolo_detector, cloud_sync):
        import time as _time
        while cls._stop_event is not None and not cls._stop_event.is_set():
            try:
                frame = cls._input.get(timeout=0.1)
            except Exception:
                continue
            try:
                cls._frame_counter += 1
                frame_number = cls._frame_counter
                start_time = _time.time()

                visual_result = VisualFireSmokeDetector.detect(frame)
                if yolo_detector is None:
                    yolo_detector = YOLOFireSmokeDetector()
                yolo_result = yolo_detector.detect(frame)
                yolo_severity = yolo_detector.hazard_severity(yolo_result)
                if SEVERITY_STYLE[yolo_severity]["score"] > SEVERITY_STYLE[visual_result["visual_severity"]]["score"]:
                    visual_result["visual_severity"] = yolo_severity
                    if yolo_result.get("detections"):
                        first = yolo_result["detections"][0]
                        visual_result["visual_keyword"] = f"yolo_{first['class']}_{first['confidence']:.2f}"
                candidate = yolo_result.get("detected", False) or visual_result.get("visual_severity") != "NORMAL"
                analyze_vlm = candidate and (cls._last_vlm_candidate_frame is None or frame_number - cls._last_vlm_candidate_frame >= max(1, CONFIG["ANALYZE_EVERY_N_FRAMES"]))
                if analyze_vlm:
                    vlm_text, vlm_success = VLMInference.run_inference(frame, cls._token_budget())
                    cls._last_vlm_candidate_frame = frame_number
                    cls._latest_vlm_text = vlm_text
                    cls._inference_count += 1
                elif candidate:
                    vlm_text, vlm_success = cls._latest_vlm_text or "NORMAL: Waiting for VLM confirmation.", bool(cls._latest_vlm_text)
                else:
                    vlm_text, vlm_success = "NORMAL: YOLO detected no fire or smoke candidate.", True
                verdict = YARAVerifier.verify(vlm_text, visual_result)
                latency = _time.time() - start_time
                detection_text = ", ".join(f"{d['class']} {d['confidence']:.0%}" for d in yolo_result.get("detections", []))
                public_description = vlm_text
                if yolo_result.get("detected") and (not vlm_success or not analyze_vlm):
                    public_description = f"YOLO detected: {detection_text}. VLM confirmation: {vlm_text}"
                if not vlm_success and verdict["severity"] == "NORMAL":
                    public_description = "VLM temporarily unavailable. No confirmed fire or smoke detected from fallback analysis."
                alert = {"timestamp":datetime.now().strftime("%H:%M:%S"),"description":public_description,"success":vlm_success,"latency":latency,"verified":verdict["is_hazard"],"yolo_detected":yolo_result.get("detected",False),"yolo_classes":", ".join(yolo_result.get("classes",[])),"yolo_confidence":yolo_result.get("highest_confidence",0.0),**verdict}
                if alert["is_hazard"] and cloud_sync:
                    try:
                        upload_to_google_cloud(frame, alert, enabled=cloud_sync)
                    except Exception:
                        pass

                result = {"annotated_frame": yolo_result.get("annotated_frame", frame), "yolo": yolo_result, "alert": alert, "inference_count": cls._inference_count}
                try:
                    while cls._output.full():
                        cls._output.get_nowait()
                    cls._output.put_nowait(result)
                except Exception:
                    pass
            except Exception:
                continue


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar() -> Any:
    resource_metrics_placeholder = None

    with st.sidebar:

        if st.session_state.camera_running:
            st.success("🟢 Camera running")
        else:
            st.warning("🔴 Camera stopped")

        st.caption("Detection runs automatically when the camera is started.")

        st.session_state.sound_enabled = st.toggle(
            "Alert sound",
            value=st.session_state.sound_enabled,
        )

        st.session_state.cloud_sync_enabled = st.toggle(
            "Sync Data",
            value=st.session_state.cloud_sync_enabled,
        )
        st.markdown(" ")
        st.markdown("## Model")

        ollama_available = shutil.which("ollama") is not None
        st.caption(f"Ollama package: {'Available' if ollama_available else 'Unavailable'}")
        st.caption(f"YARA package: {'Available' if YARA_AVAILABLE else 'Unavailable'}")
        st.caption(f"VLM: {CONFIG['VLM_MODEL']}")

        yolo_path = CONFIG["YOLO_MODEL_PATH"]
        yolo_loaded = is_yolo_model_available()

        st.caption(f"YOLO fire/smoke: {'Available' if yolo_loaded else 'Unavailable'}")
        st.caption(f"YOLO model: {os.path.basename(yolo_path)}")

        timeout_option = st.selectbox(
            "Inference timeout target",
            ["1.0s", "2.0s", "3.0s", "5.0s"],
            index=1,
        )
        CONFIG["INFERENCE_TIMEOUT"] = float(timeout_option.replace("s", ""))

        st.markdown(" ")
        st.markdown("## Resources")

        resource_metrics_placeholder = st.empty()
        render_resource_metrics(resource_metrics_placeholder)

        if st.button("Reset logs", use_container_width=True):
            st.session_state.alert_history.clear()
            st.session_state.latency_history.clear()
            st.success("Logs reset.")

        if st.session_state.last_error:
            st.info("Model/YARA backend is not fully available. Visual fallback remains active.")

    return resource_metrics_placeholder


def render_resource_metrics(placeholder) -> None:
    cpu = st.session_state.get("current_cpu", 0.0)
    ram = st.session_state.get("current_ram", 0.0)

    cpu_delta = "🔴" if cpu >= CONFIG["CPU_THRESHOLD"] else ("🟡" if cpu >= 70.0 else "🟢")
    ram_delta = "🔴" if ram >= CONFIG["RAM_THRESHOLD"] else ("🟡" if ram >= 70.0 else "🟢")

    with placeholder.container():
        st.metric("CPU", f"{cpu:.1f}%", cpu_delta)
        st.metric("RAM", f"{ram:.1f}%", ram_delta)
        st.metric("Token Budget", ResourceGovernor.recommended_token_budget())
        st.metric("Inferences", st.session_state.inference_count)

        if st.session_state.latency_history:
            avg_latency = sum(st.session_state.latency_history) / len(st.session_state.latency_history)
            st.metric("Avg Latency", f"{avg_latency * 1000:.0f} ms")
        else:
            st.metric("Avg Latency", "—")


# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================
def get_latest_alert() -> Optional[Dict[str, Any]]:
    if not st.session_state.alert_history:
        return None
    return st.session_state.alert_history[-1]


def get_live_alert() -> Optional[Dict[str, Any]]:
    if not st.session_state.camera_running:
        return None
    return st.session_state.get("latest_detection")


def describe_hazard_word(alert: Dict[str, Any]) -> str:
    classes = (alert.get("yolo_classes") or "").lower()
    keyword = (alert.get("matched_keyword") or "").lower()
    description = (alert.get("description") or "").lower()
    text_blob = f"{classes} {keyword} {description}"

    has_fire = "fire" in text_blob or "flame" in text_blob
    has_smoke = "smoke" in text_blob

    if has_fire and has_smoke:
        return "FIRE & SMOKE"
    if has_fire:
        return "FIRE"
    if has_smoke:
        return "SMOKE"
    return "HAZARD"


def render_status_cards() -> None:
    latest = st.session_state.get("latest_detection") or {}
    severity = latest.get("severity", "NORMAL")
    total = len(st.session_state.alert_history)
    hazards = sum(1 for item in st.session_state.alert_history if item["is_hazard"])
    confidence = latest.get("confidence", 0.0) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Severity", f"{severity}")
    c2.metric("Frames Processed", st.session_state.frames_processed)
    c3.metric("Confidence", f"{confidence:.0f}%")
    c4.metric("Hazards", hazards, f"of {total} events")


def render_alert_center() -> None:
    st.markdown("### Alerts")
    latest = get_live_alert()
    severity = (latest or {}).get("severity", "NORMAL")

    if latest and severity in ("MEDIUM", "HIGH", "CRITICAL"):
        color = SEVERITY_STYLE[severity]["color"]
        hazard_word = describe_hazard_word(latest)
        badge_text = f"{severity} {hazard_word} DETECTED"
        st.markdown(
            f"""
            <div style="
            position:fixed;
            top:20px;
            right:20px;
            z-index:999999;
            background:{color};
            color:white;
            padding:15px 20px;
            border-radius:12px;
            font-weight:bold;
            box-shadow:0 8px 20px rgba(0,0,0,.3);
            animation: fadeIn 0.3s ease;
            ">
            {badge_text}
            </div>
            """,
            unsafe_allow_html=True
        )

    if not st.session_state.camera_running:
        st.info("Camera is stopped. No live alert.")
        return

    if not latest:
        st.info("Waiting for analysed frame...")
        return

    severity = latest.get("severity", "NORMAL")
    style = SEVERITY_STYLE[severity]

    if severity == "NORMAL":
        title = "Environment is Safe"
        user_message = "No fire or smoke detected."
        action_message = "No action is required."
    elif severity == "MEDIUM":
        title = "Possible Smoke Detected"
        user_message = "The system detected a possible early smoke warning."
        action_message = "Please monitor the area carefully."
    elif severity == "HIGH":
        title = "Fire or Smoke Detected"
        user_message = "The system detected clear signs of fire or smoke."
        action_message = "Please check the area immediately and prepare to evacuate."
    else:
        title = "DANGER — Critical Hazard"
        user_message = "Active fire or heavy smoke may be present."
        action_message = "Evacuate immediately and contact emergency support."

    st.markdown(
        f"""
        <div class="status-card" style="border-left:10px solid {style['color']};">
            <h1>{style['icon']} {severity}</h1>
            <h3>{title}</h3>
            <p>{user_message}</p>
            <p style="font-weight:bold;color:#c2410c;">{action_message}</p>
            <p class="muted">Last checked: {latest.get('timestamp', '--:--:--')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# LIVE VIDEO STREAM & RENDER LOOP
# =============================================================================
def render_video_frame(video_placeholder, status_placeholder) -> None:
    frame = VideoCaptureManager.capture_frame()
    if frame is None:
        status_placeholder.info("Starting camera...")
        return

    current_time = time.time()
    last_time = st.session_state.get("last_frame_time", current_time)
    fps = 1 / max(current_time - last_time, 0.001)
    st.session_state.fps = fps
    st.session_state.last_frame_time = current_time
    st.session_state.frames_processed += 1
    st.session_state.last_frame = frame
    st.session_state.last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    InferenceWorker.submit(frame)

    result = InferenceWorker.poll()
    if result is not None:
        previous = st.session_state.get("latest_detection") or {}
        alert = result.get("alert") or previous
        st.session_state.latest_detection = alert
        st.session_state.latest_inference_result = result
        st.session_state.yolo_detection = result.get("yolo", {})
        st.session_state.latest_vlm_text = result.get("alert", {}).get("description", st.session_state.get("latest_vlm_text", ""))
        st.session_state.inference_count = result.get("inference_count", st.session_state.get("inference_count", 0))
        st.session_state.latency_history.append(alert.get("latency", 0.0))

        previous_severity = st.session_state.get("last_severity", "NORMAL")
        current_severity = alert.get("severity", "NORMAL")
        if current_severity != previous_severity:
            if current_severity == "CRITICAL" and st.session_state.get("sound_enabled", True):
                if not st.session_state.get("alarm_active", False):
                    play_alert_sound(loop=True)
                    st.session_state.alarm_active = True
            elif st.session_state.get("alarm_active", False):
                play_alert_sound(stop=True)
                st.session_state.alarm_active = False
            if current_severity in ("MEDIUM", "HIGH") and st.session_state.get("sound_enabled", True):
                play_alert_sound(loop=False)
        st.session_state.last_severity = current_severity
        if current_severity == "NORMAL" and st.session_state.get("alarm_active", False):
            play_alert_sound(stop=True)
            st.session_state.alarm_active = False
        st.session_state.alert_history.append(alert)

    result = st.session_state.get("latest_inference_result") or {}
    yolo_result = result.get("yolo", {})
    severity = (st.session_state.get("latest_detection") or {}).get("severity", "NORMAL")

    display_frame = frame.copy()
    if yolo_result.get("detected") and result.get("annotated_frame") is not None:
        annotated = result["annotated_frame"]
        if annotated.shape == display_frame.shape:
            display_frame = annotated.copy()

    overlay = display_frame.copy()
    cv2.rectangle(overlay, (10, 10), (430, 82), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, display_frame, 0.50, 0, display_frame)
    cv2.putText(display_frame, f"FPS: {fps:.1f} | YOLO: {'DETECTED' if yolo_result.get('detected') else 'LIVE'}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
    cv2.putText(display_frame, f"INF: {st.session_state.get('inference_count', 0)} | STATUS: {severity}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
    video_placeholder.image(display_frame, channels="BGR", use_container_width=True)

    yara_error = (st.session_state.get("latest_detection") or {}).get("yara_error")

    if not YOLO_AVAILABLE:
        status_placeholder.warning("Ultralytics is not installed. Run: pip install ultralytics")
    elif not is_yolo_model_available():
        status_placeholder.warning(f"YOLO model not found at: {CONFIG['YOLO_MODEL_PATH']}")
    elif yolo_result.get("error"):
        status_placeholder.error(f"YOLO inference error: {yolo_result['error']}")
    elif yara_error:
        status_placeholder.warning(f"YARA issue: {yara_error}")
    elif yolo_result.get("detected"):
        detections = ", ".join(f"{d['class']} ({d['confidence']:.0%})" for d in yolo_result.get("detections", []))
        status_placeholder.warning(f"YOLO detection: {detections}")
    else:
        status_placeholder.success("Live YOLO fire/smoke monitoring running...")


def _render_stopped_state(video_placeholder, status_placeholder) -> None:
    if st.session_state.get("last_frame_rgb") is not None:
        video_placeholder.image(st.session_state.last_frame_rgb, use_container_width=True)
    else:
        video_placeholder.image(
            VideoCaptureManager.placeholder_frame("Camera Stopped", "Click Start to begin monitoring."),
            use_container_width=True,
        )
    status_placeholder.info("Camera is stopped. Detection paused.")


def render_alert_center_into(placeholder) -> None:
    with placeholder.container():
        render_alert_center()


def render_video_controls() -> None:
    _, btn_col1, btn_col2 = st.columns([5, 1, 1])

    with btn_col1:
        if st.button("▶ Start", use_container_width=True, disabled=st.session_state.camera_running):
            st.session_state.camera_running = True
            st.session_state.frames_processed = 0
            st.session_state.latest_vlm_text = "NORMAL: Camera initialized."
            st.session_state.latest_detection = None
            st.session_state.last_vlm_candidate_frame = None
            st.session_state.last_error = ""
            st.session_state.last_frame_time = time.time()
            st.session_state.alarm_active = False
            st.session_state.last_severity = "NORMAL"
            st.session_state.latest_inference_result = None
            st.session_state.inference_count = 0
            InferenceWorker.start(st.session_state.get("yolo_detector"), st.session_state.get("cloud_sync_enabled", False))
            st.rerun()

    with btn_col2:
        if st.button("⏹ Stop", use_container_width=True, disabled=not st.session_state.camera_running):
            if st.session_state.get("alarm_active", False):
                play_alert_sound(stop=True)
                st.session_state.alarm_active = False

            InferenceWorker.stop()
            VideoCaptureManager.stop_capture_thread()

            pause_camera()
            st.rerun()


def render_live_monitor(resource_metrics_placeholder, status_cards_placeholder) -> None:
    render_video_controls()

    cam_col, alert_col = st.columns([2, 1])
    with cam_col:
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    with alert_col:
        alert_placeholder = st.empty()

    if not st.session_state.camera_running:
        _render_stopped_state(video_placeholder, status_placeholder)
        render_alert_center_into(alert_placeholder)
        return

    while st.session_state.camera_running:
        render_video_frame(video_placeholder, status_placeholder)
        render_alert_center_into(alert_placeholder)
        with status_cards_placeholder.container():
            render_status_cards()
        if resource_metrics_placeholder is not None:
            render_resource_metrics(resource_metrics_placeholder)
        time.sleep(CONFIG["LIVE_LOOP_SLEEP_SECONDS"])


def render_forensic_log() -> None:
    st.markdown("### Alert Timeline")

    if not st.session_state.alert_history:
        st.info("No events logged yet.")
        return

    df = pd.DataFrame(list(st.session_state.alert_history))
    display_cols = [
        "timestamp", "severity", "matched_source", "matched_keyword",
        "yara_severity", "visual_severity", "confidence", "latency",
        "fire_ratio", "smoke_ratio", "success", "description",
        "yolo_detected", "yolo_classes", "yolo_confidence",
    ]

    display_df = df[display_cols].copy()
    display_df["latency_ms"] = (display_df["latency"] * 1000).round(1)
    display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
    display_df = display_df.drop(columns=["latency"])

    st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download alerts log CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cybervision_forensic_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_charts() -> None:
    c1, c2 = st.columns(2)
    metrics_df = pd.DataFrame(list(st.session_state.metrics_history))

    if not metrics_df.empty:
        metrics_df = metrics_df.set_index("time")
        with c1:
            st.markdown("#### CPU/RAM Usage")
            st.line_chart(metrics_df[["cpu", "ram"]])
    else:
        c1.info("No resource metrics yet.")

    if st.session_state.alert_history:
        alert_df = pd.DataFrame(list(st.session_state.alert_history))
        severity_counts = (
            alert_df["severity"]
            .value_counts()
            .reindex(["NORMAL", "MEDIUM", "HIGH", "CRITICAL"])
            .fillna(0)
        )
        with c2:
            st.markdown("#### Severity Distribution")
            st.bar_chart(severity_counts)
    else:
        c2.info("No alert statistics yet.")


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1 class="hero-title">
                🔥 CyberVision: Adaptive Context Optimization at the Edge
            </h1>
            <span class="pill">Real-time Fire and Smoke Detection</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = SystemMonitor.get_metrics()
    cpu = metrics["cpu"]
    ram = metrics["ram"]

    if cpu >= CONFIG["CPU_THRESHOLD"] or ram >= CONFIG["RAM_THRESHOLD"]:
        st.warning(
            f"⚠️ **Critical System Load!** CPU: {cpu:.1f}% | RAM: {ram:.1f}%. System throttling active.",
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main() -> None:
    init_session_state()
    apply_theme()
    render_header()
    resource_metrics_placeholder = render_sidebar()

    status_cards_placeholder = st.empty()
    with status_cards_placeholder.container():
        render_status_cards()

    tab_live, tab_logs, tab_analytics = st.tabs([" Live Monitor", " Alert Logs", " Analytics"])

    with tab_live:
        render_live_monitor(resource_metrics_placeholder, status_cards_placeholder)

    with tab_logs:
        render_forensic_log()

    with tab_analytics:
        render_charts()


if __name__ == "__main__":
    main()