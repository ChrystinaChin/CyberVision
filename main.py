import base64
import io
import json
import os
import queue
import socket
import sqlite3
import threading
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

# Must be the absolute first Streamlit command executed
st.set_page_config(
    page_title="CyberVision",
    layout="wide",
    initial_sidebar_state="expanded",
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
    from google.cloud import firestore, storage
    GCP_AVAILABLE = True
except Exception:
    storage = None
    firestore = None
    GCP_AVAILABLE = False

try:
    # Lets the app pull video from the *visitor's* browser webcam over
    # WebRTC. Needed because a Streamlit Cloud container has no physical
    # camera of its own for cv2.VideoCapture to open (see
    # local_camera_available() below) — the server can't see a webcam that
    # only exists on the visitor's machine.
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    webrtc_streamer = None
    WebRtcMode = None
    VideoProcessorBase = object
    RTCConfiguration = None
    WEBRTC_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG: Dict[str, Any] = {
    "APP_NAME": "CyberVision",
    "APP_SUBTITLE": "Adaptive Context Optimization at the Edge",
    "VLM_MODEL": "moondream",
    "CAMERA_INDEX": 0,
    "CAMERA_SOURCES": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 3x3 Grid Sources
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 300,
    "JPEG_QUALITY": 70,
    "FRAME_DELAY_SEC": 0.22,
    "ANALYZE_EVERY_N_FRAMES": 12,
    "CPU_THRESHOLD": 95,
    "RAM_THRESHOLD": 90,
    "ALERT_HISTORY_MAX": 100,
    "METRICS_HISTORY_MAX": 150,
    "LATENCY_HISTORY_MAX": 100,
    "TOAST_DISPLAY_SECONDS": 6.0,
    "YARA_RULE_PATH": "hazard_rules.yar",
    "GCP_BUCKET": "your-gcp-bucket-name",
    "GCP_PROJECT_ID": "your-gcp-project-id",
    "DB_FILE": "cybervision_buffer.db",
    "PENDING_UPLOADS_DIR": "pending_gcp_uploads",
    "YOLO_MODEL_PATH": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "YOLO_dataset.pt",
    ),
    "YOLO_CONFIDENCE": 0.45,
    "YOLO_IMAGE_SIZE": 320,
}

SEVERITY_STYLE = {
    "NORMAL": {"score": 0, "icon": ":material/check_circle:", "color": "#30d158"},
    "MEDIUM": {"score": 1, "icon": ":material/warning:", "color": "#ffd60a"},
    "HIGH": {"score": 2, "icon": ":material/error:", "color": "#ff9500"},
    "CRITICAL": {"score": 3, "icon": ":material/dangerous:", "color": "#ff3b30"},
}

VLM_RESULT_QUEUE: queue.Queue = queue.Queue()


# =============================================================================
# OFFLINE SQLITE BUFFER & GCP FIRESTORE SYNC THREAD
# =============================================================================
def init_db() -> None:
    with sqlite3.connect(CONFIG["DB_FILE"]) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                severity TEXT,
                description TEXT,
                camera_id TEXT,
                synced INTEGER DEFAULT 0
            )
        """)
        conn.commit()


def save_event_locally(timestamp: float, severity: str, description: str, camera_id: str = "CAM_01") -> None:
    with sqlite3.connect(CONFIG["DB_FILE"]) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pending_events (timestamp, severity, description, camera_id, synced)
            VALUES (?, ?, ?, ?, 0)
        """, (timestamp, severity, description, camera_id))
        conn.commit()


def is_wifi_connected(host: str = "8.8.8.8", port: int = 53, timeout: int = 2) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


def sync_worker_loop() -> None:
    while True:
        if is_wifi_connected() and GCP_AVAILABLE and firestore is not None:
            try:
                db_client = firestore.Client(project=CONFIG["GCP_PROJECT_ID"])
                with sqlite3.connect(CONFIG["DB_FILE"]) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, timestamp, severity, description, camera_id FROM pending_events WHERE synced = 0")
                    unsynced_rows = cursor.fetchall()

                    for row in unsynced_rows:
                        event_id, ts, sev, desc, cam_id = row
                        doc_ref = db_client.collection("hazard_events").document(f"event_{event_id}")
                        doc_ref.set({
                            "timestamp": ts,
                            "severity": sev,
                            "description": desc,
                            "camera_id": cam_id,
                            "synced_at": firestore.SERVER_TIMESTAMP
                        })
                        cursor.execute("UPDATE pending_events SET synced = 1 WHERE id = ?", (event_id,))
                    conn.commit()
            except Exception:
                pass

            flush_pending_cloud_uploads()
        time.sleep(10)


def start_sync_thread() -> None:
    if "sync_thread_started" not in st.session_state:
        init_db()
        sync_thread = threading.Thread(target=sync_worker_loop, daemon=True)
        sync_thread.start()
        st.session_state.sync_thread_started = True


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
        "fps": 0.0,
        "last_frame_time": time.time(),
        "latest_detection": None,
        "yolo_severity": "NORMAL",
        "yolo_detector": None,
        "yolo_detection": None,
        "last_vlm_candidate_frame": None,
        "latest_vlm_text": "",
        "alarm_active": False,
        "vlm_in_progress": False,
        "last_triggered_alert_ts": 0.0,
        "active_hazard_toast": None,
        "active_page": "Dashboard",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data
def inject_custom_css() -> None:
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
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {{
            background-color: {bg} !important;
            background:
                radial-gradient(circle at top left, rgba(249,115,22,0.18), transparent 32%),
                linear-gradient(135deg, {bg} 0%, #ffffff 45%, #fff1e6 100%) !important;
            color: {text} !important;
        }}

        .main .block-container {{
            padding-top: 0.4rem;
            padding-bottom: 1rem;
            max-width: 1600px;
        }}

        div[data-testid="stVerticalBlock"] {{
            gap: 0.55rem !important;
        }}

        [data-testid="stAlert"] {{
            padding: 0.5rem 0.9rem !important;
            margin-bottom: 0 !important;
        }}

        div[data-baseweb="select"] {{
            border-radius: 14px !important;
        }}

        div[data-baseweb="select"] > div {{
            background: rgba(255,255,255,0.95) !important;
            border: 1px solid {border} !important;
            color: {accent_dark} !important;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #ffffff 0%, {panel_warm} 100%) !important;
            border-right: 1px solid {border};
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: {accent_dark} !important;
            font-weight: 900 !important;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: {text};
        }}

        .muted {{
            color: {muted};
        }}

        .hero {{
            position: relative;
            padding: 0.45rem 1.1rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,237,213,0.96)),
                radial-gradient(circle at top right, rgba(249,115,22,0.28), transparent 35%);
            border: 1px solid {border};
            border-radius: 20px;
            margin-bottom: 0.4rem;
            box-shadow: 0 10px 26px rgba(249,115,22,0.13);
            overflow: hidden;
        }}

        .hero-title {{
            font-size: 1.55rem;
            font-weight: 900;
            margin: 0.15rem 0 0 0;
            color: {accent_dark};
            letter-spacing: -0.03em;
        }}

        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            background: rgba(249,115,22,0.12);
            border: 1px solid rgba(249,115,22,0.35);
            color: {accent_dark};
            font-weight: 900;
            font-size: 0.7rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.94);
            border: 1px solid {border};
            border-radius: 12px;
            padding: 0.3rem 0.5rem;
            height: 84px;
            min-height: 84px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 6px 14px rgba(249,115,22,0.08);
        }}

        div[data-testid="stMetricValue"] {{
            color: {accent_dark};
            font-weight: 900;
            font-size: 1.25rem !important;
        }}

        /* Delta badges (the little rounded "↑ of 1" / "↑ ⚠" pill under a
           metric) default to Streamlit's green/red — flatten that to a
           neutral, slightly transparent grey. The CPU/RAM "_high" rules
           further down are more specific and still turn red on alert. */
        div[data-testid="stMetricDelta"] {{
            background: rgba(107, 114, 128, 0.12) !important;
            border-radius: 999px !important;
            padding: 0.05rem 0.5rem !important;
            width: fit-content;
        }}

        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] svg,
        div[data-testid="stMetricDelta"] span {{
            color: {muted} !important;
            fill: {muted} !important;
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
        }}

        /* ---------------------------------------------------------------
           HAZARD TOAST — a fully custom fixed-position card (rendered by
           render_custom_hazard_toast via plain st.markdown), NOT st.toast.

           st.toast has its own internal height clamp that silently cuts
           off longer messages (the action line / timestamp vanished with
           no scrollbar or ellipsis) and no stylesheet override — however
           specific, however many !important — could reach it, because
           that clamp lives in Streamlit's own component state rather
           than in plain overridable CSS. Owning the markup outright
           sidesteps that entirely: nothing here can silently truncate.
        --------------------------------------------------------------- */
        .hazard-toast {{
            position: fixed;
            top: 1.1rem;
            right: 1.1rem;
            z-index: 9999;
            background: #ffffff;
            border: 1px solid {border};
            border-left: 7px solid {accent};
            border-radius: 18px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.22);
            padding: 1.1rem 1.3rem;
            width: auto;
            min-width: 340px;
            max-width: 420px;
        }}

        .hazard-toast .toast-severity {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-weight: 900;
            font-size: 1.15rem;
            color: #1f2937;
        }}

        .hazard-toast .toast-title {{
            font-weight: 700;
            font-size: 0.95rem;
            color: #1f2937;
            margin-top: 0.1rem;
        }}

        .hazard-toast .toast-line {{
            font-weight: 600;
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: 0.4rem;
            white-space: normal;
            word-break: break-word;
            overflow-wrap: break-word;
        }}

        .hazard-toast .toast-detection {{
            color: #2563eb;
        }}

        .hazard-toast .toast-action {{
            color: #1f2937;
            font-weight: 800;
        }}

        .hazard-toast .toast-meta {{
            color: {muted};
            font-size: 0.8rem;
            margin-top: 0.45rem;
        }}

        /* Tighten tab bar spacing so the video grid sits closer to it */
        .stTabs {{
            margin-top: -0.3rem !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 1.2rem !important;
        }}

        .stTabs [data-baseweb="tab-panel"] {{
            padding-top: 0.4rem !important;
        }}

        /* -----------------------------------------------------------
           SIDEBAR WORKSPACE NAVIGATION
        ----------------------------------------------------------- */
        /* Sidebar layout: turn every element between the sidebar root and
           the account block into a flex column that fills the sidebar's
           height. The account block then uses margin-top:auto to sit at the
           very bottom. (":has" is used so this works no matter how many
           wrapper divs a given Streamlit version puts around the block.) */
        [data-testid="stSidebarContent"] {{
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }}

        [data-testid="stSidebar"] div:has(.st-key-sidebar_bottom_block):not(.st-key-sidebar_bottom_block) {{
            display: flex !important;
            flex-direction: column !important;
            flex: 1 1 auto !important;
            min-height: 0 !important;
        }}

        [data-testid="stSidebarUserContent"] {{
            padding-bottom: 0.6rem !important;
        }}

        /* Google account card + Log out button: pinned to the bottom edge.
           "sticky" keeps it visible if the nav list ever grows tall enough
           for the sidebar to scroll. */
        .st-key-sidebar_bottom_block {{
            margin-top: auto !important;
            flex: 0 0 auto !important;
            position: sticky !important;
            bottom: 0 !important;
            background: {panel_warm} !important;
            padding-top: 0.3rem;
            z-index: 5;
        }}

        .brand-block {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.2rem 0 1.1rem 0;
        }}

        .brand-icon {{
            font-size: 1.7rem;
            line-height: 1;
        }}

        .brand-name {{
            font-weight: 900;
            font-size: 1.15rem;
            letter-spacing: -0.02em;
            color: {accent_dark};
            line-height: 1.1;
        }}

        .brand-subtitle {{
            font-size: 0.65rem;
            color: {muted};
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .nav-section-label {{
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {muted};
            margin: 0.6rem 0 0.5rem 0.1rem;
        }}

        [data-testid="stSidebar"] .stButton > button {{
            justify-content: flex-start !important;
            text-align: left !important;
            background: transparent !important;
            border: 1px solid transparent !important;
            box-shadow: none !important;
            font-weight: 700 !important;
            color: {text} !important;
            padding: 0.5rem 0.7rem !important;
        }}

        [data-testid="stSidebar"] .stButton > button > div {{
            display: flex !important;
            width: 100% !important;
            justify-content: space-between !important;
            align-items: center !important;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background: rgba(249,115,22,0.10) !important;
            color: {accent_dark} !important;
            transform: none !important;
        }}

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {accent}, {accent_dark}) !important;
            color: #ffffff !important;
            border: 1px solid {accent_dark} !important;
            box-shadow: 0 8px 18px rgba(249,115,22,0.30) !important;
        }}

        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
            background: linear-gradient(135deg, {accent_dark}, #9a3412) !important;
            color: #ffffff !important;
        }}

        .sidebar-footer-divider {{
            border-top: 1px solid {border};
            margin: 1.2rem 0 0.9rem 0;
        }}

        /* -----------------------------------------------------------
           RESOURCE METRICS — CPU / RAM turn red once they cross the
           configured warning threshold.
        ----------------------------------------------------------- */
        .st-key-cpu_metric_high [data-testid="stMetricValue"],
        .st-key-ram_metric_high [data-testid="stMetricValue"],
        .st-key-cpu_metric_high [data-testid="stMetricDelta"],
        .st-key-ram_metric_high [data-testid="stMetricDelta"] {{
            color: #ff3b30 !important;
        }}

        .st-key-cpu_metric_high [data-testid="stMetricDelta"] svg,
        .st-key-ram_metric_high [data-testid="stMetricDelta"] svg {{
            fill: #ff3b30 !important;
        }}

        /* -----------------------------------------------------------
           SEVERITY METRIC — colour follows the YOLO severity:
           NORMAL = green, MEDIUM = yellow, HIGH / CRITICAL = red.
        ----------------------------------------------------------- */
        .st-key-severity_metric_normal [data-testid="stMetricValue"],
        .st-key-severity_metric_normal [data-testid="stMetricValue"] * {{
            color: #16a34a !important;
        }}
        .st-key-severity_metric_medium [data-testid="stMetricValue"],
        .st-key-severity_metric_medium [data-testid="stMetricValue"] * {{
            color: #eab308 !important;
        }}
        .st-key-severity_metric_high [data-testid="stMetricValue"],
        .st-key-severity_metric_high [data-testid="stMetricValue"] *,
        .st-key-severity_metric_critical [data-testid="stMetricValue"],
        .st-key-severity_metric_critical [data-testid="stMetricValue"] * {{
            color: #dc2626 !important;
        }}

        .status-footer {{
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            margin-top: 1rem;
        }}

        .status-row {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.82rem;
            font-weight: 600;
            color: {muted};
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
            box-shadow: 0 0 0 3px rgba(0,0,0,0.03);
        }}

        .profile-card {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.55rem 0.6rem;
            border-radius: 14px;
            background: rgba(249,115,22,0.08);
            border: 1px solid {border};
            margin-bottom: 0.5rem;
        }}

        .profile-avatar {{
            width: 34px;
            height: 34px;
            border-radius: 50%;
            object-fit: cover;
            flex-shrink: 0;
        }}

        .profile-avatar-fallback {{
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, {accent}, {accent_dark});
            color: #ffffff;
            font-weight: 800;
            font-size: 0.9rem;
        }}

        .profile-text {{
            min-width: 0;
        }}

        .profile-name {{
            font-weight: 800;
            font-size: 0.82rem;
            color: {text};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .profile-email {{
            font-size: 0.68rem;
            color: {muted};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        /* -----------------------------------------------------------
           DASHBOARD HEADER + LIVE STATUS PILL
        ----------------------------------------------------------- */
        .dash-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.35rem 0.1rem 0.9rem 0.1rem;
            flex-wrap: wrap;
            gap: 0.6rem;
        }}

        .dash-header-left {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }}

        .dash-header-icon {{
            font-size: 1.6rem;
        }}

        .dash-header-title {{
            font-weight: 900;
            font-size: 1.35rem;
            letter-spacing: -0.02em;
            color: {text};
            line-height: 1.15;
        }}

        .dash-header-subtitle {{
            font-size: 0.85rem;
            color: {muted};
            font-weight: 500;
        }}

        .status-pill-live {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            border: 1px solid;
            background: rgba(255,255,255,0.9);
            font-weight: 800;
            font-size: 0.72rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .status-pill-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
        }}

        /* -----------------------------------------------------------
           CARD PANELS (video panel, stats side panel)
        ----------------------------------------------------------- */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(255,255,255,0.94) !important;
            border: 1px solid {border} !important;
            border-radius: 18px !important;
            box-shadow: 0 10px 26px rgba(249,115,22,0.10) !important;
        }}

        .stats-panel-title {{
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {muted};
            margin-bottom: 0.4rem;
        }}

        .video-status-bar {{
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.5rem;
            padding: 0.5rem 0.2rem 0.1rem 0.2rem;
            font-size: 0.8rem;
        }}

        .video-status-bar b {{
            color: {accent_dark};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# AUDIO & NOTIFICATION DISPATCHER
# =============================================================================
def build_beep_wav_base64(duration: float = 0.35, freq: int = 1050, sample_rate: int = 44100) -> str:
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


def play_alert_sound() -> None:
    """Embeds robust auto-playing HTML5 audio element into Streamlit canvas."""
    if not st.session_state.sound_enabled:
        return

    audio_b64 = build_beep_wav_base64(duration=0.45, freq=1100)
    audio_html = f"""
        <audio autoplay style="display:none;">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


def build_siren_wav_base64(
    duration: float = 1.6,
    low_freq: int = 900,
    high_freq: int = 1500,
    sample_rate: int = 44100,
) -> str:
    """Alternating two-tone siren clip used for the continuous CRITICAL alarm."""
    segment_len = max(1, int(sample_rate * duration / 4))
    t_seg = np.linspace(0, duration / 4, segment_len, False)

    def tone(freq: int) -> np.ndarray:
        return np.sin(freq * t_seg * 2 * np.pi)

    wave_data = np.concatenate([tone(high_freq), tone(low_freq), tone(high_freq), tone(low_freq)])
    audio = (wave_data * 32767 * 0.9).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def play_critical_alarm(is_critical: bool) -> None:
    """
    Keeps a siren ringing continuously for as long as severity stays CRITICAL.

    Because this is called on every ~0.1s fragment tick with byte-identical
    markup while active, Streamlit doesn't need to be told to "loop" anything
    itself — the <audio loop> element just keeps playing across reruns. As
    soon as this stops being called with is_critical=True (severity drops or
    the camera stops), the element is no longer emitted and the sound stops.
    """
    st.session_state.alarm_active = bool(is_critical) and st.session_state.sound_enabled

    if not st.session_state.alarm_active:
        return

    siren_b64 = build_siren_wav_base64()
    st.markdown(
        f"""
        <audio autoplay loop style="display:none;">
            <source src="data:audio/wav;base64,{siren_b64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def dispatch_hazard_alerts(severity_str: str, hazard_title: str) -> None:
    """Show a single fire/smoke alert in the requested notification format."""
    severity_norm = str(severity_str).strip().upper()

    if severity_norm not in ["MEDIUM", "HIGH", "CRITICAL"]:
        return

    now = time.time()

    # Cooldown period of 3 seconds between continuous notification bursts.
    if now - st.session_state.get("last_triggered_alert_ts", 0) <= 3.0:
        return

    st.session_state.last_triggered_alert_ts = now

    # IMPORTANT: never display "Fire & Smoke Detected".
    # The detector resolves the notification to exactly ONE hazard type.
    hazard_norm = str(hazard_title).strip().upper()

    if "FIRE" in hazard_norm:
        alert_title = "FIRE DETECTED"
        hazard_word = "fire"
    else:
        alert_title = "SMOKE DETECTED"
        hazard_word = "smoke"

    detection_line = f"The system detected clear signs of {hazard_word}."
    action_line = "Please evacuate from the area immediately."
    checked_time = datetime.now().strftime("%H:%M:%S")

    # Rendered by render_custom_hazard_toast() as a plain fixed-position
    # HTML card, NOT st.toast — st.toast has its own internal height
    # clamp that silently cut off the action line / timestamp on longer
    # messages, and no CSS override (however specific) could reach it
    # since that clamp lives in Streamlit's own component state rather
    # than in plain stylesheet-overridable CSS. Storing the fields here
    # and rendering them ourselves sidesteps that entirely.
    st.session_state.active_hazard_toast = {
        "severity": severity_norm,
        "alert_title": alert_title,
        "detection_line": detection_line,
        "action_line": action_line,
        "checked_time": checked_time,
        "triggered_at": now,
    }

    # CRITICAL gets the continuous siren (played elsewhere, every pipeline
    # tick, independent of this 3s toast cooldown) — no need to also fire
    # the short one-off beep here, which would just overlap it.
    if severity_norm != "CRITICAL":
        play_alert_sound()


def render_custom_hazard_toast() -> None:
    """Renders the most recent fire/smoke alert (see dispatch_hazard_alerts)
    as a fixed top-right card for CONFIG["TOAST_DISPLAY_SECONDS"], then
    stops rendering it — a self-expiring notification with none of
    st.toast's baked-in truncation."""
    toast_data = st.session_state.get("active_hazard_toast")
    if not toast_data:
        return

    if time.time() - toast_data["triggered_at"] > CONFIG["TOAST_DISPLAY_SECONDS"]:
        return

    st.markdown(
        f"""
        <div class="hazard-toast">
            <div class="toast-title">{toast_data['alert_title']}</div>
            <div class="toast-line toast-detection">{toast_data['detection_line']}</div>
            <div class="toast-line toast-action">{toast_data['action_line']}</div>
            <div class="toast-meta">Last checked: {toast_data['checked_time']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )




# =============================================================================
# SYSTEM MONITORING
# =============================================================================
class SystemMonitor:
    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        try:
            cpu = float(psutil.cpu_percent(interval=0.03))
            ram = float(psutil.virtual_memory().percent)

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
        metrics = SystemMonitor.get_metrics()
        worst = max(metrics["cpu"], metrics["ram"])

        if worst >= 85:
            return 128
        if worst >= 70:
            return 256
        if worst >= 50:
            return 512

        return 1024


# =============================================================================
# MULTI-CAMERA HANDLING & 3x3 MATRIX
# =============================================================================
# A public STUN server is required for the visitor's browser and the
# Streamlit Cloud container to negotiate a WebRTC connection across the
# open internet (they're on different networks/behind NAT) — without an
# ICE server the connection just never completes.
RTC_CONFIGURATION = (
    RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    if WEBRTC_AVAILABLE else None
)


@st.cache_resource(show_spinner=False)
def local_camera_available() -> bool:
    """
    Probes once per server process whether cv2 can actually open a physical
    camera here. True on a laptop/desktop run; always False on Streamlit
    Cloud (and any other headless container), which is what produces the
    "Camera unavailable" placeholder — there's no webcam attached to that
    machine for cv2.VideoCapture to find. Cached so this cheap-but-not-free
    probe runs once, not on every 0.1s fragment tick.
    """
    try:
        cap = cv2.VideoCapture(CONFIG["CAMERA_SOURCES"][0])
        ok = cap.isOpened()
        cap.release()
        return ok
    except Exception:
        return False


class BrowserCameraProcessor(VideoProcessorBase):
    """
    Receives frames pushed from the visitor's own browser webcam over
    WebRTC. recv() runs on streamlit-webrtc's background thread, so the
    frame is handed off through a lock rather than touched directly from
    Streamlit's render thread.
    """
    def __init__(self) -> None:
        self.frame_bgr: Optional[np.ndarray] = None
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.frame_bgr = img
        return frame


def render_browser_camera_widget() -> Optional["BrowserCameraProcessor"]:
    """
    Mounts the WebRTC component once (outside the fast 0.1s fragment, so the
    connection isn't torn down and rebuilt on every tick) and returns its
    processor so the fragment can just read the latest frame off it.
    """
    if not WEBRTC_AVAILABLE:
        st.warning(
            "This server has no physical camera, and the `streamlit-webrtc` "
            "package isn't installed, so there's no way to pull video from "
            "your browser either. Add `streamlit-webrtc` and `av` to "
            "requirements.txt to enable browser-camera streaming."
        )
        return None

    st.caption(":material/videocam: No physical camera on this server — streaming from your browser's camera instead. Allow camera access when prompted.")
    ctx = webrtc_streamer(
        key="cybervision_browser_camera",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=BrowserCameraProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    return ctx.video_processor


@st.cache_resource(show_spinner=False)
def get_camera_caps() -> Dict[int, cv2.VideoCapture]:
    caps = {}
    for idx, src in enumerate(CONFIG["CAMERA_SOURCES"]):
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            caps[idx] = cap
        else:
            cap.release()
    return caps


def release_camera() -> None:
    try:
        caps = get_camera_caps()
        for cap in caps.values():
            if cap and cap.isOpened():
                cap.release()
        get_camera_caps.clear()
    except Exception:
        pass


def create_blank_tile(width: int = 320, height: int = 240, label: str = "NO CAMERA DETECTED") -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.45, 1)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(frame, label, (text_x, text_y), font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (40, 40, 40), 1)
    return frame


def construct_3x3_grid(active_frames: Dict[int, np.ndarray], tile_w: int = 320, tile_h: int = 240) -> np.ndarray:
    tiles = []
    for idx in range(9):
        cam_key = f"CAM_{idx+1:02d}"
        if idx in active_frames and active_frames[idx] is not None:
            tile = cv2.resize(active_frames[idx], (tile_w, tile_h))
            cv2.putText(tile, f"{cam_key} [LIVE]", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (0, 255, 0), 1)
        else:
            tile = create_blank_tile(tile_w, tile_h, label=f"{cam_key}: NO CAMERA DETECTED")
        tiles.append(tile)

    row1 = np.hstack([tiles[0], tiles[1], tiles[2]])
    row2 = np.hstack([tiles[3], tiles[4], tiles[5]])
    row3 = np.hstack([tiles[6], tiles[7], tiles[8]])

    return np.vstack([row1, row2, row3])


class VideoCaptureManager:
    @staticmethod
    def capture_active_frames(browser_frame: Optional[np.ndarray] = None) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
        # A frame handed in from the visitor's browser (via WebRTC) takes
        # priority and skips cv2 entirely — used on Streamlit Cloud, where
        # cv2.VideoCapture has no physical camera to open.
        if browser_frame is not None:
            primary_frame = cv2.resize(browser_frame, (CONFIG["FRAME_WIDTH"], CONFIG["FRAME_HEIGHT"]))
            return {0: browser_frame}, primary_frame

        caps = get_camera_caps()
        active_frames = {}
        primary_frame = None

        if not caps:
            st.session_state.last_error = "No camera streams open."
            return {}, None

        for idx, cap in list(caps.items()):
            if cap.isOpened():
                for _ in range(2):
                    cap.grab()
                ret, frame = cap.read()
                if ret and frame is not None:
                    active_frames[idx] = frame
                    if primary_frame is None:
                        primary_frame = cv2.resize(frame, (CONFIG["FRAME_WIDTH"], CONFIG["FRAME_HEIGHT"]))

        return active_frames, primary_frame

    @staticmethod
    def placeholder_frame(message: str, subtext: str = "") -> np.ndarray:
        frame = np.zeros(
            (CONFIG["FRAME_HEIGHT"], CONFIG["FRAME_WIDTH"], 3),
            dtype=np.uint8
        )
        frame[:] = (22, 28, 55)

        cv2.putText(
            frame,
            message,
            (55, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 170),
            2,
        )

        if subtext:
            cv2.putText(
                frame,
                subtext,
                (55, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (220, 220, 220),
                1,
            )

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
        }

        if frame is None or self.model is None:
            return empty

        try:
            results = self.model(
                frame,
                imgsz=CONFIG["YOLO_IMAGE_SIZE"],
                conf=CONFIG["YOLO_CONFIDENCE"],
                verbose=False,
                device="cpu",
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
            }

        except Exception as exc:
            st.session_state.last_error = f"YOLO detection issue: {exc}"
            return empty

    @staticmethod
    def hazard_severity(yolo_result: Dict[str, Any]) -> str:
        if not yolo_result.get("detected"):
            return "NORMAL"

        highest = "NORMAL"
        for detection in yolo_result.get("detections", []):
            class_name = detection["class"].lower()
            confidence = detection["confidence"]

            if "fire" in class_name:
                candidate = "CRITICAL" if confidence >= 0.50 else "HIGH"
            elif "smoke" in class_name:
                candidate = "HIGH" if confidence >= 0.30 else "MEDIUM"
            else:
                continue

            if SEVERITY_STYLE[candidate]["score"] > SEVERITY_STYLE[highest]["score"]:
                highest = candidate

        return highest


# =============================================================================
# VISUAL FALLBACK DETECTION
# =============================================================================
class VisualFireSmokeDetector:
    @staticmethod
    def detect(frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
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

        red_or_orange_hue = ((h <= 18) | (h >= 172))
        bright_saturated = (s >= 160) & (v >= 200)
        rgb_dominance = (r > 190) & (r > g * 1.25) & (r > b * 1.85) & (g > 60)
        non_skin = skin_mask == 0

        fire_pixels = red_or_orange_hue & bright_saturated & rgb_dominance & non_skin
        fire_mask = fire_pixels.astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_DILATE, kernel)

        low_saturation = s < 35
        mid_brightness = (v > 110) & (v < 200)
        gray_dominance = (
            (np.abs(r.astype(int) - g.astype(int)) < 18)
            & (np.abs(g.astype(int) - b.astype(int)) < 18)
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
        keyword = ""

        if fire_ratio >= 0.050 and largest_fire_ratio >= 0.040:
            severity = "CRITICAL"
            keyword = "visual_fire_critical"
        elif fire_ratio >= 0.025 and largest_fire_ratio >= 0.020:
            severity = "HIGH"
            keyword = "visual_fire_high"
        elif fire_ratio >= 0.010:
            severity = "MEDIUM"
            keyword = "visual_fire_medium"
        elif smoke_ratio >= 0.35:
            severity = "CRITICAL"
            keyword = "visual_smoke_critical"
        elif smoke_ratio >= 0.20:
            severity = "HIGH"
            keyword = "visual_smoke_high"
        elif smoke_ratio >= 0.10:
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
# OLLAMA VLM INFERENCE (THREAD-SAFE ASYNC NON-BLOCKING)
# =============================================================================
class VLMInference:
    @staticmethod
    def _async_worker(frame: np.ndarray, token_budget: int) -> None:
        try:
            image_b64 = frame_to_base64_jpeg(frame)
            response = ollama.chat(
                model=CONFIG["VLM_MODEL"],
                messages=[
                    {
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
                    }
                ],
                options={
                    "num_ctx": token_budget,
                    "num_predict": 80,
                    "temperature": 0.0,
                },
            )

            text = response.get("message", {}).get("content", "").strip()
            if text:
                VLM_RESULT_QUEUE.put({"success": True, "text": text, "error": ""})
            else:
                VLM_RESULT_QUEUE.put({"success": False, "text": "", "error": "Ollama returned empty response."})
        except Exception as exc:
            VLM_RESULT_QUEUE.put({"success": False, "text": "", "error": f"Ollama backend issue: {exc}"})

    @staticmethod
    def trigger_async_inference(frame: np.ndarray) -> None:
        if not OLLAMA_AVAILABLE:
            st.session_state.last_error = "Ollama Python package is not installed."
            return

        if st.session_state.get("vlm_in_progress", False):
            return

        st.session_state.vlm_in_progress = True
        token_budget = ResourceGovernor.recommended_token_budget()
        thread = threading.Thread(
            target=VLMInference._async_worker,
            args=(frame.copy(), token_budget),
            daemon=True
        )
        thread.start()

    @staticmethod
    def drain_worker_queue() -> None:
        while not VLM_RESULT_QUEUE.empty():
            try:
                res = VLM_RESULT_QUEUE.get_nowait()
                st.session_state.vlm_in_progress = False
                if res["success"]:
                    st.session_state.latest_vlm_text = res["text"]
                    st.session_state.inference_count += 1
                    st.session_state.last_error = ""
                else:
                    st.session_state.last_error = res["error"]
            except queue.Empty:
                break


# =============================================================================
# YARA VERIFICATION
# =============================================================================
@st.cache_resource(show_spinner=False)
def compile_yara_rules():
    if not YARA_AVAILABLE:
        return None

    if not os.path.exists(CONFIG["YARA_RULE_PATH"]):
        st.session_state.last_error = "hazard_rules.yar not found."
        return None

    return yara.compile(filepath=CONFIG["YARA_RULE_PATH"])


class YARAVerifier:
    @staticmethod
    def verify(vlm_text: str, visual_result: Dict[str, Any]) -> Dict[str, Any]:
        yara_severity = "NORMAL"
        matched_rule = "no_yara_match"
        matched_source = "YARA_TEXT"

        if YARA_AVAILABLE and vlm_text:
            try:
                rules = compile_yara_rules()
                matches = rules.match(data=vlm_text.encode("utf-8", errors="ignore")) if rules else []

                if matches:
                    highest_score = -1
                    for match in matches:
                        severity = match.meta.get("severity", "NORMAL")
                        score = SEVERITY_STYLE.get(severity, SEVERITY_STYLE["NORMAL"])["score"]

                        if score > highest_score:
                            highest_score = score
                            yara_severity = severity
                            matched_rule = match.rule

            except Exception:
                yara_severity = "NORMAL"
                matched_rule = "yara_verification_unavailable"
        else:
            matched_rule = "yara_not_installed" if not YARA_AVAILABLE else "no_vlm_text"

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
        }


# =============================================================================
# GOOGLE CLOUD FORENSIC BACKUP & LOCAL BUFFER
# =============================================================================
def _queue_image_for_later_upload(frame: np.ndarray, alert: Dict[str, Any]) -> None:
    """
    Buffers a frame + its alert metadata to local disk so it can be pushed
    to the GCS bucket once network connectivity returns. This mirrors the
    SQLite buffer used for the text alert log, but for the image/JSON pair
    that upload_to_google_cloud_async otherwise sends immediately.
    """
    try:
        pending_dir = CONFIG["PENDING_UPLOADS_DIR"]
        os.makedirs(pending_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        severity = alert.get("severity", "UNKNOWN")
        base_name = f"{severity}_{stamp}"

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return

        with open(os.path.join(pending_dir, f"{base_name}.jpg"), "wb") as f:
            f.write(buffer.tobytes())

        with open(os.path.join(pending_dir, f"{base_name}.json"), "w") as f:
            json.dump(alert, f, default=str)
    except Exception:
        pass


def flush_pending_cloud_uploads() -> None:
    """
    Drains the local image/alert buffer to the GCS bucket. Called from
    sync_worker_loop only when is_wifi_connected() is already True, so this
    never attempts network I/O while offline. Files that fail to upload are
    left in place and retried on the next pass (every ~10s).
    """
    if not GCP_AVAILABLE or storage is None:
        return
    if CONFIG["GCP_BUCKET"] == "your-gcp-bucket-name":
        return

    pending_dir = CONFIG["PENDING_UPLOADS_DIR"]
    if not os.path.isdir(pending_dir):
        return

    try:
        client = storage.Client()
        bucket = client.bucket(CONFIG["GCP_BUCKET"])
    except Exception:
        return

    for filename in sorted(os.listdir(pending_dir)):
        if not filename.endswith(".jpg"):
            continue

        base_name = filename[:-4]
        image_path = os.path.join(pending_dir, filename)
        meta_path = os.path.join(pending_dir, f"{base_name}.json")

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            bucket.blob(f"cybervision_alerts/{base_name}.jpg").upload_from_string(
                image_bytes, content_type="image/jpeg"
            )

            if os.path.isfile(meta_path):
                with open(meta_path, "r") as f:
                    meta_text = f.read()
                bucket.blob(f"cybervision_alerts/{base_name}.json").upload_from_string(
                    meta_text, content_type="application/json"
                )
                os.remove(meta_path)

            os.remove(image_path)
        except Exception:
            # Leave both files in place — retried automatically next pass.
            continue


def upload_to_google_cloud_async(frame: np.ndarray, alert: Dict[str, Any]) -> None:
    def _worker():
        # Always logged locally first, regardless of network or cloud_sync
        # settings — this is the source of truth until it's synced.
        save_event_locally(
            time.time(),
            alert.get("severity", "UNKNOWN"),
            alert.get("description", ""),
            camera_id="CAM_01"
        )

        if not st.session_state.cloud_sync_enabled or not GCP_AVAILABLE or storage is None:
            return

        if CONFIG["GCP_BUCKET"] == "your-gcp-bucket-name":
            return

        # Offline-first: only attempt the live upload when there's actually
        # network. If offline (or the upload fails mid-flight), buffer the
        # frame + alert locally so flush_pending_cloud_uploads() can retry
        # it once connectivity returns — nothing gets silently dropped.
        if not is_wifi_connected():
            _queue_image_for_later_upload(frame, alert)
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
            _queue_image_for_later_upload(frame, alert)

    threading.Thread(target=_worker, daemon=True).start()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def determine_hazard_title(latest: Dict[str, Any]) -> str:
    """Resolve the alert to exactly one hazard type: FIRE or SMOKE."""
    if not latest or latest.get("severity") == "NORMAL":
        return "Environment is Safe"

    classes = [c.lower() for c in latest.get("yolo_classes", "").split(",") if c.strip()]
    description = latest.get("description", "").lower()
    keyword = str(latest.get("matched_keyword", "")).lower()

    has_fire = (
        any("fire" in item for item in classes)
        or "fire" in description
        or "fire" in keyword
        or latest.get("fire_ratio", 0) > 0.010
    )

    has_smoke = (
        any("smoke" in item for item in classes)
        or "smoke" in description
        or "smoke" in keyword
        or latest.get("smoke_ratio", 0) > 0.10
    )

    # Fire takes priority if both signals are present. This guarantees that
    # the user-facing notification NEVER says "Fire & Smoke Detected".
    if has_fire:
        return "Fire Detected"
    if has_smoke:
        return "Smoke Detected"

    # If the severity is hazardous but the source does not identify the type,
    # use the VLM description when possible; otherwise default to smoke.
    if "fire" in description or "fire" in keyword:
        return "Fire Detected"

    return "Smoke Detected"


# =============================================================================
# DETECTION PIPELINE
# =============================================================================
def run_detection_pipeline(frame: np.ndarray) -> Dict[str, Any]:
    if frame is None or frame.size == 0:
        default_alert = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "description": "NORMAL: Camera stream starting...",
            "success": True,
            "latency": 0.0,
            "verified": False,
            "yolo_detected": False,
            "yolo_classes": "",
            "yolo_confidence": 0.0,
            "severity": "NORMAL",
            "is_hazard": False,
            "confidence": 0.0,
            "matched_keyword": "none",
            "matched_source": "SYSTEM",
            "yara_severity": "NORMAL",
            "visual_severity": "NORMAL",
            "fire_ratio": 0.0,
            "smoke_ratio": 0.0,
            "yara_available": YARA_AVAILABLE,
        }
        play_critical_alarm(False)
        st.session_state.yolo_severity = "NORMAL"
        return {
            "annotated_frame": frame,
            "yolo": {"detected": False, "detections": []},
            "alert": default_alert,
        }

    ResourceGovernor.check_resource_pressure()
    start_time = time.time()

    VLMInference.drain_worker_queue()

    visual_result = VisualFireSmokeDetector.detect(frame)

    yolo_detector = st.session_state.get("yolo_detector")
    if yolo_detector is None:
        yolo_detector = YOLOFireSmokeDetector()
        st.session_state.yolo_detector = yolo_detector

    yolo_result = yolo_detector.detect(frame)
    st.session_state.yolo_detection = yolo_result

    yolo_severity = yolo_detector.hazard_severity(yolo_result)
    # Exposed to the UI so the Severity card always mirrors the live YOLO result.
    st.session_state.yolo_severity = yolo_severity

    if SEVERITY_STYLE[yolo_severity]["score"] > SEVERITY_STYLE[visual_result["visual_severity"]]["score"]:
        visual_result["visual_severity"] = yolo_severity
        if yolo_result.get("detections"):
            first = yolo_result["detections"][0]
            visual_result["visual_keyword"] = f"yolo_{first['class']}_{first['confidence']:.2f}"

    candidate_hazard = (
        yolo_result.get("detected", False)
        or visual_result.get("visual_severity", "NORMAL") != "NORMAL"
    )

    frame_number = st.session_state.frames_processed
    analyze_this_frame = (
        candidate_hazard
        and frame_number % max(1, CONFIG["ANALYZE_EVERY_N_FRAMES"]) == 0
    )

    if analyze_this_frame:
        VLMInference.trigger_async_inference(frame)

    vlm_text = st.session_state.get("latest_vlm_text", "")
    if not vlm_text:
        vlm_text = "NORMAL: Monitoring active."

    verdict = YARAVerifier.verify(vlm_text, visual_result)

    latency = time.time() - start_time
    st.session_state.latency_history.append(latency)

    public_description = vlm_text
    if yolo_result.get("detected"):
        detection_text = ", ".join(f"{d['class']} {d['confidence']:.0%}" for d in yolo_result["detections"])
        public_description = f"YOLO detected: {detection_text}. VLM: {vlm_text}"

    alert = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "description": public_description,
        "success": True,
        "latency": latency,
        "verified": verdict["is_hazard"],
        "yolo_detected": yolo_result.get("detected", False),
        "yolo_classes": ", ".join(yolo_result.get("classes", [])),
        "yolo_confidence": yolo_result.get("highest_confidence", 0.0),
        **verdict,
    }

    st.session_state.alert_history.append(alert)
    st.session_state.latest_detection = alert

    # Trigger audio & hazard-card notifications on threat detection
    hazard_title = determine_hazard_title(alert)
    dispatch_hazard_alerts(alert["severity"], hazard_title)

    # Continuous siren: checked every fragment tick (unlike the toast/beep
    # above, this is NOT gated by the 3s cooldown), so it starts the moment
    # severity hits CRITICAL and stops the instant it drops below CRITICAL.
    play_critical_alarm(alert["severity"] == "CRITICAL")

    if verdict["is_hazard"]:
        upload_to_google_cloud_async(frame, alert)

    return {
        "annotated_frame": yolo_result.get("annotated_frame", frame),
        "yolo": yolo_result,
        "alert": alert,
    }


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar_profile() -> None:
    """
    Shows the signed-in Google account (avatar/name/email) plus a Log out
    control. Backed by Streamlit's native st.user / st.logout(), which
    authenticates directly against Google's OAuth endpoints — no separate
    Cloud Function is needed for the sign-in flow itself.
    """
    st.markdown("<div class='sidebar-footer-divider'></div>", unsafe_allow_html=True)

    user_name = getattr(st.user, "name", None) or "Signed in"
    user_email = getattr(st.user, "email", "") or ""
    user_picture = getattr(st.user, "picture", None)

    if user_picture:
        avatar_html = f'<img class="profile-avatar" src="{user_picture}" alt="avatar" />'
    else:
        initial = (user_name or "?")[:1].upper()
        avatar_html = f'<div class="profile-avatar profile-avatar-fallback">{initial}</div>'

    st.markdown(
        f"""
        <div class="profile-card">
            {avatar_html}
            <div class="profile-text">
                <div class="profile-name">{user_name}</div>
                <div class="profile-email">{user_email}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Log out", icon=":material/logout:", use_container_width=True, key="logout_btn"):
        st.logout()


def render_login_page() -> None:
    """Google sign-in gate shown before the dashboard is reachable."""
    inject_custom_css()

    st.markdown("<div style='height: 12vh;'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.3, 1])

    with mid:
        st.markdown(
            """
            <div style="text-align:center;">
                <div style="font-size:2.6rem;">🔥</div>
                <div style="font-weight:900; font-size:1.5rem; color:#c2410c; margin-top:0.2rem;">CyberVision</div>
                <div style="font-size:0.7rem; color:#6b7280; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:1.4rem;">
                    Edge Fire &amp; Smoke AI
                </div>
                <p style="color:#6b7280; font-size:0.9rem; margin-bottom:1.4rem;">
                    Sign in to access the live monitoring dashboard.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Sign in with Google", icon=":material/login:", use_container_width=True, type="primary"):
            try:
                st.login("google")
            except Exception as exc:
                st.error(
                    "Google sign-in isn't configured yet. Add an [auth] / "
                    "[auth.google] section with your OAuth client credentials to "
                    f".streamlit/secrets.toml (see secrets.toml.example). Details: {exc}"
                )


def render_sidebar_nav() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-block">
                <div class="brand-icon">🔥</div>
                <div>
                    <div class="brand-name">CyberVision</div>
                    <div class="brand-subtitle">Edge Fire &amp; Smoke AI</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cam_running = st.session_state.camera_running
        wifi_online = is_wifi_connected()

        st.markdown(
            f"""
            <div class="status-footer">
                <div class="status-row">
                    <span class="status-dot" style="background:{'#30d158' if cam_running else '#ff3b30'};"></span>
                    Camera {"running" if cam_running else "stopped"}
                </div>
                <div class="status-row">
                    <span class="status-dot" style="background:{'#30d158' if wifi_online else '#ffd60a'};"></span>
                    Network {"online" if wifi_online else "offline"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sidebar-footer-divider'></div>", unsafe_allow_html=True)

        st.markdown("<div class='nav-section-label'>Workspace</div>", unsafe_allow_html=True)

        nav_items = [
            ("Dashboard", "dashboard"),
            ("Analytics", "bar_chart"),
            ("Data Logs", "table_chart"),
            ("Export Reports", "description"),
        ]

        for label, icon_name in nav_items:
            is_active = st.session_state.active_page == label
            if st.button(
                label,
                icon=f":material/{icon_name}:",
                key=f"nav_{label.replace(' ', '_').lower()}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_page = label
                st.rerun()

        # Profile + Log out live in this keyed container; the CSS class
        # ".st-key-sidebar_bottom_block" pins it to the bottom of the sidebar.
        with st.container(key="sidebar_bottom_block"):
            render_sidebar_profile()


def render_dashboard_settings_panel() -> None:
    with st.container(border=True):
        st.markdown("<div class='stats-panel-title'>Server Setup</div>", unsafe_allow_html=True)

        st.session_state.sound_enabled = st.toggle(
            "Alert sound",
            value=st.session_state.sound_enabled,
        )

        wifi_online = is_wifi_connected()
        sync_label = (
            "Data synced _to cloud_"
            if (st.session_state.cloud_sync_enabled and wifi_online)
            else "Data synced _locally_"
        )
        st.session_state.cloud_sync_enabled = st.toggle(
            sync_label,
            value=st.session_state.cloud_sync_enabled,
        )

        timeout_option = st.selectbox(
            "Inference timeout target",
            ["1.0s", "2.0s", "3.0s", "5.0s"],
            index=1,
            key="inference_timeout_select"
        )
        CONFIG["INFERENCE_TIMEOUT"] = float(timeout_option.replace("s", ""))

        st.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='stats-panel-title'>Resources</div>", unsafe_allow_html=True)

        metrics = SystemMonitor.get_metrics()
        cpu_high = metrics["cpu"] >= CONFIG["CPU_THRESHOLD"]
        ram_high = metrics["ram"] >= CONFIG["RAM_THRESHOLD"]
        cpu_delta = ":material/check_circle:" if metrics["cpu"] < 70 else ":material/warning:" if not cpu_high else ":material/error:"
        ram_delta = ":material/check_circle:" if metrics["ram"] < 70 else ":material/warning:" if not ram_high else ":material/error:"

        m1, m2 = st.columns(2)
        with m1:
            with st.container(key="cpu_metric_high" if cpu_high else "cpu_metric_normal"):
                st.metric("CPU", f"{metrics['cpu']:.1f}%", cpu_delta)
        with m2:
            with st.container(key="ram_metric_high" if ram_high else "ram_metric_normal"):
                st.metric("RAM", f"{metrics['ram']:.1f}%", ram_delta)

        if st.session_state.latency_history:
            avg_latency = sum(st.session_state.latency_history) / len(st.session_state.latency_history)
            st.metric("Avg Latency", f"{avg_latency * 1000:.0f} ms")
        else:
            st.metric("Avg Latency", "—")

        if st.session_state.last_error:
            st.info("Model/YARA backend notice: Active fallback running.")


# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================
def render_dashboard_header() -> None:
    cam_running = st.session_state.camera_running
    yolo_ready = is_yolo_model_available()

    if cam_running and yolo_ready:
        pill_text, pill_color = "YOLO ACTIVE", "#30d158"
    elif cam_running:
        pill_text, pill_color = "FALLBACK MODE", "#ffd60a"
    else:
        pill_text, pill_color = "STANDBY", "#9ca3af"

    st.markdown(
        f"""
        <div class="dash-header">
                <div>
                    <div class="dash-header-title">Live Camera Feeds</div>
                    <div class="dash-header-subtitle">Real-time AI fire &amp; smoke detection and alerting.</div>
                </div>
            </div>
            <div class="status-pill-live" style="border-color:{pill_color}66; color:{pill_color};">
                <span class="status-pill-dot" style="background:{pill_color};"></span>
                {pill_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_live_stats_body() -> None:
    latest = st.session_state.get("latest_detection") or {}

    # Severity mirrors the YOLO severity (fire/smoke rules in
    # YOLOFireSmokeDetector.hazard_severity), not the blended YARA/VLM one.
    severity = str(st.session_state.get("yolo_severity", "NORMAL")).upper()
    if severity not in SEVERITY_STYLE:
        severity = "NORMAL"

    total = len(st.session_state.alert_history)
    hazards = sum(1 for item in st.session_state.alert_history if item["is_hazard"])
    confidence = (latest.get("confidence", 0) * 100 if latest else 0)

    with st.container(border=True):
        st.markdown("<div class='stats-panel-title'>Active Monitoring</div>", unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            # The key drives the colour via the .st-key-severity_metric_* CSS:
            # NORMAL green, MEDIUM yellow, HIGH / CRITICAL red.
            with st.container(key=f"severity_metric_{severity.lower()}"):
                st.metric("Severity", severity)
        r1c2.metric("Confidence", f"{confidence:.0f}%")

        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Frames", st.session_state.frames_processed)
        r2c2.metric("Hazards", hazards, f"of {total}")


def render_live_stats_panel() -> None:
    # The camera loop runs inside its own 0.1s fragment, so a panel drawn only
    # once per full script run would stay frozen. Give the panel its own
    # fragment so the severity card updates while the camera is running.
    refresh = 0.5 if st.session_state.get("camera_running") else None
    st.fragment(_render_live_stats_body, run_every=refresh)()


def render_video_status_bar() -> None:
    fps = st.session_state.get("fps", 0.0)
    avg_latency = (
        sum(st.session_state.latency_history) / len(st.session_state.latency_history)
        if st.session_state.latency_history else 0.0
    )
    resolution = f"{CONFIG['FRAME_WIDTH']}x{CONFIG['FRAME_HEIGHT']}"

    st.markdown(
        f"""
        <div class="video-status-bar muted">
            <span>RESOLUTION: <b>{resolution}</b></span>
            <span>INFERENCE: <b>{avg_latency * 1000:.0f} ms</b></span>
            <span>FPS: <b>{fps:.1f}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_resource_trend_preview() -> None:
    metrics_df = pd.DataFrame(list(st.session_state.metrics_history))
    if metrics_df.empty:
        st.info("No resource metrics yet.")
        return
    metrics_df = metrics_df.set_index("time")
    st.line_chart(metrics_df[["cpu", "ram"]], height=180)


# =============================================================================
# LIVE VIDEO STREAM (FULL-WIDTH 3x3 MATRIX)
# =============================================================================
def render_video_frame(video_placeholder, status_placeholder, browser_processor=None) -> None:
    # Evaluated every 0.1s fragment tick (see live_camera_fragment) so the
    # hazard card fades in immediately and disappears on its own once
    # CONFIG["TOAST_DISPLAY_SECONDS"] has elapsed — no early-return above
    # this, so it still gets a last render even if Stop is pressed right
    # after a hazard fires.
    render_custom_hazard_toast()

    if not st.session_state.get("camera_running", False):
        if st.session_state.get("last_frame_rgb") is not None:
            video_placeholder.image(st.session_state.last_frame_rgb, use_container_width=True)
        else:
            video_placeholder.image(
                VideoCaptureManager.placeholder_frame("Camera Stopped", "Click Start to begin monitoring."),
                use_container_width=True,
            )
        status_placeholder.info("Camera is stopped. Detection paused.")
        return

    browser_frame = None
    if browser_processor is not None:
        with browser_processor.lock:
            if browser_processor.frame_bgr is not None:
                browser_frame = browser_processor.frame_bgr.copy()

    active_frames, primary_frame = VideoCaptureManager.capture_active_frames(browser_frame)

    if primary_frame is None and not active_frames:
        if browser_processor is not None:
            status_placeholder.info("Waiting for your browser camera — allow camera access above.")
        else:
            status_placeholder.error("Camera unavailable.")
        return

    current_time = time.time()
    last_time = st.session_state.get("last_frame_time", current_time)
    fps = 1 / max(current_time - last_time, 0.001)

    st.session_state.fps = fps
    st.session_state.last_frame_time = current_time
    st.session_state.frames_processed += 1

    pipeline_result = run_detection_pipeline(primary_frame)

    if 0 in active_frames and pipeline_result.get("annotated_frame") is not None:
        active_frames[0] = pipeline_result["annotated_frame"]

    grid_matrix = construct_3x3_grid(active_frames)

    severity = st.session_state.get("yolo_severity", "NORMAL")

    yolo_result = pipeline_result.get("yolo", {})
    yolo_status = "DETECTED" if yolo_result.get("detected") else "CLEAR"

    hud_line_1 = f"FPS: {fps:.1f} | YOLO: {yolo_status}"
    hud_line_2 = f"INF: {st.session_state.get('inference_count', 0)} | STATUS: {severity}"

    overlay = grid_matrix.copy()
    cv2.rectangle(overlay, (10, 10), (430, 82), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, grid_matrix, 0.50, 0, grid_matrix)

    cv2.putText(grid_matrix, hud_line_1, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
    cv2.putText(grid_matrix, hud_line_2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

    grid_rgb = cv2.cvtColor(grid_matrix, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame_rgb = grid_rgb

    video_placeholder.image(grid_rgb, channels="RGB", use_container_width=True)

    if not YOLO_AVAILABLE:
        status_placeholder.warning("Ultralytics is not installed.")
    elif not is_yolo_model_available():
        status_placeholder.warning("YOLO model not found.")
    elif yolo_result.get("detected"):
        detections = ", ".join(f"{d['class']} ({d['confidence']:.0%})" for d in yolo_result.get("detections", []))
        status_placeholder.warning(f"YOLO detection: {detections}")
    else:
        status_placeholder.success("Live 3x3 multi-camera monitoring active...")


@st.fragment(run_every=0.1)
def live_camera_fragment(video_container, status_container, browser_processor=None):
    render_video_frame(video_container, status_container, browser_processor)


def render_video_feed() -> None:
    header_col, btn_col1, btn_col2 = st.columns([6, 1, 1])

    with btn_col1:
        if st.button("Start", icon=":material/play_arrow:", use_container_width=True, disabled=st.session_state.camera_running, key="start_cam_btn"):
            st.session_state.camera_running = True
            st.session_state.last_error = ""
            st.session_state.last_vlm_candidate_frame = None
            st.session_state.latest_vlm_text = ""
            st.session_state.last_frame_time = time.time()
            st.session_state.latest_detection = None
            st.session_state.yolo_severity = "NORMAL"
            st.session_state.alarm_active = False

            if st.session_state.get("yolo_detector") is None:
                st.session_state.yolo_detector = YOLOFireSmokeDetector()

            st.rerun()

    with btn_col2:
        if st.button("Stop", icon=":material/stop:", use_container_width=True, disabled=not st.session_state.camera_running, key="stop_cam_btn"):
            st.session_state.camera_running = False
            st.session_state.yolo_severity = "NORMAL"
            st.session_state.alarm_active = False
            release_camera()
            st.rerun()

    browser_processor = None
    if st.session_state.camera_running and not local_camera_available():
        browser_processor = render_browser_camera_widget()

    video_placeholder = st.empty()
    status_placeholder = st.empty()

    if not st.session_state.camera_running:
        if st.session_state.get("last_frame_rgb") is not None:
            video_placeholder.image(st.session_state.last_frame_rgb, use_container_width=True)
        else:
            video_placeholder.image(
                VideoCaptureManager.placeholder_frame("Camera Stopped", "Click Start to begin monitoring."),
                use_container_width=True,
            )
        status_placeholder.info("Camera is stopped. Detection paused.")
        return

    live_camera_fragment(video_placeholder, status_placeholder, browser_processor)


def _build_forensic_display_df() -> Optional[pd.DataFrame]:
    if not st.session_state.alert_history:
        return None

    df = pd.DataFrame(list(st.session_state.alert_history))

    display_cols = [
        "timestamp", "severity", "matched_source", "matched_keyword",
        "yara_severity", "visual_severity", "confidence", "latency",
        "fire_ratio", "smoke_ratio", "success", "description",
        "yolo_detected", "yolo_classes", "yolo_confidence",
    ]

    display_df = df[display_cols].copy()
    display_df["latency_ms"] = (display_df["latency"] * 1000).round(1)
    display_df["confidence"] = ((display_df["confidence"] * 100).round(1).astype(str) + "%")
    display_df = display_df.drop(columns=["latency"])
    return display_df


def render_data_logs_page() -> None:
    header_col, btn_col = st.columns([5, 1.3])
    with header_col:
        st.markdown("### Data Logs")
        st.markdown("<p class='muted'>Full forensic event history captured by the detection pipeline.</p>", unsafe_allow_html=True)
    with btn_col:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        if st.button("Reset logs", icon=":material/restart_alt:", use_container_width=True, key="reset_logs_btn"):
            st.session_state.alert_history.clear()
            st.session_state.latency_history.clear()
            st.session_state.latest_detection = None
            st.session_state.yolo_severity = "NORMAL"
            st.success("Logs reset.")

    display_df = _build_forensic_display_df()
    if display_df is None:
        st.info("No events logged yet.")
        return

    st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)


def render_export_reports_page() -> None:
    st.markdown("### Export Reports")
    st.markdown("<p class='muted'>Download the forensic alert log for offline analysis or auditing.</p>", unsafe_allow_html=True)

    display_df = _build_forensic_display_df()
    if display_df is None:
        st.info("No events logged yet — nothing to export.")
        return

    st.metric("Events ready to export", len(display_df))

    st.download_button(
        "Download alerts log CSV",
        icon=":material/download:",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cybervision_forensic_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_csv_btn"
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
        severity_counts = alert_df["severity"].value_counts().reindex(["NORMAL", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
        with c2:
            st.markdown("#### Severity Distribution")
            st.bar_chart(severity_counts)
    else:
        c2.info("No alert statistics yet.")


def render_analytics_page() -> None:
    st.markdown("### Analytics")
    st.markdown("<p class='muted'>Resource usage trends and hazard severity distribution.</p>", unsafe_allow_html=True)
    render_charts()


def render_dashboard_page() -> None:
    metrics = SystemMonitor.get_metrics()
    if metrics["cpu"] > 90.0 or metrics["ram"] > 88.0:
        st.error(f"Critical System Load! CPU: {metrics['cpu']}% | RAM: {metrics['ram']}%. System throttling.", icon=":material/error:")

    render_dashboard_header()

    video_col, stats_col = st.columns([2.6, 1], gap="medium")

    with video_col:
        with st.container(border=True):
            render_video_feed()
            render_video_status_bar()

    with stats_col:
        render_live_stats_panel()
        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        render_dashboard_settings_panel()

    st.markdown("<div style='height: 0.7rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Live Trends")
    render_resource_trend_preview()


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    init_session_state()

    try:
        logged_in = bool(getattr(st.user, "is_logged_in", False))
    except Exception:
        logged_in = False

    if not logged_in:
        render_login_page()
        return

    start_sync_thread()
    inject_custom_css()

    render_sidebar_nav()

    page = st.session_state.active_page

    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Analytics":
        render_analytics_page()
    elif page == "Data Logs":
        render_data_logs_page()
    elif page == "Export Reports":
        render_export_reports_page()
    else:
        st.session_state.active_page = "Dashboard"
        render_dashboard_page()


if __name__ == "__main__":
    main()