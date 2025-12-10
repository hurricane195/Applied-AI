"""
--------------------------------RUN WITH THIS---------------------------------------

cd "..........."
source venv_hailo/bin/activate

python RaspberryPi_YOLOv8n_for_Halo_MediaPipe_Whisper.py \
  --camera /dev/video1 \
  --mic 2 \
  --chunk-sec 1.0 \
  --vad-threshold 0.007
------------------------------------------------------------------------------------


# assign camera and mic as needed
"""

import os
import sys
import time
import math
import argparse
import threading
import queue
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)

from mediapipe import solutions as mp_solutions
mp_hands   = mp_solutions.hands
mp_drawing = mp_solutions.drawing_utils
mp_styles  = mp_solutions.drawing_styles

# -------------------------- GPIO / LEDs --------------------------------------

os.environ.setdefault("GPIOZERO_PIN_FACTORY", "lgpio")
from gpiozero import LED

# ============================ Config =========================================

HAILO_HEF = ".......yolov8nUNCCLogo.hef"


CONF_THRES        = 0.35     # logo confidence threshold
YOLO_TARGET_CLASS = 0

DETECT_TIME = 60.0           # length of “DETECT OFF” window (gestures+voice)

# How long a logo is considered "present" after last YOLO hit
LOGO_PERSIST_SEC = 1.0

STABLE_FRAMES    = 2
GESTURE_COOLDOWN = 0.5
# Global lockout between *different* gesture tokens
GESTURE_TOKEN_GLOBAL_COOLDOWN = 1.5   # seconds

OK_DIST_THR       = 0.16
L_SEPARATION_THR  = 0.48
THUMB_UP_MIN_ANGLE_DEG = 25
THUMB_UP_MIN_DIR_Y     = 0.04
INDEX_CURL_MAX_FOR_OK_DEG = 95.0

WHISPER_MODEL     = "tiny.en"
WHISPER_DEVICE    = "cpu"
WHISPER_COMPUTE   = "int8"
VOICE_SR          = 16000
VOICE_BLOCK       = 1024
DEFAULT_CHUNK_SEC = 1.0
DEFAULT_SPEECH_TIME = 6.0

# GPIO pins
PIN_TOP    = 12
PIN_THIRD  = 13
PIN_SECOND = 16
PIN_BOTTOM = 26
PIN_LOCK   = 5
PIN_READY  = 6   # detect indicator LED
PIN_CLOSE  = 23
PIN_OPEN   = 24

TOKENS = {
    "one_finger":   "B0011E",
    "two_fingers":  "B0022E",
    "three_fingers":"B0033E",
    "four_fingers": "B0044E",
    "five_open":    "B0330E",
    "thumbs_up":    "B0110E",
    "l_sign":       "B0111E",
    "ok_sign":      "B0333E",
    "rock_on":      "B1234E",
    "call_me":      "B4321E",
}

# Expanded synonyms for voice phrases
VOICE_TO_TOKEN = {
    "lock the doors":            "B0111E",
    "unlock the doors":          "B0333E",

    "turn on the top light":     "B0044E",
    "turn on the fourth light":  "B0044E",
    "turn off the top light":    "B4400E",
    "turn off the fourth light": "B4400E",

    "turn on the bottom light":  "B0011E",
    "turn on the first light":   "B0011E",
    "turn off the bottom light": "B1100E",
    "turn off the first light":  "B1100E",

    "turn on the third light":   "B0033E",
    "turn off the third light":  "B3300E",

    "turn on the second light":  "B0022E",
    "turn off the second light": "B2200E",

    # All-lights ON
    "turn on all the lights":    "B1234E",
    "turn on the lights":        "B1234E",
    "switch on the lights":      "B1234E",

    # All-lights OFF
    "turn off all the lights":   "B4321E",
    "turn off the lights":       "B4321E",
    "switch off the lights":     "B4321E",
}

KEYWORDS = {
    "lock", "unlock", "open", "close", "shut",
    "door", "doors",
    "light", "lights",
    "top", "bottom", "first","second", "third", "fourth",
    "all",
}

# LEDs
top_led    = LED(PIN_TOP)
third_led  = LED(PIN_THIRD)
second_led = LED(PIN_SECOND)
bottom_led = LED(PIN_BOTTOM)
lock_led   = LED(PIN_LOCK)
ready_led  = LED(PIN_READY)
close_led  = LED(PIN_CLOSE)
open_led   = LED(PIN_OPEN)

def gpio_init_states():
  
    lock_led.on()
    close_led.on()
    open_led.off()
    top_led.off()
    third_led.off()
    second_led.off()
    bottom_led.off()
    ready_led.on()   # HIGH == DETECT ON (waiting for logo)

# ============================ Geometry helpers ===============================

def dist(lm, i, j):
    dx = lm[i].x - lm[j].x
    dy = lm[i].y - lm[j].y
    return math.hypot(dx, dy)

def hand_scale(lm):
    anchors = [(0,5),(0,9),(5,17),(1,17),(5,9)]
    return max(dist(lm,a,b) for a,b in anchors) + 1e-6

def angle(a,b,c):
    bax = a.x-b.x; bay = a.y-b.y
    bcx = c.x-b.x; bcy = c.y-b.y
    dot = bax*bcx + bay*bcy
    na  = math.hypot(bax,bay) + 1e-9
    nb  = math.hypot(bcx,bcy) + 1e-9
    cosv = max(-1.0, min(1.0, dot/(na*nb)))
    return math.acos(cosv)

# ============================ Hailo YOLO =====================================

class HailoYOLOInference:
    def __init__(self, hef_path: str):
        print(f"[hailo] Loading HEF: {hef_path}")
        self.target = VDevice()
        self.hef = HEF(hef_path)

        self.configure_params = ConfigureParams.create_from_hef(
            hef=self.hef,
            interface=HailoStreamInterface.PCIe
        )
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_infos = self.hef.get_output_vstream_infos()
        self.output_name = self.output_vstream_infos[0].name

        print(f"[hailo] Input stream: {self.input_vstream_info.name}, shape={self.input_vstream_info.shape}")
        print(f"[hailo] Output 0: {self.output_name}, shape={self.output_vstream_infos[0].shape}")

        self.input_height, self.input_width, self.input_channels = self.input_vstream_info.shape

        self.input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False,
            format_type=FormatType.FLOAT32
        )
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False,
            format_type=FormatType.FLOAT32
        )

        self._debug_dumped = False

    def preprocess(self, img_bgr):
        img_resized = cv2.resize(img_bgr, (self.input_width, self.input_height))
        return np.expand_dims(img_resized.astype(np.float32), axis=0)

    def infer_raw(self, img_bgr):
        input_tensor = self.preprocess(img_bgr)
        with InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params
        ) as infer_pipeline:
            input_data = {self.input_vstream_info.name: input_tensor}
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_data)

        raw = infer_results[self.output_name]

        if not self._debug_dumped:
            try:
                print("[hailo_yolo] type(raw):", type(raw))
                if isinstance(raw, list):
                    print("[hailo_yolo] len(raw):", len(raw))
                    if len(raw) > 0:
                        print("[hailo_yolo] type(raw[0]):", type(raw[0]))
                        if hasattr(raw[0], "__len__"):
                            print("[hailo_yolo] len(raw[0]):", len(raw[0]))
                        if len(raw[0]) > 0 and hasattr(raw[0][0], "shape"):
                            print("[hailo_yolo] raw[0][0].shape:", raw[0][0].shape)
                            print("[hailo_yolo] raw[0][0][:5]:", raw[0][0][:5])
                elif isinstance(raw, np.ndarray):
                    print("[hailo_yolo] raw.shape:", raw.shape, "dtype:", raw.dtype)
            except Exception as e:
                print("[hailo_yolo] debug error:", e)
            self._debug_dumped = True

        return raw

def decode_hailo_nms(raw, input_w, input_h, frame_w, frame_h, conf_thres=CONF_THRES):
    boxes = []

    if isinstance(raw, list):
        if len(raw) == 0:
            return []
        per_class = raw[0]
        if not isinstance(per_class, (list, tuple)):
            per_class = [per_class]
        if len(per_class) == 0:
            return []

        cls_idx = YOLO_TARGET_CLASS if YOLO_TARGET_CLASS < len(per_class) else 0
        dets = per_class[cls_idx]
        if not isinstance(dets, np.ndarray):
            dets = np.asarray(dets)
        if dets.ndim != 2 or dets.shape[1] < 5:
            print("[decode] unexpected dets shape (list case):", dets.shape)
            return []

        sx = frame_w / float(input_w)
        sy = frame_h / float(input_h)
        coord_debug_done = False

        for row in dets:
            x1_raw, y1_raw, x2_raw, y2_raw, conf = map(float, row[:5])
            if conf < conf_thres:
                continue

            if not coord_debug_done:
                print(f"[decode] sample det: x1_raw={x1_raw:.4f}, y1_raw={y1_raw:.4f}, "
                      f"x2_raw={x2_raw:.4f}, y2_raw={y2_raw:.4f}, conf={conf:.4f}")
                coord_debug_done = True

            max_coord = max(abs(x1_raw), abs(y1_raw), abs(x2_raw), abs(y2_raw))
            if max_coord <= 1.5:
                x1_in = x1_raw * input_w
                y1_in = y1_raw * input_h
                x2_in = x2_raw * input_w
                y2_in = y2_raw * input_h
            else:
                x1_in = x1_raw
                y1_in = y1_raw
                x2_in = x2_raw
                y2_in = y2_raw

            x1 = max(0.0, min(frame_w - 1.0, x1_in * sx))
            y1 = max(0.0, min(frame_h - 1.0, y1_in * sy))
            x2 = max(0.0, min(frame_w - 1.0, x2_in * sx))
            y2 = max(0.0, min(frame_h - 1.0, y2_in * sy))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes

    if isinstance(raw, np.ndarray):
        if raw.ndim != 3 or raw.shape[0] != 1:
            print("[decode] unexpected ndarray raw shape:", raw.shape)
            return []
        det = raw[0]
        if det.shape[0] == 5:
            det_5xN = det
        elif det.shape[1] == 5:
            det_5xN = det.T
        else:
            print("[decode] cannot interpret det ndarray shape:", det.shape)
            return []

        sx = frame_w / float(input_w)
        sy = frame_h / float(input_h)
        coord_debug_done = False

        for j in range(det_5xN.shape[1]):
            x1_raw, y1_raw, x2_raw, y2_raw, conf = map(float, det_5xN[:, j])
            if conf < conf_thres:
                continue
            if not coord_debug_done:
                print(f"[decode] sample det (ndarray): x1_raw={x1_raw:.4f}, y1_raw={y1_raw:.4f}, "
                      f"x2_raw={x2_raw:.4f}, y2_raw={y2_raw:.4f}, conf={conf:.4f}")
                coord_debug_done = True

            max_coord = max(abs(x1_raw), abs(y1_raw), abs(x2_raw), abs(y2_raw))
            if max_coord <= 1.5:
                x1_in = x1_raw * input_w
                y1_in = y1_raw * input_h
                x2_in = x2_raw * input_w
                y2_in = y2_raw * input_h
            else:
                x1_in = x1_raw
                y1_in = y1_raw
                x2_in = x2_raw
                y2_in = y2_raw

            x1 = max(0.0, min(frame_w - 1.0, x1_in * sx))
            y1 = max(0.0, min(frame_h - 1.0, y1_in * sy))
            x2 = max(0.0, min(frame_w - 1.0, x2_in * sx))
            y2 = max(0.0, min(frame_h - 1.0, y2_in * sy))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes

    print("[decode] unsupported raw type:", type(raw))
    return []

# ============================ GPIO / gestures =================================

def apply_token(tok: str):
    print(f"[GPIO] apply_token: {tok}")
    if tok == "B0111E":
        lock_led.on()
    elif tok == "B0333E":
        lock_led.off()
    elif tok == "B0110E":
        close_led.off()
        open_led.on()
    elif tok == "B0330E":
        open_led.off()
        close_led.on()
    elif tok == "B0044E":
        top_led.on()
    elif tok == "B0033E":
        third_led.on()
    elif tok == "B0022E":
        second_led.on()
    elif tok == "B0011E":
        bottom_led.on()
    elif tok == "B4400E":
        top_led.off()
    elif tok == "B3300E":
        third_led.off()
    elif tok == "B2200E":
        second_led.off()
    elif tok == "B1100E":
        bottom_led.off()
    elif tok == "B1234E":
        top_led.on(); third_led.on(); second_led.on(); bottom_led.on()
    elif tok == "B4321E":
        top_led.off(); third_led.off(); second_led.off(); bottom_led.off()

@dataclass
class HandCtx:
    landmarks: any
    handedness: str = "Right"

def finger_up(lm, tip, pip, mcp, tol):
    return (lm[tip].y < lm[pip].y - tol) and (lm[pip].y < lm[mcp].y - tol)

def classify_gesture(lm, handed):
    T = [4,8,12,16,20]
    P = [3,6,10,14,18]
    M = [2,5,9,13,17]
    sc = hand_scale(lm)
    tol = 0.03 * sc

    idx_up   = finger_up(lm, T[1], P[1], M[1], tol)
    mid_up   = finger_up(lm, T[2], P[2], M[2], tol)
    ring_up  = finger_up(lm, T[3], P[3], M[3], tol)
    pinky_up = finger_up(lm, T[4], P[4], M[4], tol)

    thumb_angle_deg = math.degrees(angle(lm[M[0]], lm[P[0]], lm[T[0]]))
    thumb_straight = thumb_angle_deg <= THUMB_UP_MIN_ANGLE_DEG
    if handed.lower().startswith("right"):
        thumb_side = (lm[T[0]].x < lm[P[0]].x - 0.015*sc)
    else:
        thumb_side = (lm[T[0]].x > lm[P[0]].x + 0.015*sc)
    dir_y_up = (lm[0].y - lm[T[0]].y)
    thumb_up = (thumb_straight or dir_y_up > THUMB_UP_MIN_DIR_Y) and thumb_side

    index_curl_deg = math.degrees(angle(lm[M[1]], lm[P[1]], lm[T[1]]))

    up = dict(thumb=thumb_up, index=idx_up, middle=mid_up, ring=ring_up, pinky=pinky_up)
    n_up = sum(up.values())
    d_thumb_idx = dist(lm, T[0], T[1]) / sc

    if (d_thumb_idx < OK_DIST_THR) and ((not idx_up) or (index_curl_deg < INDEX_CURL_MAX_FOR_OK_DEG)):
        return "ok_sign", up
    if pinky_up and (not idx_up) and (not mid_up) and (not ring_up) and thumb_side:
        return "call_me", up
    if (idx_up and pinky_up) and (not mid_up) and (not ring_up):
        return "rock_on", up
    if (thumb_up and idx_up) and (not mid_up) and (not ring_up) and (not pinky_up) and (d_thumb_idx > L_SEPARATION_THR):
        return "l_sign", up
    if thumb_up and (not idx_up) and (not mid_up) and (not ring_up) and (not pinky_up) and (dir_y_up > THUMB_UP_MIN_DIR_Y):
        return "thumbs_up", up
    if n_up == 5:  return "five_open",     up
    if n_up == 4:  return "four_fingers",  up
    if n_up == 3:  return "three_fingers", up
    if n_up == 2:  return "two_fingers",   up
    if n_up == 1 and idx_up:
        return "one_finger", up
    return None, up

# ============================ Camera helpers =================================

def try_open(device, api, width, height, test_timeout=2.0):
    cap = cv2.VideoCapture(device, api)
    if not cap.isOpened():
        return None
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    t0 = time.time()
    ok, _ = cap.read()
    while not ok and (time.time() - t0) < test_timeout:
        time.sleep(0.05)
        ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    print(
        f"[cam] opened {device} (api={api}) "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )
    return cap

def open_camera_resilient(req, width, height):
    cand = []
    if req is not None:
        try:
            idx = int(str(req))
            cand += [(idx, cv2.CAP_V4L2), (idx, cv2.CAP_ANY)]
        except ValueError:
            s = str(req)
            cand += [(s, cv2.CAP_V4L2), (s, cv2.CAP_ANY)]
    by_id = "/dev/v4l/by-id"
    if os.path.isdir(by_id):
        for n in sorted(os.listdir(by_id)):
            if "video-index0" in n or "video0" in n:
                p = os.path.join(by_id, n)
                cand += [(p, cv2.CAP_V4L2), (p, cv2.CAP_ANY)]
    for p in ("/dev/video0", "/dev/video1"):
        cand += [(p, cv2.CAP_V4L2), (p, cv2.CAP_ANY)]
    for i in (0, 1, 2, 3):
        cand += [(i, cv2.CAP_V4L2), (i, cv2.CAP_ANY)]
    tried = set()
    for dev, api in cand:
        k = f"{dev}|{api}"
        if k in tried:
            continue
        tried.add(k)
        cap = try_open(dev, api, width, height)
        if cap is not None:
            return cap
    raise RuntimeError("No usable camera found. Try --camera /dev/videoX or --camera N")

# ============================ Whisper / VoiceGate ============================

def normalize_text(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha() or c.isspace()).strip()

def match_command(norm: str):
    # Doors: lock / unlock 
    if "door" in norm or "doors" in norm:
        if "unlock" in norm:
            return "B0333E"
        if "lock" in norm:
            return "B0111E"

    # LIGHTS
    if "light" in norm or "lights" in norm:
        if "all" in norm:
            if "off" in norm:
                return "B4321E"
            if "on" in norm or "up" in norm:
                return "B1234E"
        if "top" in norm:
            if "off" in norm:
                return "B4400E"
            if "on" in norm or "up" in norm:
                return "B0044E"
        if "bottom" in norm:
            if "off" in norm:
                return "B1100E"
            if "on" in norm or "up" in norm:
                return "B0011E"
        if "second" in norm:
            if "off" in norm:
                return "B2200E"
            if "on" in norm or "up" in norm:
                return "B0022E"
        if "third" in norm:
            if "off" in norm:
                return "B3300E"
            if "on" in norm or "up" in norm:
                return "B0033E"

    # Fallback: exact phrase lookup (including synonyms in VOICE_TO_TOKEN)
    for phrase, token in VOICE_TO_TOKEN.items():
        if phrase in norm:
            return token

    return None

def resolve_audio_device(user_arg):
    devs = sd.query_devices()
    print("\n[audio] Available devices:")
    for idx, d in enumerate(devs):
        print(
            f"  {idx}: {d['name']} (in={d['max_input_channels']}, "
            f"out={d['max_output_channels']}, sr={d['default_samplerate']})"
        )

    if user_arg is None:
        print("[audio] No --mic specified, using default input.")
        return None

    try:
        idx = int(str(user_arg))
        if 0 <= idx < len(devs) and devs[idx]["max_input_channels"] > 0:
            print(f"[audio] Using device index {idx}: {devs[idx]['name']}")
            return idx
        else:
            print(f"[audio] Index {idx} invalid or no input channels.")
    except ValueError:
        pass

    target = str(user_arg).lower()
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and target in d["name"].lower():
            print(f"[audio] Using device name match {i}: {d['name']}")
            return i

    print(f"[audio] Could not resolve mic '{user_arg}', using default.")
    return None

class VoiceGate:
    def __init__(self, mic_device=None, speech_time=DEFAULT_SPEECH_TIME,
                 use_wake=False, chunk_sec=DEFAULT_CHUNK_SEC,
                 vad_threshold=0.01):
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE
        )
        self.stop_evt = threading.Event()
        self.cmd_queue = queue.Queue()

        self.use_wake = use_wake
        self.state = "idle"
        self.capture_until = 0.0

        self.buf = np.zeros(0, dtype=np.float32)
        self.chunk_sec = max(0.5, float(chunk_sec))
        self.chunk_len = int(VOICE_SR * self.chunk_sec)
        self.mic_device = mic_device
        self.speech_time = speech_time
        self.vad_threshold = float(vad_threshold)

        self.history = []
        self.HIST_MAX = 5
        self.WINDOW_CHUNKS = 3

        self.last_token = None
        self.last_token_time = 0.0
        self.TOKEN_COOLDOWN = 1.5

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)

    def _reset_history(self):
        self.history.clear()
        print("[voice] history cleared after command.")

    def _maybe_push_token(self, tok, text, label):
        now = time.time()
        if tok == self.last_token and (now - self.last_token_time) < self.TOKEN_COOLDOWN:
            print(f"[voice {label}] token {tok} suppressed (cooldown).")
            return
        self.last_token = tok
        self.last_token_time = now
        print(f"[voice {label}] Command matched: '{text}' → {tok}")
        self.cmd_queue.put(("voice", tok, text))
        self._reset_history()

    def _run(self):
        def cb(indata, frames, t, status):
            if status:
                print("[voice] stream status:", status)
            try:
                self.buf = np.concatenate([self.buf, indata[:, 0]])
            except Exception as e:
                print("[voice] buffer concat error:", e)

        stream_kwargs = dict(
            samplerate=VOICE_SR,
            channels=1,
            dtype="float32",
            blocksize=VOICE_BLOCK,
            callback=cb,
        )
        if self.mic_device is not None:
            stream_kwargs["device"] = self.mic_device

        print(f"[voice] Using mic device: {self.mic_device if self.mic_device is not None else 'default'}")
        print(f"[voice] Chunk length: {self.chunk_sec:.2f} s, VAD threshold: {self.vad_threshold:.4f}")

        try:
            with sd.InputStream(**stream_kwargs):
                print("[voice] Speak commands directly.")

                while not self.stop_evt.is_set():
                    if len(self.buf) < self.chunk_len:
                        time.sleep(0.05)
                        continue

                    seg = self.buf[:self.chunk_len]
                    self.buf = self.buf[self.chunk_len:]
                    if seg.size == 0:
                        continue

                    rms = float(np.sqrt(np.mean(seg**2)))
                    if rms < self.vad_threshold:
                        continue

                    segments, _ = self.model.transcribe(seg, language="en")
                    text = "".join(s.text for s in segments).strip()
                    if not text:
                        continue

                    norm = normalize_text(text)
                    print(f"[voice raw] '{text}'  (norm='{norm}', rms={rms:.4f})")

                    words = norm.split()
                    if len(words) == 1 and words[0] not in KEYWORDS:
                        print(f"[voice] ignoring short noise: '{norm}'")
                        continue

                    if norm:
                        self.history.append(norm)
                        if len(self.history) > self.HIST_MAX:
                            self.history = self.history[-self.HIST_MAX:]

                    window_norm = normalize_text(
                        " ".join(self.history[-self.WINDOW_CHUNKS:])
                    )

                    tok = match_command(window_norm)
                    if tok:
                        self._maybe_push_token(tok, text, "win")
                    else:
                        tok_single = match_command(norm)
                        if tok_single:
                            self._maybe_push_token(tok_single, text, "single")

        except Exception as e:
            print("[voice] InputStream failed, disabling voice thread:", e)

# ============================ MAIN ==========================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default=None)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mic", default=None)
    p.add_argument("--speech-time", type=float, default=DEFAULT_SPEECH_TIME)
    p.add_argument("--chunk-sec", type=float, default=DEFAULT_CHUNK_SEC)
    p.add_argument("--vad-threshold", type=float, default=0.01)
    args = p.parse_args()

    print("Python:", sys.executable)
    gpio_init_states()

    try:
        hailo_yolo = HailoYOLOInference(HAILO_HEF)
    except Exception as e:
        print("[hailo] Failed to initialize:", e)
        return

    hands = mp_hands.Hands(
        model_complexity=0, max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = open_camera_resilient(args.camera, args.width, args.height)

    vg = None
    try:
        mic_index = resolve_audio_device(args.mic)
        vg = VoiceGate(
            mic_device=mic_index,
            speech_time=args.speech_time,
            use_wake=False,
            chunk_sec=args.chunk_sec,
            vad_threshold=args.vad_threshold,
        )
        vg.start()
    except Exception as e:
        print("[voice] Failed to initialize VoiceGate:", e)
        vg = None

    # detect_active == False  → DETECT ON (awaiting logo)
    # detect_active == True   → DETECT OFF (timer running)
    detect_active = False
    detect_until = 0.0

    gest_hist = deque(maxlen=8)
    last_applied = None
    last_apply_t = 0.0
    stable_gesture = None

    # Last *gesture token* that actually fired (not just label)
    last_gesture_token = None
    last_gesture_token_time = 0.0

    last_voice_text = ""

    # last time we saw a "valid" logo (YOLO)
    last_logo_time = 0.0

    t0, frames = time.time(), 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[cam] read failed")
                break

            frames += 1
            if frames >= 10:
                now_fps = time.time()
                fps = frames / (now_fps - t0 + 1e-9)
                t0, frames = now_fps, 0

            h0, w0 = frame.shape[:2]

            # --- YOLO ---
            logo_boxes = []
            logo_scores = []
            raw_boxes_count = 0
            logo_detected_now = False
            max_conf = 0.0

            try:
                raw = hailo_yolo.infer_raw(frame)
                boxes_raw = decode_hailo_nms(
                    raw,
                    input_w=hailo_yolo.input_width,
                    input_h=hailo_yolo.input_height,
                    frame_w=w0,
                    frame_h=h0,
                    conf_thres=CONF_THRES,
                )
                raw_boxes_count = len(boxes_raw)

                # Fill logo_boxes in [x, y, w, h] space (YOLO-only)
                for (x1, y1, x2, y2, conf) in boxes_raw:
                    w_box = x2 - x1
                    h_box = y2 - y1
                    logo_boxes.append([x1, y1, w_box, h_box])
                    logo_scores.append(conf)

                logo_detected_now = len(logo_boxes) > 0
                max_conf = max(logo_scores) if logo_scores else 0.0

            except Exception as e:
                print("[hailo yolo] inference error:", e)

            # Draw logo boxes
            for i, b in enumerate(logo_boxes):
                x, y, w_box, h_box = b
                conf = logo_scores[i]
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"LOGO {conf:.2f}",
                    (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # current time
            now = time.time()

            # update "recent logo" timestamp
            if logo_detected_now:
                last_logo_time = now

            logo_recent = (now - last_logo_time) < LOGO_PERSIST_SEC

            # --- DETECT state machine ---
            if not detect_active and logo_detected_now:
                detect_active = True
                detect_until = now + DETECT_TIME
                gest_hist.clear()
                stable_gesture = None
                last_applied = None
                last_gesture_token = None
                last_gesture_token_time = 0.0
                print("[state] DETECT → ACTIVE (logo hit)")
            elif detect_active and now >= detect_until:
                detect_active = False
                gest_hist.clear()
                stable_gesture = None
                last_applied = None
                last_gesture_token = None
                last_gesture_token_time = 0.0
                print("[state] DETECT → INACTIVE (timer expired)")

            # Keep detect LED strictly tied to detect state:
            # DETECT ON  (awaiting logo) -> LED HIGH
            # DETECT OFF (window active) -> LED LOW
            if detect_active:
                ready_led.off()
            else:
                ready_led.on()

            # NEW gating:
            # gestures enabled ONLY if DETECT active AND logo not recent
            gestures_blocked_by_logo = detect_active and logo_recent
            gestures_enabled = detect_active and not gestures_blocked_by_logo

            # if blocked by logo, wipe gesture state
            if gestures_blocked_by_logo:
                gest_hist.clear()
                stable_gesture = None
                last_applied = None
                last_gesture_token = None
                last_gesture_token_time = 0.0

            applied_token = None

            # --------- Gestures ----------
            if gestures_enabled:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(img_rgb)
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0].landmark
                    handed = "Right"
                    if res.multi_handedness:
                        handed = res.multi_handedness[0].classification[0].label
                    g, _ = classify_gesture(lm, handed)
                    if g:
                        gest_hist.append(g)
                    mp_drawing.draw_landmarks(
                        frame,
                        res.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                stabilized = None
                if len(gest_hist) >= STABLE_FRAMES:
                    last_n = list(gest_hist)[-STABLE_FRAMES:]
                    if len(set(last_n)) == 1:
                        stabilized = last_n[-1]
                stable_gesture = stabilized or stable_gesture

                if stable_gesture:
                    gesture_token = TOKENS.get(stable_gesture)

                    if gesture_token:
                        # 1) per-gesture debounce
                        can_fire_label = (
                            (last_applied != stable_gesture) or
                            (now - last_apply_t > GESTURE_COOLDOWN)
                        )

                        # 2) global token cooldown: block different tokens for a short window
                        if last_gesture_token is not None:
                            dt = now - last_gesture_token_time
                            if (gesture_token != last_gesture_token) and (
                                dt < GESTURE_TOKEN_GLOBAL_COOLDOWN
                            ):
                                print(
                                    f"[gest] token {gesture_token} blocked: different from last token "
                                    f"{last_gesture_token} (dt={dt:.2f}s < {GESTURE_TOKEN_GLOBAL_COOLDOWN}s)"
                                )
                                can_fire_global = False
                            else:
                                can_fire_global = True
                        else:
                            can_fire_global = True

                        if can_fire_label and can_fire_global:
                            last_applied = stable_gesture
                            last_apply_t = now
                            last_gesture_token = gesture_token
                            last_gesture_token_time = now
                            apply_token(gesture_token)
                            applied_token = gesture_token

            # --------- Voice queue ----------
            if vg is not None:
                while True:
                    try:
                        source, token, raw_text = vg.cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        if detect_active:
                            print(
                                f"[main] Voice command accepted (DETECT active): "
                                f"{token} '{raw_text}'"
                            )
                            apply_token(token)
                            applied_token = token
                            last_voice_text = raw_text
                            # clear gesture state so last pose doesn't re-fire
                            gest_hist.clear()
                            stable_gesture = None
                            last_applied = None
                            last_gesture_token = None
                            last_gesture_token_time = 0.0
                        else:
                            print(
                                f"[main] Voice command IGNORED (DETECT off): "
                                f"{token} '{raw_text}'"
                            )

            # --------- HUD overlays ----------
            if detect_active:
                remain = max(0.0, detect_until - now)
                cv2.putText(
                    frame,
                    f"DETECT OFF ({remain:4.1f}s)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                if gestures_blocked_by_logo:
                    gmsg = "GESTURES: BLOCKED (logo)"
                    gcol = (0, 165, 255)
                else:
                    gmsg = "GESTURES: ACTIVE"
                    gcol = (0, 255, 0)

                cv2.putText(
                    frame,
                    gmsg,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    gcol,
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "DETECT ON (awaiting logo)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            if applied_token:
                cv2.putText(
                    frame,
                    f"CMD TOKEN: {applied_token}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            if stable_gesture:
                cv2.putText(
                    frame,
                    f"Gesture: {stable_gesture}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
            if last_voice_text:
                cv2.putText(
                    frame,
                    f"Last voice: {last_voice_text}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 255, 200),
                    1,
                )

            cv2.putText(
                frame,
                f"boxes:{len(logo_boxes)} raw:{raw_boxes_count} max:{max_conf:.2f}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 255),
                2,
            )

            cv2.putText(
                frame,
                f"{fps:.1f} FPS",
                (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Pi + Hailo (YOLO + Gestures + Whisper) Rev9", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("[main] Ctrl+C, exiting...")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        if vg is not None:
            vg.stop()
        try:
            gpio_init_states()
        except Exception:
            pass
        print("[exit] bye")

if __name__ == "__main__":
    main()
