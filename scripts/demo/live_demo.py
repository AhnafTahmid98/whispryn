
"""WHISPRYN live sound dashboard.

Runs Google YAMNet and our Random Forest model.
Audio is processed locally. No servo movement.
"""

import json
import os
import select
import subprocess
import threading
import time

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import librosa
import numpy as np

from ai_edge_litert.interpreter import Interpreter

from camera_feed import CameraFeed


# -----------------------------------------
# 1. SETTINGS
# -----------------------------------------

ROOT = Path(__file__).resolve().parents[2]

YAMNET_DIR = ROOT / "runtime/models/yamnet"
RF_FILE = ROOT / "runtime/training/whispryn_rf_v2.joblib"
LOG_FILE = ROOT / "runtime/logs/microphone.log"

AUDIO_DEVICE = os.environ.get(
    "WHISPRYN_AUDIO_DEVICE", "plughw:2,0"
)

RATE = 16000
CHUNK_SAMPLES = 8000
RF_SAMPLES = RATE * 3

GOOGLE_CLASSES = ["Clapping", "Whistling", "Speech"]
RF_CLASSES = ["background", "clap", "hey", "whistle"]


# -----------------------------------------
# 2. LIVE DATA
# -----------------------------------------

state = {
    "status": "Starting",
    "error": "",
    "top": "Waiting for microphone",
    "yamnet": [0.0, 0.0, 0.0],
    "rf": [0.0, 0.0, 0.0, 0.0],
    "rf_name": "Waiting for audio",
    "history": [],
    "rms": 0.0,
    "yamnet_ms": 0.0,
    "rf_ms": 0.0,
    "device": AUDIO_DEVICE,
}

lock = threading.Lock()
stop = threading.Event()
camera = CameraFeed(
    stop,
    ROOT / "runtime/logs/camera.log"
)
recorder = None


def set_state(**changes):
    with lock:
        state.update(changes)


# -----------------------------------------
# 3. MICROPHONE ERROR HANDLING
# -----------------------------------------

def microphone_error():
    """Read the actual ALSA error message."""

    if LOG_FILE.exists():
        details = LOG_FILE.read_text(
            errors="replace"
        ).strip()

        if details:
            return details[-600:]

    return (
        "No ALSA details. Check the microphone "
        "connection and recording device."
    )


def read_chunk(process):
    """Read half a second of microphone audio."""

    needed = CHUNK_SAMPLES * 2
    result = bytearray()

    while len(result) < needed and not stop.is_set():

        ready, _, _ = select.select(
            [process.stdout], [], [], 1.0
        )

        if not ready:
            if process.poll() is not None:
                raise RuntimeError(microphone_error())
            continue

        part = os.read(
            process.stdout.fileno(),
            needed - len(result)
        )

        if not part:
            process.wait(timeout=2)
            raise RuntimeError(microphone_error())

        result.extend(part)

    if len(result) != needed:
        return None

    return bytes(result)


# -----------------------------------------
# 4. LOAD AI AND PROCESS LIVE SOUND
# -----------------------------------------

def listen():
    global recorder

    try:
        # Check that both models exist.
        yamnet_file = YAMNET_DIR / "yamnet.tflite"

        if not yamnet_file.exists():
            raise FileNotFoundError(
                "YAMNet model is missing."
            )

        if not RF_FILE.exists():
            raise FileNotFoundError(
                "Random Forest v2 model is missing."
            )

        # Load Google YAMNet.
        yamnet = Interpreter(
            model_path=str(yamnet_file),
            num_threads=2
        )

        yamnet.allocate_tensors()

        labels = (
            YAMNET_DIR / "yamnet_label_list.txt"
        ).read_text().splitlines()

        indices = [
            labels.index(name)
            for name in GOOGLE_CLASSES
        ]

        input_info = yamnet.get_input_details()[0]
        output_info = yamnet.get_output_details()[0]

        yamnet_samples = int(
            np.prod(input_info["shape"])
        )

        # Load our trained Random Forest.
        saved = joblib.load(RF_FILE)
        rf_model = saved["model"]

        # Start ReSpeaker microphone capture.
        LOG_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        command = [
            "arecord",
            "-q",
            "-D", AUDIO_DEVICE,
            "-t", "raw",
            "-f", "S16_LE",
            "-r", str(RATE),
            "-c", "1",
        ]

        with LOG_FILE.open("w") as log:

            recorder = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=log,
                bufsize=0
            )

            set_state(
                status="Listening",
                error=""
            )

            audio = np.empty(
                0, dtype=np.float32
            )

            history = deque(maxlen=80)
            paired = deque(maxlen=80)

            while not stop.is_set():

                raw = read_chunk(recorder)

                if raw is None:
                    break

                # Convert PCM audio into float samples.
                part = np.frombuffer(
                    raw, dtype="<i2"
                ).astype(np.float32) / 32768.0

                # Keep the latest 3 seconds.
                audio = np.concatenate(
                    (audio, part)
                )[-RF_SAMPLES:]

                volume = float(
                    np.sqrt(np.mean(part ** 2))
                )

                if len(audio) < yamnet_samples:
                    continue

                # ---------------------------------
                # Google YAMNet prediction
                # ---------------------------------

                clip = audio[-yamnet_samples:]

                yamnet.set_tensor(
                    input_info["index"],
                    clip.reshape(input_info["shape"])
                )

                started = time.perf_counter()

                yamnet.invoke()

                yamnet_ms = (
                    time.perf_counter() - started
                ) * 1000

                scores = yamnet.get_tensor(
                    output_info["index"]
                ).reshape(-1)

                google_scores = [
                    float(scores[i])
                    for i in indices
                ]

                top = labels[
                    int(np.argmax(scores))
                ]

                history.append(google_scores)

                # ---------------------------------
                # WHISPRYN Random Forest prediction
                # ---------------------------------

                rf_scores = [0.0] * len(RF_CLASSES)
                rf_name = "Collecting 3 seconds of audio"
                rf_ms = 0.0

                if len(audio) >= RF_SAMPLES:

                    started = time.perf_counter()

                    mfcc = librosa.feature.mfcc(
                        y=audio,
                        sr=RATE,
                        n_mfcc=13
                    )

                    # Same order as RF v2 training:
                    # 13 means, then 13 std values.
                    features = np.concatenate([
                        mfcc.mean(axis=1),
                        mfcc.std(axis=1)
                    ])

                    probabilities = rf_model.predict_proba(
                        features.reshape(1, -1)
                    )[0]

                    results = dict(
                        zip(
                            rf_model.classes_,
                            probabilities
                        )
                    )

                    rf_scores = [
                        float(results.get(name, 0.0))
                        for name in RF_CLASSES
                    ]

                    rf_name = str(
                        rf_model.classes_[
                            int(np.argmax(probabilities))
                        ]
                    )

                    rf_ms = (
                        time.perf_counter() - started
                    ) * 1000

                if rf_name in RF_CLASSES:
                    paired.append([google_scores, rf_scores])

                # Update the dashboard.
                set_state(
                    status="Listening",
                    error="",
                    top=top,
                    yamnet=google_scores,
                    rf=rf_scores,
                    rf_name=rf_name,
                    history=list(history),
                    paired=list(paired),
                    rms=volume,
                    yamnet_ms=yamnet_ms,
                    rf_ms=rf_ms,
                )

    except Exception as error:

        message = str(error)

        print(
            "Microphone/AI error:",
            message,
            flush=True
        )

        set_state(
            status="Error",
            error=message
        )

    finally:

        if recorder is not None:

            if recorder.poll() is None:
                recorder.terminate()

            try:
                recorder.wait(timeout=2)
            except subprocess.TimeoutExpired:
                recorder.kill()
                recorder.wait()

            if recorder.stdout:
                recorder.stdout.close()


# -----------------------------------------
# 5. DASHBOARD HTML, CSS AND JAVASCRIPT
# -----------------------------------------

HOME = r"""
<!doctype html>
<html lang="en">

<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>WHISPRYN · Live Audio Lab</title>

<style>

:root {
    color-scheme: light;
    font-family: Inter, system-ui, Arial, sans-serif;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #ffffff;
    color: #17243b;
}

header {
    border-bottom: 1px solid #e6ebf2;
}

.top {
    max-width: 1540px;
    margin: auto;
    padding: 21px 30px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 15px;
}

.logo {
    background: #153f83;
    color: white;
    width: 45px;
    height: 45px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    font-weight: 900;
    font-size: 24px;
}

h1 {
    margin: 0;
    font-size: 24px;
    letter-spacing: -0.7px;
}

.sub {
    color: #718198;
    margin-top: 4px;
    font-size: 13px;
}

.status {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 15px;
    border: 1px solid #dfe6ef;
    border-radius: 30px;
    font-weight: 650;
    font-size: 13px;
}

.dot {
    width: 9px;
    height: 9px;
    background: #a3afbf;
    border-radius: 50%;
}

.status.ok .dot {
    background: #19a46e;
}

.status.bad .dot {
    background: #d94f4f;
}

main {
    max-width: 1540px;
    margin: 0 auto;
    padding: 28px 30px 45px;
}

.heading {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 22px;
}

.heading h2 {
    font-size: 29px;
    letter-spacing: -0.9px;
    margin: 0;
}

.heading p {
    margin: 8px 0 0;
    color: #66768b;
}

.note {
    color: #66768b;
    font-size: 12px;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 18px;
}

.card {
    background: white;
    border: 1px solid #e3e9f1;
    border-radius: 15px;
    padding: 22px;
    box-shadow: 0 3px 14px rgba(19, 46, 87, 0.045);
    min-width: 0;
}

.metric-label {
    color: #677992;
    font-size: 13px;
    font-weight: 650;
}

.metric-value {
    margin: 10px 0 4px;
    font-size: 28px;
    font-weight: 750;
    letter-spacing: -0.7px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.metric-note {
    color: #8190a3;
    font-size: 12px;
}

.layout {
    display: grid;
    grid-template-columns: minmax(0, 1.65fr) minmax(330px, 1fr);
    gap: 18px;
    align-items: start;
}

.left, .right {
    display: grid;
    gap: 18px;
    min-width: 0;
}

.card-header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    margin-bottom: 18px;
}

.card h3 {
    margin: 0;
    font-size: 18px;
    letter-spacing: -0.25px;
}

.small-tag {
    color: #65778b;
    font-size: 12px;
    background: #f3f6fa;
    padding: 7px 10px;
    border-radius: 8px;
}

canvas {
    display: block;
    width: 100%;
    height: 285px;
}

.camera-preview {
    display: block;
    width: 100%;
    aspect-ratio: 16 / 9;
    object-fit: contain;
    background: #101827;
    border-radius: 12px;
}

.legend {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    color: #53657c;
    font-size: 13px;
    margin-top: 15px;
}

.key {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 6px;
}

.bar-row {
    margin: 17px 0;
}

.bar-head {
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    margin-bottom: 9px;
}

.bar-head strong {
    font-variant-numeric: tabular-nums;
}

.track {
    height: 11px;
    background: #eef2f6;
    border-radius: 20px;
    overflow: hidden;
}

.fill {
    width: 0;
    height: 100%;
    border-radius: 20px;
    transition: width 0.25s ease;
}

#message {
    display: none;
    border: 1px solid #f1c1c1;
    background: #fff7f7;
    border-radius: 12px;
    padding: 15px;
    color: #963333;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    margin-bottom: 18px;
}

#message.visible {
    display: block;
}

.below {
    border-top: 1px solid #eef2f6;
    margin-top: 18px;
    padding-top: 14px;
    color: #78889b;
    font-size: 12px;
    line-height: 1.55;
}

.info {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}

.info div {
    background: #f7f9fc;
    border-radius: 9px;
    padding: 12px;
    font-size: 12px;
    color: #718198;
}

.info strong {
    display: block;
    font-size: 15px;
    color: #203650;
    margin-top: 6px;
}

footer {
    margin-top: 22px;
    color: #73839a;
    font-size: 12px;
    line-height: 1.6;
}

@media (max-width: 1000px) {

    .metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .layout {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 600px) {

    main {
        padding: 18px 14px;
    }

    .top {
        padding: 15px;
    }

    .heading h2 {
        font-size: 23px;
    }

    .metrics {
        gap: 10px;
    }

    .card {
        padding: 16px;
    }

    .metric-value {
        font-size: 20px;
    }
}

</style>
</head>

<body>

<header>
<div class="top">

    <div class="brand">
        <div class="logo">W</div>

        <div>
            <h1>WHISPRYN</h1>
            <div class="sub">
                Edge AI · Live Sound Recognition
            </div>
        </div>
    </div>

    <div class="status" id="status">
        <span class="dot"></span>
        <span id="statusText">Starting...</span>
    </div>

</div>
</header>

<main>

<div class="heading">

    <div>
        <h2>Live Audio Dashboard</h2>
        <p>
            Raspberry Pi 5 · ReSpeaker microphone · Sound classification
        </p>
    </div>

    <div class="note">
        Audio processed locally · No servo control
    </div>

</div>

<div id="message"></div>

<!-- Top summary cards -->

<section class="metrics">

    <div class="card">
        <div class="metric-label">
            GOOGLE YAMNET · TOP EVENT
        </div>
        <div class="metric-value" id="topEvent">—</div>
        <div class="metric-note">General sound category</div>
    </div>

    <div class="card">
        <div class="metric-label">
            WHISPRYN RF · PREDICTION
        </div>
        <div class="metric-value" id="rfEvent">—</div>
        <div class="metric-note">Four custom sound classes</div>
    </div>

    <div class="card">
        <div class="metric-label">
            YAMNET INFERENCE
        </div>

        <div class="metric-value">
            <span id="latency">—</span>
            <small style="font-size:15px">ms</small>
        </div>

        <div class="metric-note">
            Processing one audio window
        </div>
    </div>

    <div class="card">
        <div class="metric-label">
            RF OFFLINE ACCURACY
        </div>

        <div class="metric-value">93.8%</div>

        <div class="metric-note">
            80 recordings · cross-person test
        </div>
    </div>

</section>

<!-- Wide desktop dashboard -->

<section class="layout">

    <div class="left">

        <div class="card">
            <div class="card-header">
                <h3>Live Camera</h3>

                <span class="small-tag" id="cameraStatus">
                    Connecting...
                </span>
            </div>

            <img
                class="camera-preview"
                src="/camera.mjpg"
                alt="WHISPRYN live Raspberry Pi camera"
            >

            <div class="below">
                Camera Module 2 NoIR · 640 × 360 · 10 FPS
                · Live preview only
            </div>
        </div>

        <div class="card">

            <div class="card-header">
                <h3>Sound score history</h3>
                <span class="small-tag">
                    Last ~40 seconds
                </span>
            </div>

            <canvas id="graph"></canvas>

            <div class="legend">

                <span>
                    <i class="key"
                       style="background:#2777df"></i>
                    Clapping
                </span>

                <span>
                    <i class="key"
                       style="background:#16a67e"></i>
                    Whistling
                </span>

                <span>
                    <i class="key"
                       style="background:#e5a331"></i>
                    Speech
                </span>

            </div>
        </div>

        <div class="card">

            <div class="card-header">
                <h3>Google YAMNet</h3>
                <span class="small-tag">
                    Pretrained · 521 categories
                </span>
            </div>

            <div id="googleBars"></div>

            <div class="below">
                Scores are not calibrated accuracy.
                Speech detection does not identify
                the word "Hey".
            </div>

        </div>

    </div>

    <div class="right">

        <div class="card">

            <div class="card-header">
                <h3>WHISPRYN Random Forest v2</h3>
                <span class="small-tag">
                    Custom-trained model
                </span>
            </div>

            <div id="rfBars"></div>

            <div class="below">
                Uses rolling 3-second audio.
                Live performance may differ from
                the offline 93.8% result.
            </div>

        </div>

        <div class="card">

            <div class="card-header">
                <h3>System information</h3>
            </div>

            <div class="info">

                <div>
                    Microphone
                    <strong id="device">—</strong>
                </div>

                <div>
                    Audio level (RMS)
                    <strong id="rms">—</strong>
                </div>

                <div>
                    Random Forest processing
                    <strong id="rfMs">—</strong>
                </div>

                <div>
                    Audio sample rate
                    <strong>16 kHz</strong>
                </div>

            </div>

        </div>

    </div>

</section>

<footer>
    WHISPRYN research prototype.
    Audio is analyzed in approximately 0.5-second steps.
    The browser refreshes once per second.
    No WAV recordings or models are modified by this dashboard.
</footer>

</main>

<script>

const googleNames = [
    "Clapping",
    "Whistling",
    "Speech"
];

const googleColors = [
    "#2777df",
    "#16a67e",
    "#e5a331"
];

const rfNames = [
    "background",
    "clap",
    "hey",
    "whistle"
];

const rfColors = [
    "#8d9aac",
    "#2777df",
    "#9165cf",
    "#16a67e"
];


// Create live score bars
function showBars(id, names, scores, colors) {

    const holder = document.getElementById(id);
    holder.replaceChildren();

    names.forEach((name, i) => {

        const value = Math.max(
            0,
            Math.min(1, Number(scores[i]) || 0)
        );

        const row = document.createElement("div");
        row.className = "bar-row";

        const head = document.createElement("div");
        head.className = "bar-head";

        const label = document.createElement("span");
        label.textContent = name;

        const score = document.createElement("strong");
        score.textContent = value.toFixed(3);

        head.append(label, score);

        const track = document.createElement("div");
        track.className = "track";

        const fill = document.createElement("div");
        fill.className = "fill";

        fill.style.width = (value * 100) + "%";
        fill.style.background = colors[i];

        track.append(fill);
        row.append(head, track);
        holder.append(row);

    });
}


// Draw the live sound graph
function drawGraph(history, id="graph", colors=googleColors) {

    const canvas = document.getElementById(id);
    if (!canvas || !canvas.clientWidth) return;

    const w = canvas.getBoundingClientRect().width || 650;
    const h = 285;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const L = 42;
    const T = 15;
    const R = w - 13;
    const B = h - 26;

    ctx.font = "12px system-ui";
    ctx.lineWidth = 1;

    // Draw grid lines
    for (const value of [0, 0.25, 0.5, 0.75, 1]) {

        const y = B - value * (B - T);

        ctx.strokeStyle = "#e8eef5";

        ctx.beginPath();
        ctx.moveTo(L, y);
        ctx.lineTo(R, y);
        ctx.stroke();

        ctx.fillStyle = "#7b8ba2";
        ctx.fillText(value.toFixed(2), 3, y + 4);
    }

    if (!history || history.length < 2) {
        return;
    }

    // Draw three model-score lines
    colors.forEach((color, channel) => {

        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();

        history.forEach((scores, i) => {

            const x =
                L + (i / (history.length - 1)) * (R - L);

            const value = Math.max(
                0,
                Math.min(1, scores[channel])
            );

            const y = B - value * (B - T);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }

        });

        ctx.stroke();
    });
}


// Update the dashboard from the Raspberry Pi
async function update() {

    try {

        const response = await fetch(
            "/api/status",
            {cache: "no-store"}
        );

        if (!response.ok) {
            throw new Error(
                "Dashboard HTTP " + response.status
            );
        }

        const s = await response.json();

        document.getElementById("cameraStatus").textContent =
            s.camera_status || "Unavailable";

        document.getElementById("statusText").textContent =
            s.status;

        document.getElementById("status").className =
            "status " + (
                s.status === "Listening" ? "ok" :
                s.status === "Error" ? "bad" : ""
            );

        const message = document.getElementById("message");

        message.textContent = s.error || "";
        message.className = s.error ? "visible" : "";

        document.getElementById("topEvent").textContent =
            s.top || "—";

        document.getElementById("rfEvent").textContent =
            s.rf_name || "—";

        document.getElementById("latency").textContent =
            Number(s.yamnet_ms).toFixed(0);

        document.getElementById("device").textContent =
            s.device;

        document.getElementById("rms").textContent =
            Number(s.rms).toFixed(4);

        document.getElementById("rfMs").textContent =
            Number(s.rf_ms).toFixed(0) + " ms";

        showBars(
            "googleBars",
            googleNames,
            s.yamnet,
            googleColors
        );

        showBars(
            "rfBars",
            rfNames,
            s.rf,
            rfColors
        );

        drawGraph(s.history);

        const frames = s.status === "Listening" ? (s.paired || []) : [];
        const latest = frames[frames.length - 1];
        const predicted = rfNames.indexOf(s.rf_name);

        if (latest && predicted >= 0) {
            document.getElementById("rfEvent").textContent =
                s.rf_name + " · " + latest[1][predicted].toFixed(3);
        }

        drawGraph(frames.map(f => f[1]), "rfGraph", rfColors);

        const selector = document.getElementById("compareSound");
        if (selector) {
            const clap = selector.value === "clap";
            const yi = googleNames.indexOf(clap ? "Clapping" : "Whistling");
            const ri = rfNames.indexOf(clap ? "clap" : "whistle");

            const lines = frames.map(([yam, rf]) => [yam[yi], rf[ri]]);
            const current = lines[lines.length - 1];

            document.getElementById("pairY").textContent =
                current ? current[0].toFixed(3) : "—";
            document.getElementById("pairR").textContent =
                current ? current[1].toFixed(3) : "—";

            drawGraph(lines, "pairGraph", ["#2777df", "#9165cf"]);
        }

    } catch (error) {

        document.getElementById("statusText").textContent =
            "Disconnected";

        document.getElementById("status").className =
            "status bad";

        const message = document.getElementById("message");

        message.textContent = error.message;
        message.className = "visible";
    }
}


update();
setInterval(update, 1000);

window.addEventListener("resize", update);

</script>

</body>
</html>
"""


# -----------------------------------------
# 6. WEB SERVER
# -----------------------------------------

class Dashboard(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/":

            content = HOME.encode("utf-8")
            content_type = "text/html; charset=utf-8"

        elif self.path == "/api/status":

            with lock:
                snapshot = dict(state)

            with camera.lock:
                snapshot["camera_status"] = camera.status

            content = json.dumps(snapshot).encode("utf-8")
            content_type = "application/json; charset=utf-8"

        elif self.path == "/camera.mjpg":
            camera.stream(self)
            return

        else:
            self.send_error(404)
            return

        self.send_response(200)

        self.send_header(
            "Content-Type",
            content_type
        )

        self.send_header(
            "Cache-Control",
            "no-store"
        )

        self.send_header(
            "Content-Length",
            str(len(content))
        )

        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt, *args):
        pass


# -----------------------------------------
# 7. START DASHBOARD
# -----------------------------------------

def main():

    camera.start()

    thread = threading.Thread(
        target=listen,
        daemon=True
    )

    thread.start()

    server = ThreadingHTTPServer(
        ("127.0.0.1", 8765),
        Dashboard
    )

    print(
        "WHISPRYN dashboard: http://127.0.0.1:8765",
        flush=True
    )

    print(
        "Microphone device:",
        AUDIO_DEVICE,
        flush=True
    )

    print(
        "ALSA errors:",
        LOG_FILE,
        flush=True
    )

    print("Press Ctrl+C to stop.", flush=True)

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        pass

    finally:

        stop.set()
        camera.close()

        if recorder is not None:
            if recorder.poll() is None:
                recorder.terminate()

        server.server_close()
        thread.join(timeout=3)


if __name__ == "__main__":
    main()
