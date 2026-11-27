
"""WHISPRYN live sound dashboard."""

import html
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import joblib
import librosa
import numpy as np
from ai_edge_litert.interpreter import Interpreter


# Our project files
ROOT = Path(__file__).resolve().parents[2]

RATE = 16000
DEVICE = "plughw:2,0"
CHUNK = RATE // 2

YAMNET_DIR = ROOT / "runtime/models/yamnet"
RF_FILE = ROOT / "runtime/training/whispryn_rf_v2.joblib"

GOOGLE_SOUNDS = ["Clapping", "Whistling", "Speech"]
RF_SOUNDS = ["background", "clap", "hey", "whistle"]

COLORS = ["#58a6ff", "#24cba2", "#eeb05b"]

state = {
    "status": "Starting",
    "yamnet": [0.0] * 3,
    "rf": [0.0] * 4,
    "rf_name": "Waiting for audio",
    "top": "",
    "history": [],
    "time_ms": 0.0,
    "rms": 0.0
}

lock = threading.Lock()
stop = threading.Event()
recorder = None


# Listen to the microphone and run both models.
def listen():
    global recorder

    try:
        # Load Google's YAMNet
        model = Interpreter(
            model_path=str(YAMNET_DIR / "yamnet.tflite"),
            num_threads=2
        )
        model.allocate_tensors()

        labels = (
            YAMNET_DIR / "yamnet_label_list.txt"
        ).read_text().splitlines()

        indexes = [
            labels.index(name)
            for name in GOOGLE_SOUNDS
        ]

        input_info = model.get_input_details()[0]
        output_info = model.get_output_details()[0]

        window = int(np.prod(input_info["shape"]))

        if window != 15600:
            raise RuntimeError("Unexpected YAMNet input size")

        # Load our own trained Random Forest
        rf = joblib.load(RF_FILE)["model"]

        # Start recording from ReSpeaker
        recorder = subprocess.Popen(
            [
                "arecord", "-q",
                "-D", DEVICE,
                "-t", "raw",
                "-f", "S16_LE",
                "-r", str(RATE),
                "-c", "1"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        audio = np.empty(0, dtype=np.float32)
        history = deque(maxlen=50)

        with lock:
            state["status"] = "Listening"

        while not stop.is_set():
            # Read half a second of audio
            data = recorder.stdout.read(CHUNK * 2)

            if not data:
                raise RuntimeError("Microphone stopped")

            part = np.frombuffer(
                data, dtype="<i2"
            ).astype(np.float32) / 32768

            audio = np.concatenate([
                audio, part
            ])[-3 * RATE:]

            if len(audio) < window:
                continue

            # Google model: last 0.975 seconds
            model.set_tensor(
                input_info["index"],
                audio[-window:]
            )

            start = time.perf_counter()
            model.invoke()

            inference_ms = (
                time.perf_counter() - start
            ) * 1000

            raw_scores = model.get_tensor(
                output_info["index"]
            ).ravel()

            google_scores = [
                float(raw_scores[i])
                for i in indexes
            ]

            top = labels[int(np.argmax(raw_scores))]
            history.append(google_scores)

            # Random Forest: last 3 seconds
            rf_scores = [0.0] * 4
            rf_name = "Waiting for 3 seconds of audio"

            if len(audio) >= 3 * RATE:
                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=RATE,
                    n_mfcc=13
                )

                features = np.concatenate([
                    mfcc.mean(axis=1),
                    mfcc.std(axis=1)
                ])

                probabilities = rf.predict_proba(
                    features.reshape(1, -1)
                )[0]

                by_name = dict(
                    zip(rf.classes_, probabilities)
                )

                rf_scores = [
                    float(by_name.get(name, 0))
                    for name in RF_SOUNDS
                ]

                rf_name = str(
                    rf.classes_[int(np.argmax(probabilities))]
                )

            with lock:
                state.update(
                    status="Listening",
                    yamnet=google_scores,
                    rf=rf_scores,
                    rf_name=rf_name,
                    top=top,
                    history=list(history),
                    time_ms=inference_ms,
                    rms=float(
                        np.sqrt(np.mean(part ** 2))
                    )
                )

    except Exception as error:
        with lock:
            state["status"] = "Error: " + str(error)

    finally:
        if recorder is not None:
            if recorder.poll() is None:
                recorder.terminate()
            recorder.wait()


# Make score bars for the dashboard
def bars(names, scores, colors):
    result = ""

    for name, value, color in zip(names, scores, colors):
        value = max(0, min(1, value))

        result += (
            f'<p>{html.escape(name)}: '
            f'<b>{value:.2f}</b></p>'
            f'<div class="track">'
            f'<div class="fill" '
            f'style="width:{value * 100:.1f}%;'
            f'background:{color}"></div></div>'
        )

    return result


# Draw our live graph using SVG
def chart(history):
    lines = ""

    for sound, color in enumerate(COLORS):
        points = []

        for i, row in enumerate(history):
            x = 35 + (
                i / max(1, len(history) - 1)
            ) * 660

            y = 210 - max(
                0, min(1, row[sound])
            ) * 185

            points.append(f"{x:.1f},{y:.1f}")

        if points:
            lines += (
                f'<polyline points="{" ".join(points)}" '
                f'fill="none" stroke="{color}" '
                f'stroke-width="3"/>'
            )

    return (
        '<svg viewBox="0 0 720 240" width="100%" '
        'role="img" aria-label="Live sound graph">'
        '<path d="M35 25V210H700" fill="none" '
        'stroke="#9caebb"/>'
        '<path d="M35 117H700" stroke="#435366" '
        'stroke-dasharray="5 5"/>'
        '<text x="3" y="30" fill="#becbda">1.0</text>'
        '<text x="3" y="120" fill="#becbda">0.5</text>'
        '<text x="3" y="214" fill="#becbda">0.0</text>'
        f'{lines}</svg>'
    )


# Browser page
HOME = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>WHISPRYN Live Demo</title>

<style>
body {
    font: 17px system-ui;
    background: #101823;
    color: #eff4fa;
    margin: 0;
}
main {
    max-width: 950px;
    margin: auto;
    padding: 20px;
}
.card {
    background: #1d2b39;
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
}
.track {
    background: #465363;
    height: 13px;
    border-radius: 8px;
    overflow: hidden;
}
.fill { height: 100%; }
p { margin: 12px 0 5px; }
small { color: #aec0cf; }
</style>
</head>

<body>
<main>
<h1>WHISPRYN — Live Sound Demo</h1>
<small>Raspberry Pi 5 · ReSpeaker microphone</small>

<div id="live">Connecting...</div>

<small>
93.8% was measured offline, not live.
Servos are off. Audio is not saved.
</small>
</main>

<script>
async function update() {
    try {
        const response = await fetch('/live', {
            cache: 'no-store'
        });

        if (!response.ok) {
            throw new Error('HTTP ' + response.status);
        }

        document.getElementById('live').innerHTML =
            await response.text();

    } catch (error) {
        document.getElementById('live').textContent =
            error.message;
    }
}

update();
setInterval(update, 1000);
</script>
</body>
</html>
"""


# Build the part of the page that changes
def live_html(data):
    google = bars(
        GOOGLE_SOUNDS,
        data["yamnet"],
        COLORS
    )

    own = bars(
        RF_SOUNDS,
        data["rf"],
        ["#8298ac"] * 4
    )

    return f"""
    <div class="card">
        <b>{html.escape(data['status'])}</b>
        <p>Top Google category:
           <b>{html.escape(data['top'])}</b></p>
        <small>
        Speech detection is not exact "Hey" recognition.
        </small>
    </div>

    <div class="card">
        <h2>Google YAMNet Scores</h2>
        {google}
    </div>

    <div class="card">
        <h2>Live Sound Graph</h2>
        {chart(data['history'])}
        <small>
        Blue: Clapping |
        Green: Whistling |
        Orange: Speech
        </small>
    </div>

    <div class="card">
        <h2>WHISPRYN Random Forest v2</h2>
        <p>Prediction:
        <b>{html.escape(data['rf_name'])}</b></p>
        {own}
        <small>
        Uses rolling 3-second audio.
        Live accuracy is not yet validated.
        </small>
    </div>

    <div class="card">
        Audio RMS: {data['rms']:.4f} |
        YAMNet inference: {data['time_ms']:.0f} ms
    </div>
    """


# Send the dashboard to the browser
class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            content = HOME.encode("utf-8")

        elif self.path == "/live":
            with lock:
                current = dict(state)

            content = live_html(current).encode("utf-8")

        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8"
        )
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt, *args):
        pass


# Start microphone processing and the web dashboard
threading.Thread(
    target=listen,
    daemon=True
).start()

server = HTTPServer(("127.0.0.1", 8765), Page)

print("WHISPRYN dashboard started")
print("Local address: http://127.0.0.1:8765")
print("Press Ctrl+C to stop")

try:
    server.serve_forever()

except KeyboardInterrupt:
    pass

finally:
    stop.set()

    if recorder is not None and recorder.poll() is None:
        recorder.terminate()

    server.server_close()
