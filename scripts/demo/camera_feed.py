
"""Live camera feed for the WHISPRYN dashboard."""

import subprocess
import threading
import time
from pathlib import Path


class CameraFeed:

    def __init__(self, stop_event, log_path):
        self.stop_event = stop_event
        self.log_path = Path(log_path)

        self.lock = threading.Lock()
        self.frame = None
        self.frame_number = 0
        self.status = "Starting"

        self.process = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(
            target=self.capture,
            daemon=True
        )
        self.thread.start()

    def capture(self):
        # Send MJPEG camera frames through stdout.
        command = [
            "rpicam-vid",
            "-n",
            "-t", "0",
            "--width", "640",
            "--height", "360",
            "--framerate", "10",
            "--codec", "mjpeg",
            "--quality", "65",
            "-o", "-"
        ]

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        try:
            with self.log_path.open("wb") as logfile:

                self.process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=logfile,
                    bufsize=0
                )

                buffer = bytearray()

                while not self.stop_event.is_set():
                    data = self.process.stdout.read(16384)

                    if not data:
                        details = self.log_path.read_text(
                            errors="replace"
                        )[-400:]

                        raise RuntimeError(
                            details or "Camera stream stopped"
                        )

                    buffer.extend(data)

                    # Find complete JPEG images.
                    while True:
                        start = buffer.find(b"\xff\xd8")

                        if start < 0:
                            tail = buffer[-1:]
                            buffer.clear()

                            if tail == b"\xff":
                                buffer.extend(tail)

                            break

                        if start:
                            del buffer[:start]

                        end = buffer.find(b"\xff\xd9", 2)

                        if end < 0:
                            break

                        jpeg = bytes(buffer[:end + 2])
                        del buffer[:end + 2]

                        # Store the most recent image.
                        with self.lock:
                            self.frame = jpeg
                            self.frame_number += 1
                            self.status = "Live"

                    if len(buffer) > 4 * 1024 * 1024:
                        buffer.clear()

        except Exception as error:
            with self.lock:
                self.status = "Error: " + str(error)

            print("Camera error:", error, flush=True)

        finally:
            if self.process is not None:

                if self.process.poll() is None:
                    self.process.terminate()

                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

                if self.process.stdout:
                    self.process.stdout.close()

    def stream(self, request):
        """Show live MJPEG frames in the browser."""

        request.send_response(200)

        request.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )

        request.send_header(
            "Cache-Control", "no-store"
        )

        request.end_headers()

        last_sent = -1

        try:
            while not self.stop_event.is_set():

                with self.lock:
                    jpeg = self.frame
                    number = self.frame_number
                    status = self.status

                if status.startswith("Error:"):
                    break

                if jpeg is None or number == last_sent:
                    time.sleep(0.08)
                    continue

                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                )

                request.wfile.write(
                    header + jpeg + b"\r\n"
                )

                request.wfile.flush()
                last_sent = number

        except (
            BrokenPipeError,
            ConnectionResetError,
            OSError
        ):
            pass

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()

        if self.thread is not None:
            self.thread.join(timeout=3)
