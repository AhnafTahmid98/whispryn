
import argparse
import subprocess
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "runtime" / "camera"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=15)
    args = parser.parse_args()

    if args.fps <= 0 or args.seconds <= 0:
        parser.error("FPS and duration must be positive")

    if not shutil.which("rpicam-vid"):
        raise RuntimeError("rpicam-vid is not installed")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    command = [
        "rpicam-vid", "-n", "-t", "0",
        "--width", "640",
        "--height", "480",
        "--framerate", str(args.fps),
        "--codec", "mjpeg",
        "-o", "-"
    ]

    detector = cv2.createBackgroundSubtractorMOG2(
        history=150,
        varThreshold=32,
        detectShadows=True
    )

    frames = 0
    motion_frames = 0
    saved = False
    buffer = bytearray()

    print("WHISPRYN Motion Detection Starting...")

    start_time = time.monotonic()

    with (OUTPUT / "motion_camera.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=log,
            bufsize=0
        )

        try:
            while time.monotonic() - start_time < args.seconds:
                data = process.stdout.read(65536)

                if not data:
                    break

                buffer.extend(data)

                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        buffer.clear()
                        break

                    if start:
                        del buffer[:start]

                    end = buffer.find(b"\xff\xd9", 2)
                    if end < 0:
                        break

                    jpeg = bytes(buffer[:end + 2])
                    del buffer[:end + 2]

                    frame = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )

                    if frame is None:
                        continue

                    frames += 1
                    elapsed = time.monotonic() - start_time

                    mask = detector.apply(frame)

                    # Ignore shadows and low-confidence changes.
                    _, mask = cv2.threshold(
                        mask, 200, 255, cv2.THRESH_BINARY
                    )

                    mask = cv2.morphologyEx(
                        mask,
                        cv2.MORPH_OPEN,
                        np.ones((3, 3), np.uint8)
                    )

                    contours, _ = cv2.findContours(
                        mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    detected = False

                    # Let the camera learn the background first.
                    if elapsed > 3:
                        for contour in contours:
                            if cv2.contourArea(contour) < 1200:
                                continue

                            x, y, w, h = cv2.boundingRect(contour)

                            cv2.rectangle(
                                frame,
                                (x, y),
                                (x + w, y + h),
                                (0, 255, 0),
                                2
                            )
                            detected = True

                    if detected:
                        motion_frames += 1

                        cv2.putText(
                            frame,
                            "WHISPRYN - MOTION DETECTED",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2
                        )

                        if not saved:
                            saved = cv2.imwrite(
                                str(OUTPUT / "motion_detected.jpg"),
                                frame
                            )

                    if len(buffer) > 4 * 1024 * 1024:
                        buffer.clear()

        except KeyboardInterrupt:
            print("Stopped by user")

        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()

    elapsed = time.monotonic() - start_time

    print("\nWHISPRYN MOTION TEST")
    print("Frames decoded:", frames)
    print("Motion frames:", motion_frames)
    print("Overall FPS:", round(frames / elapsed, 2))
    print("Motion image saved:", saved)


if __name__ == "__main__":
    main()
