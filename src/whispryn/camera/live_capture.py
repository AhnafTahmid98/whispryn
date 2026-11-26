import argparse
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "runtime" / "camera"


def main():
    parser = argparse.ArgumentParser(description="WHISPRYN live camera FPS test")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--display", action="store_true", help="Show a window (local desktop only)")
    args = parser.parse_args()

    if args.fps <= 0 or args.width <= 0 or args.height <= 0 or args.seconds <= 0:
        parser.error("fps, width, height and seconds must be positive")
    if shutil.which("rpicam-vid") is None:
        raise SystemExit("rpicam-vid not found: install the Pi-compatible camera stack")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = OUTPUT_DIR / "live_opencv_frame.jpg"
    log_path = OUTPUT_DIR / "live_camera.log"
    command = [
        "rpicam-vid", "-n", "-t", "0",
        "--width", str(args.width), "--height", str(args.height),
        "--framerate", str(args.fps), "--codec", "mjpeg", "-o", "-",
    ]
    print(f"WHISPRYN: requesting {args.width}x{args.height} @ {args.fps} FPS")
    print("Live processing over SSH; no display window unless --display is used")

    frames = 0
    first_frame_time = None
    last_frame_time = None
    last_report = time.monotonic()
    saved = False
    buffer = bytearray()

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=log, bufsize=0
        )
        try:
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                data = process.stdout.read(65536)
                if not data:
                    break
                buffer.extend(data)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start == -1:
                        buffer.clear()
                        break
                    if start:
                        del buffer[:start]
                    end = buffer.find(b"\xff\xd9", 2)
                    if end == -1:
                        break
                    jpg = bytes(buffer[:end + 2])
                    del buffer[:end + 2]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    now = time.monotonic()
                    frames += 1
                    if first_frame_time is None:
                        first_frame_time = now
                    last_frame_time = now
                    fps = ((frames - 1) / (last_frame_time - first_frame_time)
                           if frames > 1 and last_frame_time > first_frame_time else 0.0)

                    cv2.putText(frame, f"WHISPRYN LIVE  {fps:.1f} FPS", (12, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    if not saved and frames >= 10:
                        saved = cv2.imwrite(str(image_path), frame)
                        print(f"Saved sample image: {image_path} (success={saved})")

                    if args.display:
                        cv2.imshow("WHISPRYN Live Camera", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return
                    if now - last_report >= 3:
                        print(f"Decoded: {frames} frames; average live FPS: {fps:.1f}")
                        last_report = now
                if len(buffer) > 4 * 1024 * 1024:
                    buffer.clear()  # Recover if a corrupt/incomplete JPEG grows excessively.
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
            if args.display:
                cv2.destroyAllWindows()

    average = ((frames - 1) / (last_frame_time - first_frame_time)
               if frames > 1 and last_frame_time > first_frame_time else 0.0)
    print(f"Frames decoded: {frames}")
    print(f"Average live decode/processing FPS: {average:.2f}")
    print(f"Camera log: {log_path}")
    if frames == 0:
        raise SystemExit("No frames received; inspect the camera log")


if __name__ == "__main__":
    main()
