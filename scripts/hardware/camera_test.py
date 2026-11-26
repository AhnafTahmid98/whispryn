import subprocess
import shutil
from pathlib import Path

import cv2


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "runtime" / "camera"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATH = OUTPUT_DIR / "camera_test.jpg"
VIDEO_PATH = OUTPUT_DIR / "camera_test.mjpeg"
FRAME_PATH = OUTPUT_DIR / "opencv_frame.jpg"


def check_dependencies():
    print("Checking WHISPRYN camera dependencies...")

    for command in ("rpicam-still", "rpicam-vid"):
        if shutil.which(command) is None:
            raise RuntimeError(f"{command} not found")

    print("OpenCV version:", cv2.__version__)
    print("Camera commands available")


def capture_image():
    print("\nCapturing image...")

    subprocess.run(
        [
            "rpicam-still",
            "-n",
            "-o", str(IMAGE_PATH),
        ],
        check=True,
    )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError("OpenCV failed to read image")

    height, width = image.shape[:2]

    print("Image captured successfully")
    print(f"Resolution: {width} x {height}")

    return image


def capture_video():
    print("\nRecording 5-second video...")

    subprocess.run(
        [
            "rpicam-vid",
            "-n",
            "-t", "5000",
            "--width", "640",
            "--height", "480",
            "--framerate", "15",
            "--codec", "mjpeg",
            "-o", str(VIDEO_PATH),
        ],
        check=True,
    )

    print("Video recorded successfully")


def process_video():
    print("\nProcessing video with OpenCV...")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Cannot open recorded video")

    frame_count = 0
    saved = False

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            if frame_count == 10:
                cv2.putText(
                    frame,
                    "WHISPRYN CAMERA",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                saved = cv2.imwrite(
                    str(FRAME_PATH),
                    frame,
                )
    finally:
        cap.release()

    print("Frames decoded:", frame_count)

    if frame_count == 0 or not saved:
        raise RuntimeError("OpenCV frame processing failed")

    print("Frame saved:", FRAME_PATH)


def main():
    print("=" * 40)
    print("WHISPRYN CAMERA TEST")
    print("=" * 40)

    check_dependencies()
    capture_image()
    capture_video()
    process_video()

    print("\nWHISPRYN CAMERA TEST PASSED")


if __name__ == "__main__":
    main()
