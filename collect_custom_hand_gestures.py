"""
Collect a custom hand-gesture dataset with a webcam.

Recommended use on native Windows Python/OpenCV if WSL2 cannot access the webcam.

Examples
--------
python collect_custom_hand_gestures.py --class-name open_palm --target 200 --output data/custom_hand_gestures
python collect_custom_hand_gestures.py --class-name fist --target 200 --output data/custom_hand_gestures
python collect_custom_hand_gestures.py --class-name thumbs_up --target 200 --output data/custom_hand_gestures

Controls
--------
SPACE : start/stop automatic capture
c     : capture one image manually
q     : quit
"""

import argparse
import os
import time
from pathlib import Path

import cv2


VALID_CLASSES = ["open_palm", "fist", "thumbs_up"]


def next_image_index(class_dir: Path) -> int:
    existing = sorted(class_dir.glob("*.jpg"))
    if not existing:
        return 0

    indices = []
    for path in existing:
        stem = path.stem
        if "_" in stem:
            try:
                indices.append(int(stem.split("_")[-1]))
            except ValueError:
                pass

    return max(indices) + 1 if indices else len(existing)


def open_camera(camera_id: int):
    # DirectShow is usually more reliable for webcams on native Windows.
    if os.name == "nt":
        return cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    return cv2.VideoCapture(camera_id)


def collect_images_for_class(
    class_name: str,
    output_dir: Path,
    target_new_images: int = 200,
    camera_id: int = 0,
    roi=(160, 80, 320, 320),
    image_size=(128, 128),
    auto_capture_interval: float = 0.15,
):
    if class_name not in VALID_CLASSES:
        raise ValueError(f"class_name must be one of: {VALID_CLASSES}")

    class_dir = output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(camera_id)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Try another --camera value or check camera permissions."
        )

    x, y, w, h = roi
    img_index = next_image_index(class_dir)
    captured = 0
    auto_capture = False
    last_capture_time = 0.0

    print(f"Collecting class: {class_name}")
    print(f"Saving to: {class_dir.resolve()}")
    print("Controls: SPACE=start/stop auto capture, c=capture once, q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)

        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(x, frame_w - 1))
        y1 = max(0, min(y, frame_h - 1))
        x2 = max(1, min(x + w, frame_w))
        y2 = max(1, min(y + h, frame_h))

        roi_img = frame[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi_img, image_size)

        now = time.time()
        should_capture = auto_capture and (now - last_capture_time >= auto_capture_interval)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            auto_capture = not auto_capture
            print("Auto capture:", "ON" if auto_capture else "OFF")
        if key == ord("c"):
            should_capture = True

        if should_capture and captured < target_new_images:
            filename = class_dir / f"{class_name}_{img_index:06d}.jpg"
            cv2.imwrite(str(filename), roi_resized)
            img_index += 1
            captured += 1
            last_capture_time = now

        if captured >= target_new_images:
            auto_capture = False

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        status = "AUTO ON" if auto_capture else "AUTO OFF"
        cv2.putText(
            frame,
            f"Class: {class_name} | Captured: {captured}/{target_new_images}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"{status} | SPACE auto | c once | q quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Collect custom hand gesture dataset", frame)

    cap.release()
    cv2.destroyAllWindows()

    print(f"Finished. New images captured for {class_name}: {captured}")
    print(f"Class folder: {class_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect hand gesture images with a webcam.")
    parser.add_argument("--class-name", required=True, choices=VALID_CLASSES)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("../data/custom_hand_gestures"))
    parser.add_argument("--roi", type=int, nargs=4, default=[160, 80, 320, 320], metavar=("X", "Y", "W", "H"))
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--interval", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    collect_images_for_class(
        class_name=args.class_name,
        output_dir=args.output,
        target_new_images=args.target,
        camera_id=args.camera,
        roi=tuple(args.roi),
        image_size=tuple(args.image_size),
        auto_capture_interval=args.interval,
    )


if __name__ == "__main__":
    main()
