"""
live_detection.py

Windows-side live webcam inference script for the AI204 DCNN hand gesture project.

Use this script in a Windows Python environment, not WSL, because the webcam is
normally available directly to Windows/OpenCV.

Expected trained model output:
    models/gesture_custom_cnn.keras

Expected class names file, optional but recommended:
    models/class_names.json

Example commands from the Windows project folder:
    python live_detection.py --model models\gesture_custom_cnn.keras --class-names models\class_names.json
    python live_detection.py --model models\gesture_custom_cnn.keras --dataset data\custom_hand_gestures

Controls:
    q  : quit
    s  : save current ROI frame to debug_capture.jpg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf


DEFAULT_CLASSES = ["fist", "open_palm", "thumbs_up"]


def parse_roi(value: str) -> Tuple[int, int, int, int]:
    """Parse ROI string in the form x,y,w,h."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must have the format x,y,w,h")
    try:
        x, y, w, h = map(int, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive")
    return x, y, w, h


def load_class_names(class_names_path: Path | None, dataset_dir: Path | None) -> List[str]:
    """Load class names from JSON, dataset folders, or fallback defaults."""
    if class_names_path is not None and class_names_path.exists():
        with class_names_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [str(x) for x in data]

        if isinstance(data, dict):
            # Supported formats:
            # {"class_names": ["fist", "open_palm", "thumbs_up"]}
            # {"0": "fist", "1": "open_palm", "2": "thumbs_up"}
            # {"fist": 0, "open_palm": 1, "thumbs_up": 2}
            if "class_names" in data and isinstance(data["class_names"], list):
                return [str(x) for x in data["class_names"]]

            if all(str(k).isdigit() for k in data.keys()):
                return [str(data[str(i)]) for i in sorted(map(int, data.keys()))]

            if all(isinstance(v, int) for v in data.values()):
                return [name for name, _ in sorted(data.items(), key=lambda item: item[1])]

        raise ValueError(
            f"Unsupported class names JSON format in: {class_names_path}"
        )

    if dataset_dir is not None and dataset_dir.exists():
        folders = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
        if folders:
            return folders

    print("Warning: class names file/dataset not found. Using default class order:")
    print(DEFAULT_CLASSES)
    return DEFAULT_CLASSES.copy()


def preprocess_roi(roi_bgr: np.ndarray, image_size: int) -> np.ndarray:
    """Convert OpenCV BGR ROI to model-ready RGB batch."""
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_rgb = cv2.resize(roi_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    roi_rgb = roi_rgb.astype("float32") / 255.0
    return np.expand_dims(roi_rgb, axis=0)


def put_text_with_background(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.7,
    thickness: int = 2,
) -> None:
    """Draw readable text with a dark background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(
        image,
        (x - 5, y - text_h - baseline - 5),
        (x + text_w + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, origin, font, font_scale, (255, 255, 255), thickness)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live webcam inference for hand gesture recognition."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/hand_gesture_dcnn_custom_final.keras"),
        help="Path to the trained Keras model file.",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=Path("models/hand_gesture_dcnn_custom_class_names.json"),
        help="Optional JSON file containing class names in training order.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset folder. Used to infer class names if JSON is missing.",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera index. Usually 0 for the default webcam.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Input image size used during training.",
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=(180, 80, 360, 360),
        help="Region of interest as x,y,w,h. Example: 220,120,260,260",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Horizontally flip webcam frames for mirror-like display.",
    )
    parser.add_argument(
        "--use-directshow",
        action="store_true",
        help="Use cv2.CAP_DSHOW on Windows. Helpful if webcam opens slowly or fails.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model}\n"
            "Copy the trained model from WSL to your Windows project models folder, "
            "or pass the correct path with --model."
        )

    print(f"Loading model from: {args.model}")
    model = tf.keras.models.load_model(args.model)

    class_names = load_class_names(args.class_names, args.dataset)
    print("Class names:", class_names)

    if len(class_names) != int(model.output_shape[-1]):
        print(
            "Warning: number of class names does not match model output size.\n"
            f"Class names: {len(class_names)}, model outputs: {model.output_shape[-1]}"
        )

    backend = cv2.CAP_DSHOW if args.use_directshow else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera_id, backend)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Try one of these:\n"
            "  1. Use --camera-id 1 or --camera-id 2\n"
            "  2. Use --use-directshow on Windows\n"
            "  3. Check Windows camera privacy permissions\n"
            "  4. Close other apps using the camera"
        )

    # Try to request a reasonable webcam resolution.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    x, y, w, h = args.roi
    last_time = time.time()
    fps = 0.0

    print("Starting live detection. Press 'q' to quit, 's' to save ROI debug image.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from webcam.")
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        frame_h, frame_w = frame.shape[:2]

        # Keep ROI inside frame bounds.
        x1 = max(0, min(x, frame_w - 1))
        y1 = max(0, min(y, frame_h - 1))
        x2 = max(0, min(x + w, frame_w))
        y2 = max(0, min(y + h, frame_h))

        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            batch = preprocess_roi(roi, args.image_size)
            probs = model.predict(batch, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            if pred_idx < len(class_names):
                pred_label = class_names[pred_idx]
            else:
                pred_label = f"class_{pred_idx}"

            # Draw ROI and prediction.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            put_text_with_background(
                frame,
                f"Prediction: {pred_label} ({confidence:.2%})",
                (20, 40),
            )

            # Draw top probabilities.
            sorted_indices = np.argsort(probs)[::-1]
            for rank, idx in enumerate(sorted_indices[: min(3, len(sorted_indices))]):
                label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
                text = f"{rank + 1}. {label}: {probs[idx]:.2%}"
                put_text_with_background(frame, text, (20, 80 + rank * 35), 0.6, 1)

        # FPS display.
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
        last_time = now
        put_text_with_background(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 25), 0.6, 1)

        cv2.imshow("AI204 Hand Gesture Live Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s") and roi.size > 0:
            cv2.imwrite("debug_capture.jpg", roi)
            print("Saved ROI to debug_capture.jpg")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()