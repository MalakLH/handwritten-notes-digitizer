"""
ocr.py
Loads your fine-tuned PaddleOCR model and runs inference on an image file.

Usage:
    from src.ocr import load_model, run_ocr
    model = load_model()
    text  = run_ocr(model, "data/sample_images/test1.jpg")
"""

import cv2
import os
import numpy as np
from pathlib import Path


def load_model(model_dir: str = None, use_gpu: bool = False):
    """
    Load the PaddleOCR model once at startup.

    Args:
        model_dir: Path to folder containing inference.pdiparams etc.
                   Falls back to default PP-OCRv5 if not found.
        use_gpu:   True for CUDA GPU, False for CPU (default).

    Returns:
        A PaddleOCR TextRecognition model instance.
    """
    from paddleocr import TextRecognition

    device = "gpu" if use_gpu else "cpu"

    # Check that the expected inference files actually exist inside model_dir
    has_model = (
        model_dir
        and os.path.exists(os.path.join(model_dir, "inference.pdiparams"))
    )

    if has_model:
        print(f"[OCR] Loading fine-tuned model from: {model_dir}")
        model = TextRecognition(
            model_name="PP-OCRv5_mobile_rec",
            model_dir=model_dir,
            device=device
        )
    else:
        print("[OCR] Fine-tuned model not found — using default PP-OCRv5_mobile_rec.")
        model = TextRecognition(
            model_name="PP-OCRv5_mobile_rec",
            device=device
        )

    return model


def run_ocr(image_path: str, num_lines: int, model) -> str:
    """
    Segments an image into `num_lines` horizontal strips,
    runs the fine-tuned OCR model on each strip,
    and returns the concatenated raw text.

    Args:
        image_path:  Absolute path to the page image.
        num_lines:   Number of lines the user says the page contains.
        model:       The loaded fine-tuned OCR model instance.

    Returns:
        A single string with each predicted line separated by '\\n'.
    """

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    h = image.shape[0]
    strip_height = h // num_lines

    if strip_height == 0:
        raise ValueError(
            f"num_lines ({num_lines}) is larger than the image height ({h}px). "
            "Use a smaller value."
        )

    collected_lines: list[str] = []
    y = 0

    while y + strip_height <= h:
        strip = image[y : y + strip_height, :]
        predicted = model.predict(input=strip, batch_size=1)
        collected_lines.append(predicted[0]["rec_text"])
        y += strip_height

    # Handle the leftover bottom strip (when h % num_lines != 0)
    if y < h:
        leftover_strip = image[y:h, :]
        predicted = model.predict(input=leftover_strip, batch_size=1)
        collected_lines.append(predicted[0]["rec_text"])

    return "\n".join(collected_lines)
    