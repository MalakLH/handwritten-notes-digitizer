"""
ocr.py
──────
Wraps fine-tuned PaddleOCR model and handles the full recognition step.

This file does two things:
  1. Loads fine-tuned PP-OCRv5 model once at startup
  2. Runs inference and returns raw extracted text as a single string

The model handles full-page images by internally running:
  - A detection model: finds where the text lines are (bounding boxes)
  - A recognition model: reads each text line (this is what you fine-tuned)

Usage:
    from src.ocr import load_model, run_ocr
    model = load_model()
    raw_text = run_ocr(model, processed_image)
"""

import os
import numpy as np
from PIL import Image

# Model loading


def load_model(model_dir: str = None, use_gpu: bool = False):
    """
    Load the PaddleOCR model. Loads ONCE and is reused for all images.
    
    Args:
        model_dir: Path to your fine-tuned model directory.
                   or falls back to the default pretrained PP-OCRv5 model.
        use_gpu:   Set to True if running on a machine with a CUDA GPU.
                   For Hugging Face Spaces (CPU), keep this False.
    Returns:
        A PaddleOCR TextRecognition model instance.
    """
    from paddleocr import TextRecognition

    # The model_dir should contain the .pdparams file from your training run.
    if model_dir and os.path.exists(model_dir):
        print(f"[OCR] Loading fine-tuned model from: {model_dir}")
        model = TextRecognition(
            model_name="PP-OCRv5_mobile_rec",
            model_dir=model_dir,
            device="gpu" if use_gpu else "cpu"
        )
    else:
        # Fall back to the default pretrained model if no fine-tuned weights found
        print("[OCR] No fine-tuned model found. Using default PP-OCRv5_mobile_rec.")
        model = TextRecognition(
            model_name="PP-OCRv5_mobile_rec",
            device="gpu" if use_gpu else "cpu"
        )

    return model

# Inference

def run_ocr(model, image: np.ndarray) -> str:
    """
    Run OCR on an image and return all recognized text as one string.
    
    Args:
        model:  The PaddleOCR model returned by load_model().

    Returns:
        A single string with all recognized text, lines separated by newlines.
        Example: "clear old gradients from batch\nto prevent accumulation"
    
    Returns empty string if nothing was recognized (don't crash the pipeline).
    """

    # Run the model. Returns a list of result objects, one per detected line.
    results = model.predict(input=image, batch_size=1)

    # Extract just the text from each result and join into one string.
    # Each result object has a 'rec_text' field (the recognized string)
    # and 'rec_score' (confidence 0.0–1.0).
    lines = []
    for result in results:
        # result is a dict-like object; access it like result['res']
        # or use result.print() to see the full structure
        try:
            # New PaddleOCR API (v3.x) returns result as an object with .print()
            # The actual data is in result['res']
            res_data = result.get("res", result)  # handles both API styles
            text = res_data.get("rec_text", "")
            score = res_data.get("rec_score", 0.0)
            
            # Only include results with reasonable confidence (above 30%)
            # Very low confidence usually means the model saw a non-text region
            if text.strip() and score > 0.3:
                lines.append(text.strip())
        except (KeyError, AttributeError, TypeError):
            # If result format is unexpected, try direct string conversion
            try:
                text = str(result)
                if text.strip():
                    lines.append(text.strip())
            except Exception:
                continue

    return "\n".join(lines)
