"""
ocr.py
Loads your fine-tuned PaddleOCR model and runs inference on an image file.

Usage:
    from src.ocr import load_model, run_ocr
    model = load_model()
    text  = run_ocr(model, "data/sample_images/test1.jpg")
"""

import os


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


def run_ocr(model, image_path: str) -> str:
    """
    Run OCR on an image file and return all recognized text as one string.

    Args:
        model:      The model returned by load_model().
        image_path: Path to the image file (jpg, png, etc.)

    Returns:
        A single string with all recognized text, lines joined by newlines.
        Returns an empty string if nothing was recognized.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(input=image_path, batch_size=1)

    lines = []
    for result in results:
        try:
            res_data = result.get("res", result)
            text  = res_data.get("rec_text", "")
            score = res_data.get("rec_score", 0.0)
            if text.strip() and score > 0.3:
                lines.append(text.strip())
        except (KeyError, AttributeError, TypeError):
            try:
                text = str(result).strip()
                if text:
                    lines.append(text)
            except Exception:
                continue

    return "\n".join(lines)
