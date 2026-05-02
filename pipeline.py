"""
pipeline.py
───────────
The main script that chains all four steps together.

This is the heart of the project. It:
  1. Accepts an image (from command line, or called by app.py)
  2. Runs preprocessing → OCR → postprocessing → Notion export
  3. Returns the Notion page URL

You can also run individual steps in isolation for debugging.

──────────────────────────────────────────────────────────────────────────────
ARCHITECTURE OVERVIEW
──────────────────────────────────────────────────────────────────────────────

  📷 Photo of notes
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │  src/preprocessing.py                               │
  │                                                     │
  │  grayscale → CLAHE contrast → Otsu binarize         │
  │  → deskew → noise removal                          │
  │                                                     │
  │  Input:  PIL Image or file path                     │
  │  Output: numpy array (clean B&W image)              │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │  src/ocr.py                                         │
  │                                                     │
  │  Fine-tuned PP-OCRv5_mobile_rec                     │
  │  Detects text regions → reads each line             │
  │                                                     │
  │  Input:  numpy array                                │
  │  Output: raw string ("Ican ocR engne...")           │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │  src/postprocessing.py                              │
  │                                                     │
  │  Groq (Llama 3.1) or Gemini (Flash)                 │
  │  OCR error correction — no new content added        │
  │                                                     │
  │  Input:  raw OCR string                             │
  │  Output: clean string ("I can OCR engine...")       │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │  src/notion_export.py                               │
  │                                                     │
  │  Parses text structure (headings, bullets)          │
  │  Creates formatted Notion page via API              │
  │                                                     │
  │  Input:  clean string + Notion credentials          │
  │  Output: Notion page URL                            │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  📝 Notion page at https://notion.so/...

──────────────────────────────────────────────────────────────────────────────

Usage from command line:
    python pipeline.py --image path/to/photo.jpg

Usage from Python (e.g., called by app.py):
    from pipeline import run_pipeline, load_ocr_model
    model = load_ocr_model()                   # load once at startup
    url, clean_text = run_pipeline(image, model)  # call per image
"""

import argparse
import os
from dotenv import load_dotenv

from src.preprocessing import preprocess_image
from src.ocr import load_model, run_ocr
from src.postprocessing import clean_text
from src.notion_export import create_notion_page

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (called ONCE at app startup, not per image)
# ─────────────────────────────────────────────────────────────────────────────

def load_ocr_model():
    """
    Load the fine-tuned OCR model. Call this once when your app starts.
    
    Why separate from run_pipeline? Because if you call run_pipeline() for 10
    images in a row, you don't want to reload the model 10 times. Load it once,
    pass it into run_pipeline() each time.
    
    Returns:
        The loaded PaddleOCR model object.
    """
    model_dir = os.getenv("MODEL_DIR", "models/")
    use_gpu   = os.getenv("USE_GPU", "false").lower() == "true"
    return load_model(model_dir=model_dir, use_gpu=use_gpu)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    image_input,
    ocr_model,
    llm_provider: str = "groq",
    notion_token: str = None,
    notion_page_id: str = None,
    export_to_notion: bool = True
) -> tuple:
    """
    Run the full pipeline on one image.
    
    Args:
        image_input:      File path (str) or PIL Image object.
        ocr_model:        The model returned by load_ocr_model().
        llm_provider:     "groq" or "gemini".
        notion_token:     Notion integration token (or set NOTION_TOKEN in .env).
        notion_page_id:   Notion parent page ID (or set NOTION_PARENT_PAGE_ID).
        export_to_notion: Set to False to skip Notion and just return the text.
                          Useful for testing the OCR/cleanup steps in isolation.
    
    Returns:
        (notion_url, clean_text_string) tuple.
        notion_url is None if export_to_notion=False.
    
    Example:
        model = load_ocr_model()
        url, text = run_pipeline("photo.jpg", model)
        print(text)   # "Tesseract: I can OCR engine that uses..."
        print(url)    # "https://www.notion.so/abc123..."
    """
    print("\n[Pipeline] Starting...")

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    print("[Pipeline] Step 1/4: Preprocessing image...")
    processed_image = preprocess_image(image_input)
    print("[Pipeline] ✓ Preprocessing done")

    # ── Step 2: OCR ───────────────────────────────────────────────────────────
    print("[Pipeline] Step 2/4: Running OCR...")
    raw_text = run_ocr(ocr_model, processed_image)
    
    if not raw_text.strip():
        print("[Pipeline] ⚠ OCR returned no text. Check image quality.")
        return None, ""
    
    print(f"[Pipeline] ✓ OCR done. Raw text preview: {raw_text[:100]}...")

    # ── Step 3: Postprocess ───────────────────────────────────────────────────
    print(f"[Pipeline] Step 3/4: Cleaning text with {llm_provider}...")
    cleaned = clean_text(raw_text, provider=llm_provider)
    print(f"[Pipeline] ✓ Cleanup done. Clean text preview: {cleaned[:100]}...")

    # ── Step 4: Notion export ─────────────────────────────────────────────────
    if not export_to_notion:
        print("[Pipeline] Step 4/4: Skipping Notion export (export_to_notion=False)")
        return None, cleaned

    print("[Pipeline] Step 4/4: Exporting to Notion...")
    page_url = create_notion_page(
        clean_text=cleaned,
        notion_token=notion_token,
        parent_page_id=notion_page_id
    )
    print(f"[Pipeline] ✓ Done! Page: {page_url}")

    return page_url, cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Digitize a handwritten note and send it to Notion."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the image file (jpg, png)"
    )
    parser.add_argument(
        "--provider", default="groq", choices=["groq", "gemini"],
        help="LLM provider for text cleanup (default: groq)"
    )
    parser.add_argument(
        "--no-notion", action="store_true",
        help="Skip Notion export. Just print the cleaned text."
    )
    args = parser.parse_args()

    # Load model once
    print("[Pipeline] Loading OCR model (this takes a few seconds)...")
    model = load_ocr_model()

    # Run pipeline
    url, text = run_pipeline(
        image_input=args.image,
        ocr_model=model,
        llm_provider=args.provider,
        export_to_notion=not args.no_notion
    )

    print("\n" + "─" * 60)
    print("CLEANED TEXT:")
    print("─" * 60)
    print(text)
    if url:
        print("\n" + "─" * 60)
        print(f"NOTION PAGE: {url}")
