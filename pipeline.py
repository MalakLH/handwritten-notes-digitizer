"""
pipeline.py
───────────
Chains all steps together: OCR → postprocessing → Notion export.

Usage from command line:
    python pipeline.py --image data/sample_images/test1.jpg

Usage from Python (called by app.py):
    from pipeline import load_ocr_model, run_pipeline
    model = load_ocr_model()
    url, text = run_pipeline("data/sample_images/test1.jpg", model)
"""

import argparse
import os
from dotenv import load_dotenv

from src.ocr import load_model, run_ocr
from src.postprocessing import clean_text
from src.notion_export import create_notion_page

load_dotenv()


def load_ocr_model():
    """
    Load the fine-tuned OCR model. Call this ONCE at app startup, not per image.

    Returns:
        The loaded PaddleOCR model object.
    """
    model_dir = os.getenv("MODEL_DIR", "models/")
    use_gpu   = os.getenv("USE_GPU", "false").lower() == "true"
    return load_model(model_dir=model_dir, use_gpu=use_gpu)


def run_pipeline(
    num_lines: int,
    image_path: str,
    ocr_model,
    llm_provider: str = "groq",
    notion_token: str = None,
    notion_page_id: str = None,
    export_to_notion: bool = True
) -> tuple:
    """
    Run the full pipeline on one image file.

    Args:
        image_path:       Path to the image file (jpg, png, etc.)
        ocr_model:        Model returned by load_ocr_model().
        llm_provider:     "groq" or "gemini".
        notion_token:     Notion integration token (or set NOTION_TOKEN in .env).
        notion_page_id:   Notion parent page ID (or set NOTION_PARENT_PAGE_ID in .env).
        export_to_notion: Set False to skip Notion and just return the text.

    Returns:
        (notion_url, clean_text) tuple.
        notion_url is None if export_to_notion=False.
    """
    print("\n[Pipeline] Starting...")

    # ── Step 1: OCR ───────────────────────────────────────────────────────────
    print("[Pipeline] Step 1/3: Running OCR...")
    raw_text = run_ocr(image_path, num_lines, model=ocr_model)  # Adjust num_lines as needed

    if not raw_text.strip():
        print("[Pipeline] ⚠ OCR returned no text. Check the image.")
        return None, ""

    print(f"[Pipeline] ✓ OCR done. Preview: {raw_text[:80]}...")

    # ── Step 2: Postprocess ───────────────────────────────────────────────────
    print(f"[Pipeline] Step 2/3: Cleaning text with {llm_provider}...")
    cleaned = clean_text(raw_text, provider=llm_provider)
    print(f"[Pipeline] ✓ Cleanup done. Preview: {cleaned[:80]}...")

    # ── Step 3: Notion export ─────────────────────────────────────────────────
    if not export_to_notion:
        print("[Pipeline] Step 3/3: Skipping Notion export.")
        return None, cleaned

    print("[Pipeline] Step 3/3: Exporting to Notion...")
    page_url = create_notion_page(
        clean_text=cleaned,
        notion_token=notion_token,
        parent_page_id=notion_page_id
    )
    print(f"[Pipeline] ✓ Done! {page_url}")

    return page_url, cleaned

# Command-line interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Digitize a handwritten note image and send it to Notion."
    )
    parser.add_argument(
        "--num-lines", type=int, required=True,
        help="Number of lines in the image"
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
        help="Skip Notion export — just print the cleaned text."
    )
    args = parser.parse_args()

    print("[Pipeline] Loading OCR model...")
    model = load_ocr_model()

    url, text = run_pipeline(
        num_lines=args.num_lines,
        image_path=args.image,
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