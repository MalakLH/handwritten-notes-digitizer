"""
app.py
──────
Gradio web UI. Deployable on Hugging Face Spaces.

Run locally with:
    python app.py
"""

import gradio as gr
import os
import tempfile
from PIL import Image
from dotenv import load_dotenv
from pipeline import load_ocr_model, run_pipeline

load_dotenv()

print("[App] Loading OCR model at startup...")
ocr_model = load_ocr_model()
print("[App] Model ready.")


def digitize_note(
    image,             # PIL Image from Gradio
    llm_provider,      # "groq" or "gemini"
    notion_mode,       # "Create new page" | "Append to existing page" | "Skip"
    notion_token,      # string (optional, overrides .env)
    notion_page_id,    # string — parent page ID (new) or existing page ID (append)
):
    """
    Called by Gradio when the user clicks Digitize.
    Returns (clean_text, notion_url_or_status).
    """
    if image is None:
        return "Please upload an image first.", ""

    # Gradio passes a PIL Image. PaddleOCR needs a file path.
    # We save to a temp file and delete it after inference.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        image.save(tmp_path, format="JPEG")

    try:
        export      = notion_mode != "Skip"
        append_mode = notion_mode == "Append to existing page"

        url, cleaned = run_pipeline(
            image_path=tmp_path,
            ocr_model=ocr_model,
            llm_provider=llm_provider,
            notion_token=notion_token.strip() or None,
            notion_page_id=notion_page_id.strip() or None,
            export_to_notion=export,
            append_mode=append_mode,
            existing_page_id=notion_page_id.strip() or None,
        )

        notion_result = url if url else "Notion export skipped."
        return cleaned, notion_result

    except Exception as e:
        return f"Error: {str(e)}", ""

    finally:
        # Always clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# UI layout
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Handwritten Notes Digitizer") as demo:

    gr.Markdown("""
    # 📝 Handwritten Notes Digitizer
    Upload a photo of your handwritten notes and get clean, structured text — 
    optionally saved directly to Notion.
    """)

    with gr.Row():

        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Photo of handwritten notes"
            )

            llm_radio = gr.Radio(
                choices=["groq", "gemini"],
                value="groq",
                label="LLM for text cleanup"
            )

            notion_mode = gr.Radio(
                choices=["Create new page", "Append to existing page", "Skip"],
                value="Create new page",
                label="Notion export"
            )

            with gr.Accordion("Notion credentials", open=False):
                gr.Markdown(
                    "Leave blank to use values from your `.env` file.  \n"
                    "**Create new page** → paste your *parent* page ID below.  \n"
                    "**Append to existing page** → paste the *existing* page ID below."
                )
                notion_token_input = gr.Textbox(
                    label="Notion Token",
                    placeholder="secret_...",
                    type="password"
                )
                notion_page_input = gr.Textbox(
                    label="Notion Page ID",
                    placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                )

            run_btn = gr.Button("✨ Digitize", variant="primary")

        # ── Right column: outputs ─────────────────────────────────────────────
        with gr.Column():
            text_output = gr.Textbox(
                label="Extracted & Cleaned Text",
                lines=15,
            )
            notion_output = gr.Textbox(
                label="Notion Page URL",
            )

    run_btn.click(
        fn=digitize_note,
        inputs=[
            image_input,
            llm_radio,
            notion_mode,
            notion_token_input,
            notion_page_input,
        ],
        outputs=[text_output, notion_output]
    )

    gr.Examples(
        examples=[["data/sample_images/test1.jpg"]],
        inputs=image_input,
        label="Try a sample image"
    )


if __name__ == "__main__":
    demo.launch()