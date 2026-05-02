"""
app.py
──────
The Gradio web UI. This is what gets deployed on Hugging Face Spaces.

Users open this in a browser, upload a photo of their notes,
enter their API credentials, and get back clean text + a Notion page.

Run locally with:
    python app.py

Deploy to Hugging Face Spaces by pushing this file + requirements.txt
to a new Space repo.
"""

import gradio as gr
import os
from dotenv import load_dotenv
from pipeline import load_ocr_model, run_pipeline

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Load the OCR model ONCE when the app starts (not on every button click)
# ─────────────────────────────────────────────────────────────────────────────
print("[App] Loading OCR model at startup...")
ocr_model = load_ocr_model()
print("[App] Model ready.")


# ─────────────────────────────────────────────────────────────────────────────
# The function that Gradio calls when the user hits "Digitize"
# ─────────────────────────────────────────────────────────────────────────────

def digitize_note(
    image,              # PIL Image from the Gradio image input
    llm_provider,       # "groq" or "gemini" from the radio buttons
    notion_token,       # string from the text box
    notion_page_id,     # string from the text box
    export_to_notion    # bool from the checkbox
):
    """
    Called by Gradio when the user clicks the "Digitize" button.
    Returns (clean_text, notion_url_or_status_message).
    """
    if image is None:
        return "Please upload an image first.", ""

    try:
        url, clean = run_pipeline(
            image_input=image,
            ocr_model=ocr_model,
            llm_provider=llm_provider,
            notion_token=notion_token if notion_token.strip() else None,
            notion_page_id=notion_page_id if notion_page_id.strip() else None,
            export_to_notion=export_to_notion
        )

        notion_result = url if url else "Notion export skipped."
        return clean, notion_result

    except Exception as e:
        return f"Error: {str(e)}", ""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI layout
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Handwritten Notes Digitizer") as demo:

    gr.Markdown("""
    # 📝 Handwritten Notes Digitizer
    Upload a photo of your handwritten notes. The app will:
    1. Preprocess the image (grayscale, contrast, deskew)
    2. Run fine-tuned PaddleOCR to extract text
    3. Clean up OCR errors with an LLM (Groq or Gemini)
    4. Optionally create a formatted Notion page
    """)

    with gr.Row():
        # Left column: inputs
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload photo of handwritten notes"
            )
            llm_radio = gr.Radio(
                choices=["groq", "gemini"],
                value="groq",
                label="LLM for text cleanup"
            )
            export_checkbox = gr.Checkbox(
                value=True,
                label="Export to Notion"
            )
            with gr.Accordion("Notion Credentials", open=False):
                gr.Markdown(
                    "Leave blank to use values from your `.env` file. "
                    "Get a token at [notion.so/my-integrations](https://www.notion.so/my-integrations)"
                )
                notion_token_input = gr.Textbox(
                    label="Notion Token",
                    placeholder="secret_...",
                    type="password"
                )
                notion_page_input = gr.Textbox(
                    label="Notion Parent Page ID",
                    placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                )

            run_btn = gr.Button("✨ Digitize", variant="primary")

        # Right column: outputs
        with gr.Column():
            text_output = gr.Textbox(
                label="Extracted & Cleaned Text",
                lines=15,
                show_copy_button=True
            )
            notion_output = gr.Textbox(
                label="Notion Page URL",
                show_copy_button=True
            )

    # Wire the button to the function
    run_btn.click(
        fn=digitize_note,
        inputs=[
            image_input,
            llm_radio,
            notion_token_input,
            notion_page_input,
            export_checkbox
        ],
        outputs=[text_output, notion_output]
    )

    gr.Examples(
        examples=[["data/sample_images/sample1.jpg"]],
        inputs=image_input,
        label="Try a sample image"
    )


if __name__ == "__main__":
    demo.launch()
