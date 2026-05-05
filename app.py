"""
app.py
──────
Gradio web UI for Handwritten Notes Digitizer.
"""

import gradio as gr
import os
import tempfile
from PIL import Image
from dotenv import load_dotenv

# Import our pipeline and the new Notion fetching tool
from pipeline import load_ocr_model, run_pipeline
from src.notion_export import fetch_available_pages

load_dotenv()

print("[App] Loading OCR model at startup...")
ocr_model = load_ocr_model()
print("[App] Model ready.")


def digitize_note(
    num_lines,
    image,
    llm_provider,
    notion_mode,
    notion_token,
    manual_page_id,
    dropdown_page_id,
):
    """
    Called when the user clicks Digitize.
    """
    if image is None:
        return "Please upload an image first.", ""

    # 1. Prepare Notion credentials/IDs
    safe_token = notion_token.strip() if notion_token else None
    
    # Logic: If appending, use the dropdown selection. If creating new, use the manual ID.
    if notion_mode == "Append to existing page":
        final_id = dropdown_page_id
    else:
        final_id = manual_page_id.strip() if manual_page_id else None

    # 2. Save image to temp file for PaddleOCR
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        image.save(tmp_path, format="JPEG")

    try:
        export_bool = notion_mode != "Skip"
        is_append = notion_mode == "Append to existing page"

        # 3. Execute Pipeline
        url, cleaned = run_pipeline(
            num_lines=num_lines,
            image_path=tmp_path,
            ocr_model=ocr_model,
            llm_provider=llm_provider,
            notion_token=safe_token,
            notion_page_id=final_id,      # Parent ID for new pages
            export_to_notion_bool=export_bool,
            append_mode=is_append,
            existing_page_id=final_id     # Target ID for append
        )

        notion_result = url if url else "Notion export skipped."
        return cleaned, notion_result

    except Exception as e:
        return f"Error: {str(e)}", ""

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_notion_pages(token):
    """Fetches pages from Notion to populate the dropdown."""
    if not token:
        # If token is empty, try to get from .env
        token = os.getenv("NOTION_TOKEN")
    
    if not token:
        return gr.update(choices=[], value=None, label="No Token Found (Check .env)")
    
    try:
        pages_dict = fetch_available_pages(token)
        if not pages_dict:
            return gr.update(choices=[], label="No pages found (Check Integration Permissions)")
        
        # choices=[(display_name, value), ...]
        choices = [(name, p_id) for name, p_id in pages_dict.items()]
        return gr.update(choices=choices, label=f"Found {len(choices)} pages")
    except Exception as e:
        return gr.update(choices=[], label=f"Error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# UI layout
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="Handwritten Notes Digitizer") as demo:

    gr.Markdown("Handwritten Notes Digitizer")

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column():
            image_input = gr.Image(type="pil", label="Photo of handwritten notes")
            
            num_lines_input = gr.Number(value=5, label="Number of lines to extract", precision=0)
            
            llm_radio = gr.Radio(choices=["groq", "gemini"], value="groq", label="LLM for text cleanup")
            
            notion_mode_radio = gr.Radio(
                choices=["Create new page", "Append to existing page", "Skip"],
                value="Create new page",
                label="Notion Mode"
            )

            with gr.Accordion("Notion Settings", open=True):
                token_input = gr.Textbox(label="Notion Token", type="password")
                
                refresh_btn = gr.Button("Load/Refresh My Pages")
                
                page_dropdown = gr.Dropdown(
                    label="Select Page",
                    choices=[],
                    interactive=True
                )
                

            run_btn = gr.Button("Digitize", variant="primary")

        # ── Right column: outputs ─────────────────────────────────────────────
        with gr.Column():
            text_output = gr.Textbox(label="Extracted & Cleaned Text", lines=15)
            notion_output = gr.Textbox(label="Notion Page URL")

    # --- Interactions ---
    
    # 1. Refreshing the list
    refresh_btn.click(
        fn=get_notion_pages,
        inputs=[token_input],
        outputs=[page_dropdown]
    )

    # 2. Running the main task
    run_btn.click(
        fn=digitize_note,
        inputs=[
            num_lines_input,
            image_input,
            llm_radio,
            notion_mode_radio,
            token_input,
            page_dropdown
        ],
        outputs=[text_output, notion_output]
    )

if __name__ == "__main__":
    demo.launch()