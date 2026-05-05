
# 📝 Handwritten Notes Digitizer

**An end-to-end AI pipeline that turns photos of handwritten notes into clean, structured Notion pages — powered by a fine-tuned PaddleOCR model and LLM post-processing.**

Built with ❤️ by [Malak LH](https://github.com/MalakLH)

<br />
<div align="center">
  <a href="https://github.com/MalakLH/handwritten-notes-digitizer">
     <div style="width: 120px; height: 120px; background: linear-gradient(135deg, #6366F1, #8B5CF6); border-radius: 20%; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);">
        <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
     </div>
  </a>

  <h3>Handwritten Notes Digitizer</h3>

  <p align="center">
    <strong>Snap a photo. Get a Notion page.</strong>
    <br />
    <br />
    <a href="https://github.com/MalakLH/handwritten-notes-digitizer/issues/new?labels=bug">Report Bug</a>
    ·
    <a href="https://github.com/MalakLH/handwritten-notes-digitizer/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<br/>

---

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#-features">Features</a></li>
    <li><a href="#-pipeline-architecture">Pipeline Architecture</a></li>
    <li><a href="#-model">Model</a></li>
    <li><a href="#-getting-started">Getting Started</a></li>
    <li><a href="#-usage">Usage</a></li>
    <li><a href="#-configuration">Configuration</a></li>
    <li><a href="#-project-structure">Project Structure</a></li>
    <li><a href="#-license">License</a></li>
    <li><a href="#-contact">Contact</a></li>
  </ol>
</details>

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## About The Project

**Handwritten Notes Digitizer** is a personal AI project built to solve a real problem: notebooks full of handwritten notes that are impossible to search, share, or organize.

The project went through a full research-and-build cycle:

- Collected and labeled a **custom dataset of 1,000+ handwriting images**
- Benchmarked **5 OCR models**: TrOCR (small, base, large), LightOnOCR, and PaddleOCR
- Fine-tuned **PaddleOCR PP-OCRv5** twice — once on CPU, once on GPU — on the personal handwriting dataset
- Added **LLM post-processing** (Groq / Gemini) to fix OCR errors in context
- Integrated the **Notion API** to auto-create formatted pages from recognized text
- Wrapped everything in a **Gradio web UI** deployable on Hugging Face Spaces

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## ✨ Features

- 🧠 **Fine-tuned OCR**: PP-OCRv5 model trained on 1,000+ personal handwriting samples for significantly better accuracy than off-the-shelf models
- 🔧 **LLM Cleanup**: Groq (Llama 3.1) or Gemini fixes OCR errors — broken words, wrong characters, capitalization — without adding new content
- 📝 **Notion Integration**: Automatically creates a structured Notion page with headings, bullet points, and numbered lists parsed from the raw text
- ➕ **Append Mode**: Send multiple scans to the same Notion page, each separated by a timestamp divider
- 🖥️ **Gradio UI**: Clean browser interface — upload a photo, pick your LLM, choose Notion mode, click Digitize
- ⌨️ **CLI Support**: Run the full pipeline from the terminal with a single command

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🔁 Pipeline Architecture

```
📷 Photo of handwritten notes
        │
        ▼
┌───────────────────────┐
│   src/ocr.py          │  Fine-tuned PP-OCRv5 model
│   PaddleOCR inference │  Detects + reads text lines
└───────────────────────┘
        │  raw text string
        ▼
┌───────────────────────┐
│ src/postprocessing.py │  Groq (Llama 3.1) or Gemini
│   LLM error cleanup   │  Fixes OCR errors, preserves structure
└───────────────────────┘
        │  clean text string
        ▼
┌───────────────────────┐
│ src/notion_export.py  │  Notion API
│   Page creation       │  Headings, bullets, numbered lists
└───────────────────────┘
        │
        ▼
📄 Notion page at https://notion.so/...
```

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🧠 Model

The OCR model is a fine-tuned version of **PaddleOCR PP-OCRv5 mobile rec**, trained on a custom dataset of 1,000+ labeled handwriting images.

| | Pretrained PP-OCRv5 | Fine-tuned (this project) |
|---|---|---|
| Dataset | General printed + handwriting | Personal handwriting (1,000+ images) |
| Training | Baseline | 2 fine-tuning runs (CPU + GPU) |
| Target | General use | Personal handwriting style |

The fine-tuned weights are stored in `models/my_handwriting_model/` as inference format files (`.pdiparams`, `.json`, `.yml`).

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or 3.11 (PaddlePaddle does not support 3.12+)
- Git

### Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/MalakLH/handwritten-notes-digitizer.git
cd handwritten-notes-digitizer

# Step 2: Create and activate a virtual environment
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate
# Mac/Linux
source .venv/bin/activate

# Step 3: Install PaddlePaddle (CPU) — must be installed separately
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Step 4: Install all other dependencies
pip install -r requirements.txt

# Step 5: Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys
```

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 📚 Usage

### Web UI

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser. Upload a photo, choose your LLM and Notion mode, click **Digitize**.

### Command Line

```bash
# Full pipeline → creates a new Notion page
python pipeline.py --image data/sample_images/test1.jpg

# Skip Notion — just print the cleaned text
python pipeline.py --image data/sample_images/test1.jpg --no-notion

# Append to an existing Notion page instead of creating a new one
python pipeline.py --image data/sample_images/test1.jpg --append-to YOUR_PAGE_ID

# Use Gemini instead of Groq for cleanup
python pipeline.py --image data/sample_images/test1.jpg --provider gemini
```

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🪛 Configuration

Copy `.env.example` to `.env` and fill in your keys:

```env
# Path to your fine-tuned model weights folder
MODEL_DIR=models/my_handwriting_model

# Set to true only if you have a CUDA-compatible GPU
USE_GPU=false

# Get a free key at https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Get a free key at https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Create an integration at https://www.notion.so/my-integrations
NOTION_TOKEN=secret_your_token_here

# The ID at the end of your Notion page URL
NOTION_PARENT_PAGE_ID=your_page_id_here
```

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 📁 Project Structure

```
handwritten-notes-digitizer/
├── app.py                        # Gradio web UI
├── pipeline.py                   # Main orchestrator
├── src/
│   ├── __init__.py
│   ├── ocr.py                    # PaddleOCR model wrapper
│   ├── postprocessing.py         # Groq / Gemini text cleanup
│   └── notion_export.py          # Notion API integration
├── models/
│   └── my_handwriting_model/     # Fine-tuned model weights
├── data/
│   └── sample_images/            # Sample images for testing
├── notebooks/                    # Original research notebooks
│   ├── 01_trocr_small_experiments.ipynb
│   ├── 02_trocr_base_experiments.ipynb
│   ├── 03_lightonocr_testing.ipynb
│   ├── 04_paddleocr_large_testing.ipynb
│   ├── 05_finetuning_cpu.ipynb
│   ├── 06_finetuning_gpu.ipynb
│   └── 07_notion_integration.ipynb
├── requirements.txt
├── .env.example
└── .gitignore
```

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 📃 License

Distributed under the MIT License. See `LICENSE` for more information.

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 📧 Contact

**Malak LH** — [@MalakLH](https://github.com/MalakLH)

Project Link: [https://github.com/MalakLH/handwritten-notes-digitizer](https://github.com/MalakLH/handwritten-notes-digitizer)

---

*Handwritten Notes Digitizer — because your notebooks deserve to be searchable.*

[contributors-shield]: https://img.shields.io/github/contributors/MalakLH/handwritten-notes-digitizer.svg?style=for-the-badge
[contributors-url]: https://github.com/MalakLH/handwritten-notes-digitizer/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/MalakLH/handwritten-notes-digitizer.svg?style=for-the-badge
[forks-url]: https://github.com/MalakLH/handwritten-notes-digitizer/network/members
[stars-shield]: https://img.shields.io/github/stars/MalakLH/handwritten-notes-digitizer.svg?style=for-the-badge
[stars-url]: https://github.com/MalakLH/handwritten-notes-digitizer/stargazers
[issues-shield]: https://img.shields.io/github/issues/MalakLH/handwritten-notes-digitizer.svg?style=for-the-badge
[issues-url]: https://github.com/MalakLH/handwritten-notes-digitizer/issues
[license-shield]: https://img.shields.io/github/license/MalakLH/handwritten-notes-digitizer.svg?style=for-the-badge
[license-url]: https://github.com/MalakLH/handwritten-notes-digitizer/blob/master/LICENSE
