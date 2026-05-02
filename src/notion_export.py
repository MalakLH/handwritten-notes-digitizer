"""
notion_export.py
────────────────
Takes clean text (from postprocessing.py) and creates a formatted Notion page.

This is almost exactly your Notion_integration.ipynb notebook, refactored into
clean functions with docstrings explaining every decision.

The flow:
  1. parse_text_to_blocks()  → converts plain text into Notion block objects
  2. create_notion_page()    → calls the Notion API to create the actual page

Usage:
    from src.notion_export import create_notion_page
    url = create_notion_page(clean_text, notion_token, parent_page_id)
    print(url)  # → https://www.notion.so/...
"""

import re
import os
from datetime import datetime
from notion_client import Client


# ─────────────────────────────────────────────────────────────────────────────
# Block factory helpers
# These are small functions that return a Notion block dict in the format
# the Notion API expects. Each block type has a specific JSON structure.
# ─────────────────────────────────────────────────────────────────────────────

def _make_heading(text: str, level: int = 2) -> dict:
    """Create a Notion heading block (h1, h2, or h3)."""
    block_type = f"heading_{level}"
    return {
        "object": "block",
        "type": block_type,
        block_type: {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        }
    }


def _make_paragraph(text: str) -> dict:
    """Create a Notion paragraph block."""
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        }
    }


def _make_bullet(text: str) -> dict:
    """Create a Notion bulleted list item block."""
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        }
    }


def _make_numbered(text: str) -> dict:
    """Create a Notion numbered list item block."""
    return {
        "object": "block",
        "type": "numbered_list_item",
        "numbered_list_item": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        }
    }


def _is_special_line(line: str) -> bool:
    """
    Returns True if a line is a heading, bullet, or numbered item.
    Used to decide when to stop collecting paragraph lines.
    """
    return (
        line.endswith(":")                          or  # heading
        line.startswith(("-", "*", "+)", "->"))     or  # bullet
        re.match(r"^\d+[\.)] ", line) is not None       # numbered
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text → Notion blocks parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_text_to_blocks(text: str) -> tuple:
    """
    Parse plain text (from the LLM cleanup step) into a page title + list of
    Notion block dicts.
    
    Detection rules (in priority order):
      - Lines ending with ':'         → Heading (first one becomes page title)
      - Lines starting with -, *, +)  → Bullet point
      - Lines starting with 1., 2.    → Numbered list item
      - Everything else               → Paragraph (consecutive lines are merged)
    
    Args:
        text: The cleaned text string from postprocessing.clean_text()
    
    Returns:
        (title, blocks) tuple where:
          - title is a string (used as the Notion page title)
          - blocks is a list of Notion block dicts
    """
    lines = text.strip().split("\n")
    blocks = []
    title = None
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            i += 1
            continue

        # ── Heading: line ends with ':' ───────────────────────────────────────
        if stripped.endswith(":") and len(stripped) > 1:
            heading_text = stripped[:-1].strip()
            
            # The FIRST heading becomes the page title, not a block
            # (Notion pages have a dedicated title field)
            if title is None:
                title = heading_text
            else:
                blocks.append(_make_heading(heading_text, level=2))
            i += 1
            continue

        # ── Bullet point ─────────────────────────────────────────────────────
        if stripped.startswith(("-", "*", "+)", "->")):
            bullet_text = stripped.lstrip("-*+)>").strip()
            blocks.append(_make_bullet(bullet_text))
            i += 1
            continue

        # ── Numbered list ────────────────────────────────────────────────────
        if re.match(r"^\d+[\.)] ", stripped):
            item_text = re.sub(r"^\d+[\.)] +", "", stripped).strip()
            blocks.append(_make_numbered(item_text))
            i += 1
            continue

        # ── Paragraph: collect consecutive non-special lines ─────────────────
        # WHY: We merge consecutive plain lines into one paragraph block.
        # This avoids creating 50 tiny paragraph blocks for a long section of prose.
        para_lines = []
        while i < len(lines):
            current = lines[i].strip()
            if not current or _is_special_line(current):
                break
            para_lines.append(current)
            i += 1

        if para_lines:
            blocks.append(_make_paragraph(" ".join(para_lines)))
        # Note: no i++ here, the inner loop already advanced i

    # Fallback title: if the text had no headings, generate one from the timestamp
    if title is None:
        title = f"Note — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    return title, blocks


# ─────────────────────────────────────────────────────────────────────────────
# Main function — create the Notion page
# ─────────────────────────────────────────────────────────────────────────────

def create_notion_page(
    clean_text: str,
    notion_token: str = None,
    parent_page_id: str = None
) -> str:
    """
    Create a formatted Notion page from clean text and return its URL.
    
    Args:
        clean_text:     The cleaned text from postprocessing.clean_text().
        notion_token:   Your Notion integration token (starts with "secret_").
                        If None, reads from NOTION_TOKEN environment variable.
        parent_page_id: The ID of the Notion page to create the note under.
                        If None, reads from NOTION_PARENT_PAGE_ID env var.
    
    Returns:
        The URL of the newly created Notion page.
        Example: "https://www.notion.so/350e54104f86819aba08d4d2535ad4d0"
    
    Raises:
        ValueError: If credentials are missing.
    """
    # ── Get credentials ───────────────────────────────────────────────────────
    token     = notion_token     or os.getenv("NOTION_TOKEN")
    parent_id = parent_page_id   or os.getenv("NOTION_PARENT_PAGE_ID")
    
    if not token:
        raise ValueError(
            "Notion token not found. Set NOTION_TOKEN in .env or pass it directly.\n"
            "Get a token at: https://www.notion.so/my-integrations"
        )
    if not parent_id:
        raise ValueError(
            "Notion parent page ID not found. Set NOTION_PARENT_PAGE_ID in .env\n"
            "or pass it directly. It's the long ID in your Notion page URL."
        )

    # ── Parse text → blocks ───────────────────────────────────────────────────
    title, content_blocks = parse_text_to_blocks(clean_text)
    
    # ── Build header blocks (metadata callout + divider) ─────────────────────
    # These appear at the top of every page so you know when it was created
    timestamp = datetime.now().strftime("%Y-%m-%d at %H:%M")
    header_blocks = [
        {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": f"Auto-imported on {timestamp}"}
                }],
                "icon": {"emoji": "📷"}
            }
        },
        {
            "object": "block",
            "type": "divider",
            "divider": {}
        }
    ]

    all_blocks = header_blocks + content_blocks
    
    # ── Create the page ───────────────────────────────────────────────────────
    notion = Client(auth=token)
    
    # Notion API allows max 100 blocks per request.
    # We send the first 100, then append the rest in batches.
    first_batch = all_blocks[:100]
    
    response = notion.pages.create(
        parent={"page_id": parent_id},
        properties={
            "title": {
                "title": [{"type": "text", "text": {"content": title}}]
            }
        },
        children=first_batch
    )
    
    page_id = response["id"]
    
    # ── Append remaining blocks if page has more than 100 blocks ─────────────
    remaining = all_blocks[100:]
    while remaining:
        batch     = remaining[:100]
        remaining = remaining[100:]
        notion.blocks.children.append(block_id=page_id, children=batch)
    
    page_url = f"https://www.notion.so/{page_id.replace('-', '')}"
    print(f"[Notion] Page created: {page_url}")
    return page_url
