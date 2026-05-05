"""
notion_export.py
────────────────
Takes clean text (from postprocessing.py) and creates a formatted Notion page.

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

import re
import os
from datetime import datetime
from notion_client import Client

# ... (Keep _make_heading, _make_paragraph, _make_bullet, _make_numbered, _is_special_line, and parse_text_to_blocks as they are) ...

def export_to_notion(
    clean_text: str,
    notion_token: str = None,
    page_id: str = None,
    append_mode: bool = False
) -> str:
    """
    Handles both creating a new page and appending to an existing one.
    
    Args:
        clean_text: The cleaned text from the LLM.
        notion_token: Integration secret.
        page_id: Parent ID (if new) or Target ID (if append).
        append_mode: If True, adds to existing page. If False, creates new.
    """
    token = notion_token or os.getenv("NOTION_TOKEN")
    target_id = page_id or os.getenv("NOTION_PARENT_PAGE_ID")
    
    if not token or not target_id:
        raise ValueError("Missing Notion credentials (token or page ID).")

    notion = Client(auth=token)
    title, content_blocks = parse_text_to_blocks(clean_text)
    
    # Create a timestamp header for the new entry
    timestamp = datetime.now().strftime("%Y-%m-%d at %H:%M")
    header_blocks = [
        {"object": "block", "type": "divider", "divider": {}},
        {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": f"Entry added on {timestamp}"}}],
                "icon": {"emoji": "📷"}
            }
        }
    ]
    
    all_blocks = header_blocks + content_blocks

    if append_mode:
        # ── APPEND MODE: Add blocks to existing page ──────────────────────────
        print(f"[Notion] Appending to existing page: {target_id}")
        # Notion blocks.children.append takes max 100 blocks at a time
        for i in range(0, len(all_blocks), 100):
            batch = all_blocks[i:i+100]
            notion.blocks.children.append(block_id=target_id, children=batch)
        
        final_id = target_id
    else:
        # ── CREATE MODE: Create a brand new child page ────────────────────────
        print(f"[Notion] Creating new page under: {target_id}")
        first_batch = all_blocks[:100]
        response = notion.pages.create(
            parent={"page_id": target_id},
            properties={
                "title": {"title": [{"type": "text", "text": {"content": title}}]}
            },
            children=first_batch
        )
        final_id = response["id"]
        
        # Append remaining if > 100 blocks
        remaining = all_blocks[100:]
        for i in range(0, len(remaining), 100):
            batch = remaining[i:i+100]
            notion.blocks.children.append(block_id=final_id, children=batch)

    page_url = f"https://www.notion.so/{final_id.replace('-', '')}"
    return page_url

def fetch_available_pages(notion_token: str = None) -> dict:
    """Fetches all pages accessible by the integration."""
    token = notion_token or os.getenv("NOTION_TOKEN")
    if not token:
        return {}
    
    notion = Client(auth=token)
    # Search for objects of type 'page'
    results = notion.search(filter={"property": "object", "value": "page"}).get("results", [])
    
    pages = {}
    for page in results:
        # Extract title (handling cases where title might be empty)
        properties = page.get("properties", {})
        # Most pages use 'title' or 'Name' as the primary key
        title_list = properties.get("title", {}).get("title", []) or properties.get("Name", {}).get("title", [])
        title = title_list[0].get("plain_text", "Untitled") if title_list else "Untitled"
        pages[title] = page["id"]
        
    return pages