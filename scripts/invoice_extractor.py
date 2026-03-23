#!/usr/bin/env python3
"""
invoice_extractor.py — Extract structured data from scanned invoice PDFs using Claude Vision.

Usage:
    python invoice_extractor.py <invoice.pdf> [output.json]

Requirements:
    pip install anthropic pdf2image pillow
    apt-get install poppler-utils  # needed by pdf2image
"""

import anthropic
import base64
import json
import sys
from pathlib import Path


def pdf_to_images_base64(pdf_path: str) -> list[str]:
    """Convert each PDF page to a base64-encoded PNG image."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("Error: pdf2image not installed. Run: pip install pdf2image")
        sys.exit(1)

    images = convert_from_path(pdf_path, dpi=200)
    result = []
    for img in images:
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(base64.standard_b64encode(buf.getvalue()).decode("utf-8"))
    return result


def extract_invoice_data(pdf_path: str) -> dict:
    """
    Extract structured data from a scanned invoice PDF using Claude Vision.

    Returns a dict with:
        - issuer: name, tax_id, address
        - recipient: name, tax_id, address
        - invoice_number
        - date
        - due_date
        - line_items: list of {description, quantity, unit_price, total}
        - subtotal
        - tax_rate
        - tax_amount
        - total
        - currency
        - notes
    """
    client = anthropic.Anthropic()

    print(f"Converting PDF to images: {pdf_path}")
    pages_b64 = pdf_to_images_base64(pdf_path)
    print(f"  → {len(pages_b64)} page(s) detected")

    # Build content blocks: one image per page + instruction
    content = []
    for i, page_b64 in enumerate(pages_b64):
        content.append({
            "type": "text",
            "text": f"Page {i + 1} of {len(pages_b64)}:"
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": page_b64
            }
        })

    content.append({
        "type": "text",
        "text": (
            "Analyze the invoice image(s) above and extract all relevant data. "
            "Return ONLY a valid JSON object with the following structure (use null for missing fields):\n\n"
            "{\n"
            '  "issuer": {"name": null, "tax_id": null, "address": null},\n'
            '  "recipient": {"name": null, "tax_id": null, "address": null},\n'
            '  "invoice_number": null,\n'
            '  "date": null,\n'
            '  "due_date": null,\n'
            '  "line_items": [\n'
            '    {"description": null, "quantity": null, "unit_price": null, "total": null}\n'
            '  ],\n'
            '  "subtotal": null,\n'
            '  "tax_rate": null,\n'
            '  "tax_amount": null,\n'
            '  "total": null,\n'
            '  "currency": null,\n'
            '  "notes": null\n'
            "}\n\n"
            "Use ISO 8601 format for dates (YYYY-MM-DD). "
            "Use numbers (not strings) for amounts. "
            "Do not include any text outside the JSON."
        )
    })

    print("Sending to Claude Vision for extraction...")
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    return json.loads(raw)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(pdf_path).exists():
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    data = extract_invoice_data(pdf_path)

    output = json.dumps(data, indent=2, ensure_ascii=False)

    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
        print(f"Data saved to: {output_path}")
    else:
        print("\n--- Extracted Invoice Data ---")
        print(output)


if __name__ == "__main__":
    main()
