"""
CV handling: PDF -> text -> text without personal identifiers.

The CV is the *question* in this RAG system. It is embedded once per
upload and compared against the stored jobs. Before that, name, email
and phone are removed: a search index has no reason to hold them, and
the model has no reason to see them.
"""

import re
import sys
from pathlib import Path

from pypdf import PdfReader


def extract_text(path: Path) -> str:
    """All pages joined, whitespace collapsed. Raises if nothing is extractable
    (a scanned image has no text layer)."""
    reader = PdfReader(str(path))
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("No text found in PDF. Is it a scanned image?")
    return text


# Patterns for the identifiers we strip. Deliberately simple: this is a
# gesture towards governance, not a full PII engine.
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
URL = re.compile(r"(?:https?://|www\.)\S+|(?:linkedin\.com|github\.com)/\S+")


def strip_pii(text: str, name: str | None = None) -> str:
    """Replace email, phone, links, and (if given) the person's name."""
    text = EMAIL.sub("[email]", text)
    text = PHONE.sub("[phone]", text)
    text = URL.sub("[link]", text)
    if name:
        text = re.sub(re.escape(name), "[name]", text, flags=re.IGNORECASE)
    return text


if __name__ == "__main__":
    # usage: uv run python -m app.rag.cv path/to/cv.pdf ["Full Name"]
    pdf = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else None

    raw = extract_text(pdf)
    clean = strip_pii(raw, name)

    print(f"pages -> {len(raw)} characters of text\n")
    print("first 400 characters, after stripping:\n")
    print(clean[:400])
