from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import statistics

import fitz

from recall_backend.memory import estimate_tokens


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
WHITESPACE = re.compile(r"\s+")
LIST_MARKER = re.compile(r"^\s*([-*•‣◦▪]|\d+[.)]|\$|>)\s+")
TRAILING_SENTENCE_PUNCT = (".", "!", "?", ":", ";")
BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold text


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    page_start: int
    page_end: int
    token_count: int
    heading: str | None = None


def normalize_text(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def split_units(text: str) -> list[str]:
    """Split a single (already-merged) paragraph of prose into sentence-sized
    units when it's too long for one chunk unit. Only used for continuous
    prose paragraphs assembled by merge logic downstream, never for raw
    page/block text, since it can't recover line structure that's already
    been collapsed."""
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    token_count = estimate_tokens(cleaned)
    if token_count <= 220:
        return [cleaned]

    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(cleaned) if sentence.strip()]
    if len(sentences) <= 1:
        return [cleaned]

    return sentences


def looks_like_list_line(text: str) -> bool:
    """Detect cheat-sheet / table-style rows: bullet markers, shell prompts,
    or "command    description" rows with wide interior gaps (common when a
    PDF renders columns via spacing rather than real table structure)."""
    if LIST_MARKER.match(text):
        return True
    if re.search(r"\S {2,}\S", text):
        return True
    return False


def extract_page_lines(page: fitz.Page) -> list[dict]:
    """Pull line-level records (text, font size, bold) straight from
    PyMuPDF's layout dict. This is the fix for the core bug: block-level text
    extraction can merge an entire cheat-sheet page into one or two giant
    blocks with no blank-line breaks and no sentence punctuation, which then
    can't be subdivided by paragraph/sentence heuristics. Working at line
    granularity means structure is never lost in the first place."""
    raw = page.get_text("dict", sort=True)
    lines: list[dict] = []

    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # skip image/non-text blocks
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            text = normalize_text("".join(span.get("text", "") for span in spans))
            if not text:
                continue

            sizes = [span.get("size", 0.0) for span in spans]
            bold = any(int(span.get("flags", 0)) & BOLD_FLAG for span in spans)

            lines.append({
                "text": text,
                "size": max(sizes) if sizes else 0.0,
                "bold": bold,
            })

    return lines


def compute_body_size(all_lines: list[dict]) -> float:
    sizes = [line["size"] for line in all_lines if line["text"]]
    if not sizes:
        return 10.0
    return statistics.median(sizes)


def is_heading_line(line: dict, body_size: float) -> bool:
    text = line["text"]
    word_count = len(text.split())
    if word_count == 0 or word_count > 8:
        return False
    if text.endswith(TRAILING_SENTENCE_PUNCT):
        return False

    size_boost = body_size > 0 and line["size"] >= body_size * 1.12
    is_shouty = text.isupper() and word_count <= 6
    return bool(line["bold"] or size_boost or is_shouty)


def build_line_units(lines: list[dict], page_number: int, body_size: float) -> list[dict]:
    """Turn a page's line records into retrieval units:
    - heading lines stay atomic and are tagged so build_chunks can break
      sections cleanly and anchor continuation chunks with their heading
    - list/table rows stay atomic (each row is usually a complete idea)
    - everything else (wrapped prose) is merged back into paragraphs and,
      if still too long, sentence-split
    """
    tagged = [{**line, "_heading": is_heading_line(line, body_size)} for line in lines]

    units: list[dict] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        paragraph = normalize_text(" ".join(buffer))
        buffer.clear()
        if not paragraph:
            return
        for piece in split_units(paragraph):
            units.append({
                "text": piece,
                "page": page_number,
                "tokens": estimate_tokens(piece),
                "heading": False,
            })

    for line in tagged:
        text = line["text"]
        if not text:
            continue

        if line["_heading"]:
            flush_buffer()
            units.append({
                "text": text,
                "page": page_number,
                "tokens": estimate_tokens(text),
                "heading": True,
            })
            continue

        if looks_like_list_line(text):
            flush_buffer()
            units.append({
                "text": text,
                "page": page_number,
                "tokens": estimate_tokens(text),
                "heading": False,
            })
            continue

        if buffer and not buffer[-1].rstrip().endswith(TRAILING_SENTENCE_PUNCT):
            buffer.append(text)
        else:
            flush_buffer()
            buffer.append(text)

    flush_buffer()
    return units


def collect_document_units(document: fitz.Document) -> tuple[list[dict], list[int]]:
    page_lines: list[list[dict]] = []
    all_lines: list[dict] = []

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        lines = extract_page_lines(page)

        if not lines:
            # Fallback for pages where dict-based line extraction yields
            # nothing (rare edge cases) but plain text extraction still
            # finds content, e.g. some form/annotation-heavy pages.
            plain = page.get_text("text", sort=True)
            if normalize_text(plain):
                lines = [{"text": normalize_text(chunk), "size": 0.0, "bold": False}
                         for chunk in plain.splitlines() if normalize_text(chunk)]

        page_lines.append(lines)
        all_lines.extend(lines)

    body_size = compute_body_size(all_lines)

    units: list[dict] = []
    skipped_pages: list[int] = []

    for page_index, lines in enumerate(page_lines):
        if not lines:
            skipped_pages.append(page_index + 1)
            continue

        page_units = build_line_units(lines, page_index + 1, body_size)
        if not page_units:
            skipped_pages.append(page_index + 1)
            continue

        units.extend(page_units)

    return units, skipped_pages


def profile_document(units: list[dict], page_count: int) -> dict:
    total_tokens = sum(unit["tokens"] for unit in units)
    unit_count = len(units)
    avg_unit_tokens = total_tokens / max(unit_count, 1)
    heading_count = sum(1 for unit in units if unit.get("heading"))
    density = total_tokens / max(page_count, 1)

    # Reference material (cheat sheets, command tables, API listings) is made
    # of many tiny atomic units rather than sentence-length prose. Treat it
    # differently: crammed target sizes just merge unrelated rows together.
    is_reference_style = unit_count > 0 and avg_unit_tokens < 14

    if page_count <= 6:
        target_tokens = 420
        overlap_tokens = 50
        strategy = "adaptive-small"
    elif page_count <= 24:
        target_tokens = 320
        overlap_tokens = 40
        strategy = "adaptive-medium"
    else:
        target_tokens = 240
        overlap_tokens = 28
        strategy = "adaptive-large"

    if density > 900:
        target_tokens = max(180, int(target_tokens * 0.78))
        overlap_tokens = max(20, int(overlap_tokens * 0.75))
        strategy = f"{strategy}-dense"
    elif density < 180:
        target_tokens = int(target_tokens * 1.15)
        overlap_tokens = int(overlap_tokens * 1.15)
        strategy = f"{strategy}-sparse"

    if page_count >= 60:
        target_tokens = max(160, int(target_tokens * 0.9))

    if is_reference_style:
        target_tokens = max(110, min(target_tokens, 200))
        overlap_tokens = max(15, min(overlap_tokens, 25))
        strategy = f"{strategy}-reference"

    # Guarantee a sane floor on chunk count for short, dense, or heavily
    # sectioned documents, so a one-or-two-page cheat sheet doesn't collapse
    # into 2-3 chunks that blur every section together.
    min_chunks_target = max(3, heading_count, round(total_tokens / 900))
    projected_chunks = total_tokens / target_tokens if target_tokens else 0

    if total_tokens > 0 and projected_chunks < min_chunks_target:
        target_tokens = max(110, int(total_tokens / min_chunks_target))
        overlap_tokens = min(overlap_tokens, max(10, int(target_tokens * 0.15)))
        strategy = f"{strategy}-finegrained"

    return {
        "page_count": page_count,
        "total_tokens": total_tokens,
        "avg_tokens": avg_unit_tokens,
        "density": density,
        "target_tokens": target_tokens,
        "overlap_tokens": overlap_tokens,
        "strategy": strategy,
        "heading_count": heading_count,
        "is_reference_style": is_reference_style,
    }


def finalize_chunk(parts: list[dict], heading: str | None = None) -> ChunkDraft | None:
    if not parts:
        return None

    text = "\n\n".join(part["text"] for part in parts).strip()
    if not text:
        return None

    page_start = min(part["page"] for part in parts)
    page_end = max(part["page"] for part in parts)

    return ChunkDraft(
        text=text,
        page_start=page_start,
        page_end=page_end,
        token_count=estimate_tokens(text),
        heading=heading,
    )


def trim_overlap(parts: list[dict], overlap_tokens: int) -> list[dict]:
    if overlap_tokens <= 0 or not parts:
        return []

    retained: list[dict] = []
    token_total = 0

    for part in reversed(parts):
        retained.append(part)
        token_total += part["tokens"]
        if token_total >= overlap_tokens:
            break

    retained.reverse()
    return retained


def split_long_text(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    max_words = max(60, int(max_tokens / 1.3))
    if len(words) <= max_words:
        return [text.strip()]

    pieces: list[str] = []
    start = 0

    while start < len(words):
        end = min(len(words), start + max_words)
        piece = " ".join(words[start:end]).strip()
        if piece:
            pieces.append(piece)
        start = end

    return pieces


def build_chunks(units: list[dict], target_tokens: int, overlap_tokens: int) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    current_parts: list[dict] = []
    current_tokens = 0
    current_heading: str | None = None
    # Don't break a section on every heading if the section-so-far is
    # trivially small — avoids a flood of near-empty chunks in
    # heading-dense reference docs.
    min_section_tokens = max(30, int(target_tokens * 0.25))

    def flush(next_heading: str | None = None) -> None:
        nonlocal current_parts, current_tokens, current_heading
        finalized = finalize_chunk(current_parts, current_heading)
        if finalized:
            chunks.append(finalized)
        current_parts = trim_overlap(current_parts, overlap_tokens)
        current_tokens = sum(part["tokens"] for part in current_parts)
        if next_heading is not None:
            current_heading = next_heading

    for unit in units:
        unit_text = unit["text"]
        unit_tokens = unit["tokens"]
        is_heading = unit.get("heading", False)

        if is_heading:
            if current_parts and current_tokens >= min_section_tokens:
                flush(next_heading=unit_text)
            else:
                current_heading = unit_text
            current_parts.append(unit)
            current_tokens += unit_tokens
            continue

        if unit_tokens > target_tokens * 1.5:
            split_texts = split_long_text(unit_text, target_tokens)
            sub_units = [
                {"text": part, "page": unit["page"], "tokens": estimate_tokens(part), "heading": False}
                for part in split_texts
            ]
        else:
            sub_units = [unit]

        for sub_unit in sub_units:
            sub_tokens = sub_unit["tokens"]

            if current_parts and current_tokens + sub_tokens > target_tokens:
                flush()

            current_parts.append(sub_unit)
            current_tokens += sub_tokens

    final_chunk = finalize_chunk(current_parts, current_heading)
    if final_chunk:
        chunks.append(final_chunk)

    return chunks


def document_identity(pdf_path: Path) -> str:
    stat = pdf_path.stat()
    digest = hashlib.sha1(f"{pdf_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")).hexdigest()
    return digest


def ingest_pdf(pdf_path: str) -> dict:
    path = Path(pdf_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PDF path does not exist: {pdf_path}")
    if not path.is_file():
        raise FileNotFoundError(f"PDF path is not a file: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    document = fitz.open(path)
    try:
        units, skipped_pages = collect_document_units(document)
        if not units:
            raise ValueError(f"No extractable text found in PDF: {pdf_path}")

        profile = profile_document(units, document.page_count)

        chunks = build_chunks(units, profile["target_tokens"], profile["overlap_tokens"])
        if not chunks:
            raise ValueError(f"Unable to chunk PDF: {pdf_path}")

        document_name = path.stem
        document_key = document_identity(path)

        return {
            "pdf_path": str(path.resolve()),
            "document": {
                "document_id": document_key,
                "document_name": document_name,
                "document_path": str(path.resolve()),
                "page_count": profile["page_count"],
                "total_tokens": profile["total_tokens"],
                "avg_tokens": profile["avg_tokens"],
                "density": profile["density"],
                "skipped_pages": skipped_pages,
            },
            "chunks": chunks,
            "strategy": profile["strategy"],
        }
    finally:
        document.close()