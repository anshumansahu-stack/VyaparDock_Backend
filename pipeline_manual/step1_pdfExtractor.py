"""Layout-aware text extraction for single- and multi-column resume PDFs.

PyMuPDF 2.1's layout engine is used first to establish reading order. A
geometry-based line fallback is retained for documents where layout analysis
cannot return usable text. Neither path performs OCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
import re

import pymupdf


@dataclass(frozen=True, slots=True)
class _TextLine:
    """One positioned text line extracted from a page."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    index: int

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


_PICTURE_PLACEHOLDER_RE = re.compile(
    r"(?im)^\s*(?:==>\s*)?picture(?:\s+\[[^\]]+\])?(?:\s*<==)?\s*$"
)


def _clean_extracted_text(text: str) -> str:
    """Normalize line endings and remove layout-only picture markers."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = _PICTURE_PLACEHOLDER_RE.sub("", text)
    return text.strip()


def _text_from_spans(spans: list[dict]) -> str:
    """Reconstruct a line while restoring spaces lost between styled spans."""
    pieces: list[str] = []
    previous_x1: float | None = None
    previous_text = ""

    for span in spans:
        text = str(span.get("text", ""))
        if not text:
            continue

        bbox = span.get("bbox")
        try:
            x0 = float(bbox[0]) if bbox is not None else None
            x1 = float(bbox[2]) if bbox is not None else None
            font_size = float(span.get("size", 10.0))
        except (TypeError, ValueError, IndexError):
            x0 = x1 = None
            font_size = 10.0

        if (
            pieces
            and x0 is not None
            and previous_x1 is not None
            and x0 - previous_x1 > max(1.0, font_size * 0.2)
            and not previous_text[-1:].isspace()
            and not text[:1].isspace()
        ):
            pieces.append(" ")

        pieces.append(text)
        previous_text = text
        previous_x1 = x1

    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def _get_page_lines(page: pymupdf.Page) -> list[_TextLine]:
    """Read positioned text lines, splitting distant same-baseline spans.

    Some PDF producers place both columns on one baseline and expose them as
    spans of one PyMuPDF line. Large horizontal gaps are split into independent
    lines before gutter detection so the two columns cannot become one block.
    """
    lines: list[_TextLine] = []
    index = 0
    page_dict = page.get_text("dict", sort=False)
    span_gap_threshold = max(15.0, page.rect.width * 0.025)

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            spans = []
            for span in line.get("spans", []):
                bbox = span.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                try:
                    span_x0, span_y0, span_x1, span_y1 = (
                        float(value) for value in bbox[:4]
                    )
                except (TypeError, ValueError):
                    continue
                if not all(isfinite(value) for value in (span_x0, span_y0, span_x1, span_y1)):
                    continue
                spans.append((span_x0, span_y0, span_x1, span_y1, span))

            spans.sort(key=lambda item: (item[0], item[1]))
            span_groups: list[list[tuple[float, float, float, float, dict]]] = []
            for span in spans:
                if span_groups and span[0] - max(item[2] for item in span_groups[-1]) <= span_gap_threshold:
                    span_groups[-1].append(span)
                else:
                    span_groups.append([span])

            # Lines without usable span geometry still get one attempt using
            # the containing line's own bounding box.
            if not span_groups:
                bbox = line.get("bbox")
                if bbox and len(bbox) >= 4:
                    try:
                        x0, y0, x1, y1 = (float(value) for value in bbox[:4])
                    except (TypeError, ValueError):
                        continue
                    text = _text_from_spans(line.get("spans", []))
                    if text and all(isfinite(value) for value in (x0, y0, x1, y1)):
                        lines.append(_TextLine(x0, y0, x1, y1, text, index))
                        index += 1
                continue

            for group in span_groups:
                x0 = min(item[0] for item in group)
                y0 = min(item[1] for item in group)
                x1 = max(item[2] for item in group)
                y1 = max(item[3] for item in group)
                text = _text_from_spans([item[4] for item in group])
                if not text or x1 <= x0 or y1 <= y0:
                    continue
                lines.append(_TextLine(x0, y0, x1, y1, text, index))
                index += 1

    return lines


def _merge_intervals_length(intervals: list[tuple[float, float]]) -> float:
    """Measure the combined length of overlapping or adjacent intervals."""
    if not intervals:
        return 0.0

    intervals.sort()
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _find_fallback_gutter(
    lines: list[_TextLine], page_width: float, page_height: float
) -> float | None:
    """Find a stable column gap using line geometry and vertical co-occurrence.

    Wide lines (usually a page title or a section heading) are ignored while
    candidate gutters are found. A gap is accepted only when multiple lines on
    both sides overlap vertically, which prevents short single-column lines or
    hanging indents from being mistaken for a second column.
    """
    if page_width <= 0 or page_height <= 0:
        return None

    candidates = [line for line in lines if line.width <= page_width * 0.74]
    if len(candidates) < 4:
        return None

    intervals = sorted((line.x0, line.x1) for line in candidates)
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    minimum_gutter = max(10.0, page_width * 0.018)
    minimum_support = max(20.0, page_height * 0.025)
    best_gutter: float | None = None
    best_score = 0.0

    for (_, left_edge), (right_edge, _) in zip(merged, merged[1:]):
        if right_edge - left_edge < minimum_gutter:
            continue

        gutter = (left_edge + right_edge) / 2.0
        left = [line for line in candidates if line.x1 <= gutter]
        right = [line for line in candidates if line.x0 >= gutter]
        if len(left) < 2 or len(right) < 2:
            continue

        overlaps: list[tuple[float, float]] = []
        matched_left: set[int] = set()
        matched_right: set[int] = set()
        for left_line in left:
            for right_line in right:
                top = max(left_line.y0, right_line.y0)
                bottom = min(left_line.y1, right_line.y1)
                overlap = bottom - top
                minimum_line_height = min(left_line.height, right_line.height)
                if minimum_line_height <= 0 or overlap / minimum_line_height < 0.2:
                    continue
                overlaps.append((top, bottom))
                matched_left.add(left_line.index)
                matched_right.add(right_line.index)

        support = _merge_intervals_length(overlaps)
        if (
            support < minimum_support
            or len(matched_left) < 2
            or len(matched_right) < 2
        ):
            continue

        left_chars = sum(len(line.text.replace(" ", "")) for line in left)
        right_chars = sum(len(line.text.replace(" ", "")) for line in right)
        if not left_chars or not right_chars:
            continue

        balance = min(left_chars, right_chars) / max(left_chars, right_chars)
        score = support * (0.6 + 0.4 * sqrt(balance))
        if score > best_score:
            best_gutter = gutter
            best_score = score

    return best_gutter


def _line_sort_key(line: _TextLine) -> tuple[float, float, int]:
    return line.y0, line.x0, line.index


def _lines_to_text(lines: list[_TextLine]) -> str:
    """Render y-sorted lines with paragraph breaks where the vertical gap warrants."""
    ordered = sorted(lines, key=_line_sort_key)
    if not ordered:
        return ""

    result = [ordered[0].text]
    previous = ordered[0]
    for line in ordered[1:]:
        gap = line.y0 - previous.y1
        separator = "\n\n" if gap > max(5.0, previous.height * 0.8) else "\n"
        result.extend((separator, line.text))
        previous = line
    return "".join(result).strip()


def _assign_region(line: _TextLine, spanning: list[_TextLine]) -> int:
    """Place a column line before or after any full-width lines it overlaps."""
    region = 0
    for index, anchor in enumerate(spanning):
        if line.y0 >= anchor.y1:
            region = index + 1
            continue
        if line.y1 <= anchor.y0:
            break
        line_midpoint = (line.y0 + line.y1) / 2.0
        anchor_midpoint = (anchor.y0 + anchor.y1) / 2.0
        if line_midpoint >= anchor_midpoint:
            region = index + 1
        break
    return region


def _extract_page_by_geometry(
    page: pymupdf.Page,
    *,
    lines: list[_TextLine] | None = None,
    gutter: float | None = None,
) -> str:
    """Fallback reader that reconstructs lines and keeps detected columns apart."""
    if lines is None:
        lines = _get_page_lines(page)
    if not lines:
        return ""

    if gutter is None:
        gutter = _find_fallback_gutter(lines, page.rect.width, page.rect.height)
    if gutter is None:
        return _lines_to_text(lines)

    spanning = sorted(
        (line for line in lines if line.x0 < gutter < line.x1), key=_line_sort_key
    )
    columns = [line for line in lines if line not in spanning]
    regions: list[list[_TextLine]] = [[] for _ in range(len(spanning) + 1)]
    for line in columns:
        regions[_assign_region(line, spanning)].append(line)

    page_parts: list[str] = []
    for index, anchor in enumerate(spanning):
        region = regions[index]
        left = [line for line in region if line.x1 <= gutter]
        right = [line for line in region if line.x0 >= gutter]
        for column in (left, right):
            text = _lines_to_text(column)
            if text:
                page_parts.append(text)
        page_parts.append(anchor.text)

    final_region = regions[-1]
    for column in (
        [line for line in final_region if line.x1 <= gutter],
        [line for line in final_region if line.x0 >= gutter],
    ):
        text = _lines_to_text(column)
        if text:
            page_parts.append(text)

    return "\n\n".join(part for part in page_parts if part).strip()


def _extract_layout_pages(document: pymupdf.Document) -> list[str] | None:
    """Run PyMuPDF's layout model and return one text string per page."""
    try:
        chunks = document.to_text(
            page_chunks=True,
            use_ocr=False,
            force_text=True,
            show_progress=False,
        )
    except Exception:
        return None

    if not isinstance(chunks, list):
        return None

    page_texts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            return None
        page_texts.append(_clean_extracted_text(str(chunk.get("text", ""))))
    return page_texts


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract resume text in natural reading order from a PDF byte string.

    PyMuPDF 2.1's layout analysis reads normal single-column and mixed-layout
    pages. For pages with a geometrically supported column gutter, positioned
    text lines take precedence over the model's output and are emitted one
    column at a time. This avoids the row-wise interleaving that can occur when
    columns are flattened into one model region. OCR is disabled so
    scanned/image-only PDFs are reported instead of silently receiving partial
    text.

    Args:
        pdf_bytes: Complete contents of a PDF file.

    Returns:
        Extracted text, with readable page and paragraph boundaries.

    Raises:
        ValueError: If the input is empty, corrupted, encrypted, has no pages,
            or has no machine-readable text.
    """
    if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)) or not pdf_bytes:
        raise ValueError("The uploaded PDF is empty or was not provided as bytes.")

    document: pymupdf.Document | None = None
    try:
        try:
            document = pymupdf.open(stream=bytes(pdf_bytes), filetype="pdf")
        except Exception as exc:
            raise ValueError(
                "Unable to open the uploaded PDF. It may be corrupted or not a valid PDF."
            ) from exc

        if document.is_encrypted or document.needs_pass:
            raise ValueError(
                "The uploaded PDF is encrypted or password-protected; provide an unlocked PDF."
            )

        try:
            page_count = document.page_count
        except Exception as exc:
            raise ValueError("The uploaded PDF is corrupted or unreadable.") from exc

        if page_count == 0:
            raise ValueError("The uploaded PDF contains no pages.")

        layout_pages = _extract_layout_pages(document)
        extracted_pages: list[str] = []
        for page_number in range(page_count):
            try:
                page = document.load_page(page_number)
                positioned_lines = _get_page_lines(page)
                gutter = _find_fallback_gutter(
                    positioned_lines, page.rect.width, page.rect.height
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not read page {page_number + 1}; the PDF may be corrupted."
                ) from exc

            # A detected gutter is explicit evidence that this page has
            # columns. Use line-level coordinates to keep those streams apart,
            # even if the layout model placed both columns in one text region.
            if gutter is not None:
                column_text = _clean_extracted_text(
                    _extract_page_by_geometry(
                        page, lines=positioned_lines, gutter=gutter
                    )
                )
                if column_text:
                    extracted_pages.append(column_text)
                    continue

            layout_text = (
                layout_pages[page_number]
                if layout_pages is not None and page_number < len(layout_pages)
                else ""
            )

            if layout_text:
                extracted_pages.append(layout_text)
                continue

            fallback_text = _clean_extracted_text(
                _extract_page_by_geometry(page, lines=positioned_lines)
            )

            if fallback_text:
                extracted_pages.append(fallback_text)
            else:
                extracted_pages.append("")

        extracted_text = "\n\n".join(text for text in extracted_pages if text).strip()
        if not extracted_text:
            raise ValueError(
                "The PDF contains no extractable text. It may be empty or scanned/image-only; "
                "OCR is not enabled in this extractor."
            )

        return extracted_text
    finally:
        if document is not None:
            document.close()
