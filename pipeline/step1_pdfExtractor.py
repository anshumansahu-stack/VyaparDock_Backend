"""Layout-aware text extraction for resume PDFs.

The extractor uses PyMuPDF for fast text and span geometry, then orders the
positioned lines by page region and detected columns. It intentionally does
not perform OCR: image-only PDFs are reported to the caller.

Dependency: ``pip install pymupdf``
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import isfinite
import re
import sys
import unicodedata
from typing import Any

try:  # PyMuPDF's current import name, with compatibility for older installs.
    import pymupdf  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on the installed PyMuPDF version
    try:
        import fitz as pymupdf  # type: ignore[no-redef,import-not-found]
    except ImportError:  # pragma: no cover - use pdfplumber below if available
        pymupdf = None  # type: ignore[assignment]

try:
    import pdfplumber  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional fallback dependency
    pdfplumber = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class _TextLine:
    """A positioned text fragment, usually one visual line."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    index: int
    soft_hyphen_end: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


def _normalise_line(text: str) -> str:
    """Apply Unicode fixes and remove layout artifacts from one visual line."""
    text = unicodedata.normalize("NFKC", text).replace("\xa0", " ")
    text = (
        text.replace("\x00", "")
        .replace("\u00ad", "")
        .replace("\ufffe", "")
        .replace("\u200b", "")
    )
    # Private-use glyphs are often custom bullet icons with no portable meaning.
    text = "".join(char for char in text if unicodedata.category(char) != "Co")
    # Remove invisible formatting controls that otherwise split or join words.
    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Cf" or char in "\u200c\u200d"
    )
    return re.sub(r"[\t\f\v ]+", " ", text).strip()


def _clean_extracted_text(text: str) -> str:
    """Normalize page/document separators without flattening paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalise_line(line) for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
    return text.strip()


def _valid_bbox(raw_bbox: Any) -> tuple[float, float, float, float] | None:
    """Return a finite, non-empty rectangle or ``None`` for malformed data."""
    try:
        x0, y0, x1, y1 = (float(value) for value in raw_bbox[:4])
    except (TypeError, ValueError, IndexError):
        return None
    if not all(isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _span_text(
    spans: list[tuple[float, float, float, float, dict[str, Any]]]
) -> tuple[str, bool]:
    """Rebuild styled spans and preserve an end-of-line soft-hyphen signal."""
    parts: list[str] = []
    previous_x1: float | None = None
    previous_text = ""

    for x0, _y0, x1, _y1, span in spans:
        text = str(span.get("text", ""))
        if not text:
            continue
        try:
            font_size = float(span.get("size", 10.0))
        except (TypeError, ValueError):
            font_size = 10.0
        if (
            parts
            and previous_x1 is not None
            and x0 - previous_x1 > max(1.0, font_size * 0.18)
            and not previous_text[-1:].isspace()
            and not text[:1].isspace()
        ):
            parts.append(" ")
        parts.append(text)
        previous_x1 = x1
        previous_text = text

    raw_text = "".join(parts)
    soft_hyphen_end = raw_text.rstrip().endswith(("\xad", "\ufffe", "\u200b"))
    return _normalise_line(raw_text), soft_hyphen_end


def _get_page_lines(page: Any) -> list[_TextLine]:
    """Extract text spans and split separate same-baseline text boxes."""
    page_width = float(page.rect.width)
    try:
        # A dict extraction normally includes image payloads. They are not used
        # here, so omit them to keep large Canva-style pages lightweight.
        flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
        page_dict = page.get_text("dict", sort=False, flags=flags)
    except AttributeError:  # Older PyMuPDF releases may not expose these flags.
        page_dict = page.get_text("dict", sort=False)
    lines: list[_TextLine] = []
    next_index = 0

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans: list[tuple[float, float, float, float, dict[str, Any]]] = []
            font_sizes: list[float] = []
            for span in line.get("spans", []):
                bbox = _valid_bbox(span.get("bbox"))
                if bbox is None or not span.get("text"):
                    continue
                spans.append((*bbox, span))
                try:
                    font_sizes.append(float(span.get("size", 10.0)))
                except (TypeError, ValueError):
                    pass

            spans.sort(key=lambda item: (item[0], item[1]))
            if not spans:
                continue

            typical_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0
            # A modest, size-aware threshold keeps normal styled phrases together
            # while separating distant text boxes laid on the same baseline.
            # Keep the split below a typical resume column gutter. Wider gaps
            # within a logical line are rejoined later when they share a baseline.
            split_gap = max(5.5, page_width * 0.008, typical_size * 0.75)
            groups: list[list[tuple[float, float, float, float, dict[str, Any]]]] = []
            for span in spans:
                if groups and span[0] - max(item[2] for item in groups[-1]) <= split_gap:
                    groups[-1].append(span)
                else:
                    groups.append([span])

            for group in groups:
                text, soft_hyphen_end = _span_text(group)
                if not text:
                    continue
                x0 = min(item[0] for item in group)
                y0 = min(item[1] for item in group)
                x1 = max(item[2] for item in group)
                y1 = max(item[3] for item in group)
                if x1 <= x0 or y1 <= y0:
                    continue
                lines.append(
                    _TextLine(x0, y0, x1, y1, text, next_index, soft_hyphen_end)
                )
                next_index += 1

    return lines


def _get_page_lines_pdfplumber(page: Any) -> list[_TextLine]:
    """Character-box fallback for environments without PyMuPDF.

    pdfplumber is backed by pdfminer.six. Its character-level coordinates are
    useful when a page's text lines were emitted as separate, slightly offset
    glyph runs (for example, superscripts in LaTeX resumes).
    """
    page_width = float(page.width)
    chars: list[dict[str, Any]] = []
    for char in getattr(page, "chars", []):
        text = str(char.get("text", ""))
        try:
            x0 = float(char["x0"])
            top = float(char["top"])
            x1 = float(char["x1"])
            bottom = float(char["bottom"])
        except (KeyError, TypeError, ValueError):
            continue
        if not text or not all(isfinite(value) for value in (x0, top, x1, bottom)):
            continue
        if x1 < x0 or bottom <= top:
            continue
        char_copy = dict(char)
        char_copy.update(x0=x0, top=top, x1=x1, bottom=bottom, text=text)
        chars.append(char_copy)

    if not chars:
        return []

    # Cluster around the median glyph center of each row. Do not compare against
    # a row's union bbox: descenders and superscripts can make that bbox span into
    # the next actual line and accidentally join the two rows.
    rows: list[list[dict[str, Any]]] = []
    for char in sorted(chars, key=lambda item: (item["bottom"], item["x0"])):
        char_height = char["bottom"] - char["top"]
        char_center = (char["top"] + char["bottom"]) / 2.0
        try:
            char_size = float(char.get("size", char_height))
        except (TypeError, ValueError):
            char_size = char_height
        target: list[dict[str, Any]] | None = None
        for row in reversed(rows[-4:]):
            centers = sorted(
                (item["top"] + item["bottom"]) / 2.0 for item in row
            )
            row_center = centers[len(centers) // 2]
            row_sizes: list[float] = []
            for item in row:
                try:
                    row_sizes.append(float(item.get("size", item["bottom"] - item["top"])))
                except (TypeError, ValueError):
                    pass
            row_size = sorted(row_sizes)[len(row_sizes) // 2] if row_sizes else char_height
            tolerance = max(2.2, min(row_size, char_size) * 0.52)
            if abs(char_center - row_center) <= tolerance:
                target = row
                break
        if target is None:
            rows.append([char])
        else:
            target.append(char)

    lines: list[_TextLine] = []
    next_index = 0
    for row in rows:
        row.sort(key=lambda item: (item["x0"], item["top"]))
        sizes: list[float] = []
        for char in row:
            try:
                sizes.append(float(char.get("size", 10.0)))
            except (TypeError, ValueError):
                pass
        typical_size = sorted(sizes)[len(sizes) // 2] if sizes else 10.0
        split_gap = max(5.5, page_width * 0.008, typical_size * 0.75)
        groups: list[list[dict[str, Any]]] = []
        for char in row:
            if groups and char["x0"] - max(item["x1"] for item in groups[-1]) <= split_gap:
                groups[-1].append(char)
            else:
                groups.append([char])

        for group in groups:
            group.sort(key=lambda item: item["x0"])
            parts: list[str] = []
            previous: dict[str, Any] | None = None
            for char in group:
                text = char["text"]
                if previous is not None and parts:
                    gap = char["x0"] - previous["x1"]
                    size = float(previous.get("size", typical_size))
                    if (
                        gap > max(1.0, size * 0.16)
                        and not previous["text"].isspace()
                        and not text.isspace()
                    ):
                        parts.append(" ")
                parts.append(text)
                previous = char

            raw_text = "".join(parts)
            text = _normalise_line(raw_text)
            if not text:
                continue
            lines.append(
                _TextLine(
                    min(char["x0"] for char in group),
                    min(char["top"] for char in group),
                    max(char["x1"] for char in group),
                    max(char["bottom"] for char in group),
                    text,
                    next_index,
                    raw_text.rstrip().endswith(("\xad", "\ufffe", "\u200b")),
                )
            )
            next_index += 1
    return lines


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping horizontal intervals."""
    if not intervals:
        return []
    result: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _overlap_support(
    left: list[_TextLine], right: list[_TextLine]
) -> tuple[float, int, int]:
    """Measure vertical co-occurrence for lines on opposing sides of a cut."""
    intersections: list[tuple[float, float]] = []
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    for left_line in left:
        for right_line in right:
            top = max(left_line.y0, right_line.y0)
            bottom = min(left_line.y1, right_line.y1)
            overlap = bottom - top
            min_height = min(left_line.height, right_line.height)
            if min_height <= 0 or overlap / min_height < 0.18:
                continue
            intersections.append((top, bottom))
            matched_left.add(left_line.index)
            matched_right.add(right_line.index)

    merged_y = _merge_intervals(intersections)
    support = sum(end - start for start, end in merged_y)
    return support, len(matched_left), len(matched_right)


def _find_columns(lines: list[_TextLine], page_width: float, page_height: float) -> list[float]:
    """Find stable gutters without allowing centered headers to hide them.

    Candidate cuts are generated from horizontally separated, vertically
    co-occurring line fragments. Each cut is scored by two-sided vertical
    support and penalized when many text lines cross it. This lets short
    centered headers and page numbers cross the gutter without merging the
    page-wide x projection.
    """
    if page_width <= 0 or page_height <= 0 or len(lines) < 4:
        return []

    minimum_gutter = max(7.0, page_width * 0.012)
    minimum_support = max(10.0, page_height * 0.012)
    candidate_cuts: set[float] = set()

    for left_index, first in enumerate(lines):
        for second in lines[left_index + 1 :]:
            min_height = min(first.height, second.height)
            vertical_overlap = min(first.y1, second.y1) - max(first.y0, second.y0)
            if min_height <= 0 or vertical_overlap / min_height < 0.18:
                continue
            if first.x1 + minimum_gutter <= second.x0:
                midpoint = (first.x1 + second.x0) / 2.0
                candidate_cuts.add(round(midpoint / 2.0) * 2.0)
            elif second.x1 + minimum_gutter <= first.x0:
                midpoint = (second.x1 + first.x0) / 2.0
                candidate_cuts.add(round(midpoint / 2.0) * 2.0)

    scored: list[tuple[float, float]] = []
    for cut in candidate_cuts:
        if cut < page_width * 0.06 or cut > page_width * 0.94:
            continue
        left = [line for line in lines if line.x1 <= cut]
        right = [line for line in lines if line.x0 >= cut]
        if len(left) < 2 or len(right) < 2:
            continue
        support, matched_left, matched_right = _overlap_support(left, right)
        if support < minimum_support or matched_left < 2 or matched_right < 2:
            continue

        crossing_height = sum(
            line.height for line in lines if line.x0 < cut < line.x1
        )
        balance = min(matched_left, matched_right) ** 0.5
        score = support * balance / (1.0 + crossing_height / max(support, 1.0))
        scored.append((cut, score))

    if not scored:
        return []

    # Several candidate coordinates can fall inside the same whitespace gutter.
    # Consolidate those into a single boundary while retaining 3+ column grids.
    scored.sort(key=lambda item: item[0])
    clusters: list[list[tuple[float, float]]] = []
    merge_distance = max(6.0, page_width * 0.012)
    for item in scored:
        if clusters and item[0] - clusters[-1][-1][0] <= merge_distance:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    cuts = [max(cluster, key=lambda item: item[1])[0] for cluster in clusters]

    # Reject weak cuts that split a single text area at an incidental whitespace
    # gap. A real column gutter has distinct vertical text on both sides.
    accepted: list[float] = []
    for cut in cuts:
        left = [line for line in lines if line.x1 <= cut]
        right = [line for line in lines if line.x0 >= cut]
        support, matched_left, matched_right = _overlap_support(left, right)
        crossing_height = sum(line.height for line in lines if line.x0 < cut < line.x1)
        if (
            support >= minimum_support
            and matched_left >= 2
            and matched_right >= 2
            and crossing_height <= max(page_height * 0.12, support * 0.8)
        ):
            accepted.append(cut)
    return accepted


def _merge_same_baseline(lines: list[_TextLine]) -> list[_TextLine]:
    """Join fragments that share a visual baseline inside the same column."""
    ordered = sorted(lines, key=lambda line: (line.y0, line.x0, line.index))
    rows: list[list[_TextLine]] = []
    for line in ordered:
        target: list[_TextLine] | None = None
        for row in reversed(rows[-3:]):
            row_top = min(item.y0 for item in row)
            row_bottom = max(item.y1 for item in row)
            overlap = min(row_bottom, line.y1) - max(row_top, line.y0)
            min_height = min(row_bottom - row_top, line.height)
            center_delta = abs((row_top + row_bottom - line.y0 - line.y1) / 2.0)
            if min_height > 0 and (
                overlap / min_height >= 0.30
                or center_delta <= max(2.0, min_height * 0.38)
            ):
                target = row
                break
        if target is None:
            rows.append([line])
        else:
            target.append(line)

    merged: list[_TextLine] = []
    for row in rows:
        row.sort(key=lambda line: (line.x0, line.index))
        # These are separate positioned lines/blocks on one baseline. Keep an
        # explicit lexical boundary even if their boxes touch.
        row_text = " ".join(line.text.strip() for line in row if line.text.strip())
        merged.append(
            _TextLine(
                min(line.x0 for line in row),
                min(line.y0 for line in row),
                max(line.x1 for line in row),
                max(line.y1 for line in row),
                _normalise_line(row_text),
                min(line.index for line in row),
                row[-1].soft_hyphen_end,
            )
        )
    return sorted(merged, key=lambda line: (line.y0, line.x0, line.index))


def _lines_to_text(lines: list[_TextLine]) -> str:
    """Render one column in top-to-bottom order with paragraph spacing."""
    ordered = _merge_same_baseline(lines)
    if not ordered:
        return ""
    right_edges = sorted(line.x1 for line in ordered)
    right_edge = right_edges[int((len(right_edges) - 1) * 0.90)]
    typical_height = sorted(line.height for line in ordered)[len(ordered) // 2]
    edge_tolerance = max(4.0, typical_height * 0.65)

    # Join explicit discretionary markers and hard hyphens placed at the right
    # edge of a column. This repairs wrapped words without joining normal prose.
    dehyphenated: list[_TextLine] = []
    for line in ordered:
        if dehyphenated:
            previous = dehyphenated[-1]
            starts_lowercase = bool(line.text) and line.text.lstrip()[:1].islower()
            explicit_marker = previous.soft_hyphen_end
            ending = previous.text.rstrip()
            hard_wrap = (
                ending.endswith(("-", "\u2010", "\u2011"))
                and previous.x1 >= right_edge - edge_tolerance
            )
            if starts_lowercase and (explicit_marker or hard_wrap):
                if hard_wrap and not explicit_marker:
                    ending = ending[:-1]
                dehyphenated[-1] = _TextLine(
                    min(previous.x0, line.x0),
                    previous.y0,
                    max(previous.x1, line.x1),
                    line.y1,
                    _normalise_line(ending + line.text.lstrip()),
                    previous.index,
                    line.soft_hyphen_end,
                )
                continue
        dehyphenated.append(line)

    result = [dehyphenated[0].text]
    previous = dehyphenated[0]
    for line in dehyphenated[1:]:
        gap = line.y0 - previous.y1
        separator = "\n\n" if gap > max(5.0, previous.height * 0.75) else "\n"
        result.extend((separator, line.text))
        previous = line
    return "".join(result).strip()


def _region_index(line: _TextLine, anchors: list[_TextLine]) -> int:
    """Place a column line before or after vertically spanning lines."""
    region = 0
    line_mid = (line.y0 + line.y1) / 2.0
    for anchor in anchors:
        if line.y0 >= anchor.y1:
            region += 1
        elif line.y1 <= anchor.y0:
            break
        elif line_mid > (anchor.y0 + anchor.y1) / 2.0:
            region += 1
        else:
            break
    return region


def _column_for_line(line: _TextLine, cuts: list[float]) -> int:
    center = (line.x0 + line.x1) / 2.0
    return sum(center > cut for cut in cuts)


def _extract_page_text(page: Any, *, pdfplumber_backend: bool = False) -> str:
    """Order positioned text on a single page without mixing page columns."""
    if pdfplumber_backend:
        lines = _get_page_lines_pdfplumber(page)
        page_width = float(page.width)
        page_height = float(page.height)
    else:
        lines = _get_page_lines(page)
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
    if not lines:
        return ""
    cuts = _find_columns(lines, page_width, page_height)
    if not cuts:
        return _lines_to_text(lines)

    # Lines spanning a gutter are never sliced. Also retain centered one-line
    # headers and centered page numbers as anchors when an asymmetric layout's
    # gutter does not pass through the page center.
    def is_centered(line: _TextLine) -> bool:
        center = (line.x0 + line.x1) / 2.0
        return abs(center - page_width / 2.0) <= page_width * 0.04

    crossing_margin = max(8.0, page_width * 0.015)
    direct_anchors = [
        line
        for line in lines
        if (
            any(
                line.x0 < cut - crossing_margin
                and line.x1 > cut + crossing_margin
                for cut in cuts
            )
            or (line.y0 <= page_height * 0.17 and line.width >= page_width * 0.06 and is_centered(line))
            or (
                line.y0 >= page_height * 0.88
                and re.fullmatch(r"[\[(]?\d+[\])]?,?", line.text.strip()) is not None
                and is_centered(line)
            )
        )
    ]
    direct_anchor_ids = {line.index for line in direct_anchors}

    # Some resume headers/footers are assembled from separate left and right
    # text boxes rather than one wide span. At the page edges, join same-baseline
    # boxes that collectively cover most of the page before reading the columns.
    edge_band = max(36.0, page_height * 0.085)
    synthetic_anchors: list[_TextLine] = []
    synthetic_anchor_ids: set[int] = set()
    for row in _merge_same_baseline(lines):
        at_page_edge = row.y0 <= edge_band or row.y1 >= page_height - edge_band
        if not at_page_edge or row.width <= page_width * 0.60:
            continue
        members = [
            line
            for line in lines
            if min(line.y1, row.y1) - max(line.y0, row.y0)
            >= min(line.height, row.height) * 0.45
        ]
        sides = {_column_for_line(line, cuts) for line in members}
        if len(members) < 2 or len(sides) < 2:
            continue
        member_ids = {line.index for line in members}
        # Keep direct anchors intact to avoid duplicating their text.
        if member_ids & direct_anchor_ids:
            continue
        synthetic_anchors.append(row)
        synthetic_anchor_ids.update(member_ids)

    anchors = sorted(
        [line for line in direct_anchors if line.index not in synthetic_anchor_ids]
        + synthetic_anchors,
        key=lambda line: (line.y0, line.x0, line.index),
    )
    anchor_ids = direct_anchor_ids | synthetic_anchor_ids
    content = [line for line in lines if line.index not in anchor_ids]
    regions: list[list[_TextLine]] = [[] for _ in range(len(anchors) + 1)]
    for line in content:
        regions[_region_index(line, anchors)].append(line)

    segments: list[tuple[str, str, _TextLine | None]] = []
    for anchor_index, anchor in enumerate(anchors):
        region = regions[anchor_index]
        for column_index in range(len(cuts) + 1):
            column = [line for line in region if _column_for_line(line, cuts) == column_index]
            text = _lines_to_text(column)
            if text:
                segments.append(("column", text, None))
        if anchor.text:
            segments.append(("anchor", anchor.text, anchor))

    final_region = regions[-1]
    for column_index in range(len(cuts) + 1):
        column = [line for line in final_region if _column_for_line(line, cuts) == column_index]
        text = _lines_to_text(column)
        if text:
            segments.append(("column", text, None))

    output: list[str] = []
    previous_kind: str | None = None
    previous_anchor: _TextLine | None = None
    for kind, text, anchor in segments:
        if output:
            separator = "\n\n"
            if kind == previous_kind == "anchor" and previous_anchor is not None and anchor is not None:
                vertical_gap = anchor.y0 - previous_anchor.y1
                if vertical_gap <= max(14.0, previous_anchor.height * 1.2):
                    separator = "\n"
            output.append(separator)
        output.append(text)
        previous_kind = kind
        previous_anchor = anchor
    return "".join(output).strip()


def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    """Run the same coordinate ordering through pdfplumber's character boxes."""
    if pdfplumber is None:
        raise RuntimeError(
            "Install PyMuPDF (recommended) or pdfplumber to extract PDF text."
        )
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as document:
            if not document.pages:
                raise ValueError("The PDF contains no pages.")
            pages = [
                _clean_extracted_text(
                    _extract_page_text(page, pdfplumber_backend=True)
                )
                for page in document.pages
            ]
    except ValueError:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "password" in message or "encrypt" in message:
            raise ValueError(
                "The PDF is encrypted or password-protected; provide an unlocked PDF."
            ) from exc
        raise ValueError(
            "Unable to read the PDF; it may be corrupted or password-protected."
        ) from exc

    result = _clean_extracted_text("\n\n".join(text for text in pages if text))
    if not result:
        raise ValueError(
            "The PDF contains no extractable text. It may be empty or scanned/image-only; "
            "OCR is not enabled in this extractor."
        )
    return result


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract readable text from a machine-readable PDF byte string.

    Pages are processed independently, in their original order. Each page is
    split into layout regions at full-width lines and columns at geometrically
    supported whitespace gutters. Unicode is normalized to NFKC and private-
    use glyphs are removed.

    Args:
        pdf_bytes: Complete contents of a PDF file.

    Returns:
        Extracted text with line and paragraph breaks.

    Raises:
        ValueError: If the input is empty, invalid, encrypted, unreadable, or
            contains no extractable text. Scanned PDFs require a separate OCR
            step and are intentionally rejected here.
    """
    if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)) or not pdf_bytes:
        raise ValueError("PDF input is empty or was not provided as bytes.")

    pdf_data = bytes(pdf_bytes)
    if pymupdf is None:
        return _extract_with_pdfplumber(pdf_data)

    document = None
    try:
        try:
            document = pymupdf.open(stream=pdf_data, filetype="pdf")
        except Exception as exc:
            raise ValueError(
                "Unable to open the PDF; it may be corrupted or not a valid PDF."
            ) from exc

        if getattr(document, "is_encrypted", False) or getattr(document, "needs_pass", False):
            raise ValueError(
                "The PDF is encrypted or password-protected; provide an unlocked PDF."
            )

        try:
            page_count = int(document.page_count)
        except Exception as exc:
            raise ValueError("The PDF is corrupted or unreadable.") from exc
        if page_count <= 0:
            raise ValueError("The PDF contains no pages.")

        page_texts: list[str] = []
        for page_number in range(page_count):
            try:
                page = document.load_page(page_number)
                page_texts.append(_clean_extracted_text(_extract_page_text(page)))
            except Exception as exc:
                raise ValueError(
                    f"Could not read PDF page {page_number + 1}; the file may be corrupted."
                ) from exc

        result = _clean_extracted_text("\n\n".join(text for text in page_texts if text))
        if not result:
            raise ValueError(
                "The PDF contains no extractable text. It may be empty or scanned/image-only; "
                "OCR is not enabled in this extractor."
            )
        return result
    finally:
        if document is not None:
            document.close()


def _main(argv: list[str] | None = None) -> int:
    """CLI entry point: print extracted text for one local PDF file."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("Usage: python pdf_extractor.py <pdf_path>", file=sys.stderr)
        return 2
    try:
        with open(args[0], "rb") as pdf_file:
            extracted = extract_text_from_pdf(pdf_file.read())
    except (OSError, ValueError) as exc:
        print(f"PDF extraction failed: {exc}", file=sys.stderr)
        return 1
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    sys.stdout.write(extracted)
    if extracted and not extracted.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
