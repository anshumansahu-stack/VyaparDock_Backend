import fitz
import io

def blocks_overlap_vertically(b1, b2):
    # b[1] = y0 (top), b[3] = y1 (bottom)
    return b1[1] < b2[3] and b2[1] < b1[3]

def find_column_start_y(text_blocks):
    for b in text_blocks:
        overlapping = [
            other for other in text_blocks
            if other is not b
            and blocks_overlap_vertically(b, other)
            and other[0] != b[0]  # different x0, i.e. genuinely side-by-side
        ]
        if overlapping:
            return min(b[1], min(o[1] for o in overlapping))
    return 0

def cluster_into_two_columns(body_blocks):
    if not body_blocks:
        return [], []

    centers = [(b[0] + b[2]) / 2 for b in body_blocks]
    c_left, c_right = min(centers), max(centers)

    left_group, right_group = [], []
    for _ in range(10):
        left_group, right_group = [], []
        for b, c in zip(body_blocks, centers):
            if abs(c - c_left) <= abs(c - c_right):
                left_group.append(b)
            else:
                right_group.append(b)

        if left_group:
            c_left = sum((b[0] + b[2]) / 2 for b in left_group) / len(left_group)
        if right_group:
            c_right = sum((b[0] + b[2]) / 2 for b in right_group) / len(right_group)

    return sorted(left_group, key=lambda b: b[1]), sorted(right_group, key=lambda b: b[1])


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    for page in doc:
        blocks = page.get_text("blocks")
        text_blocks = sorted([b for b in blocks if b[6] == 0], key=lambda b: b[1])

        if not text_blocks:
            continue

        column_start_y = find_column_start_y(text_blocks)

        header_blocks = [b for b in text_blocks if b[1] < column_start_y]
        body_blocks = [b for b in text_blocks if b[1] >= column_start_y]

        left_col, right_col = cluster_into_two_columns(body_blocks)

        for b in header_blocks + left_col + right_col:
            text_parts.append(b[4].strip())

    doc.close()
    return "\n".join(text_parts)