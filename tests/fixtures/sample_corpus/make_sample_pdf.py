"""Generates permafrost.pdf: a minimal, hand-built two-page PDF fixture.

Written by hand (no PDF library) so the test corpus doesn't need a PDF
authoring dependency just to produce a fixture. Uncompressed content
streams keep the whole file plain ASCII and the xref offsets easy to
verify. Re-run this script to regenerate permafrost.pdf if it's ever
edited.
"""

from __future__ import annotations

from pathlib import Path

PAGE_TEXTS = [
    [
        "Permafrost is ground that stays frozen for at least two",
        "consecutive years, often for centuries. It underlies large",
        "areas of the Arctic, Subarctic, and high-altitude regions.",
    ],
    [
        "As permafrost thaws, stored organic carbon becomes available",
        "to microbial decomposition, which can release carbon dioxide",
        "and methane into the atmosphere.",
    ],
]


def _content_stream(lines: list[str]) -> bytes:
    ops = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
    for i, line in enumerate(lines):
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        if i > 0:
            ops.append("T*")
        ops.append(f"({escaped}) Tj")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    return stream


def build_pdf(pages: list[list[str]]) -> bytes:
    objects: list[bytes] = [b""]  # 1-indexed; objects[0] unused

    num_pages = len(pages)
    catalog_obj_num = 1
    pages_obj_num = 2
    font_obj_num = 3
    # Page objects: 4 .. 4 + num_pages - 1
    # Content stream objects follow immediately after.
    page_obj_nums = [4 + i for i in range(num_pages)]
    content_obj_nums = [4 + num_pages + i for i in range(num_pages)]

    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    objects.append(
        f"<< /Type /Catalog /Pages {pages_obj_num} 0 R >>".encode("latin-1")
    )  # 1: catalog
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>".encode("latin-1")
    )  # 2: pages
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )  # 3: font

    for i in range(num_pages):
        objects.append(
            (
                f"<< /Type /Page /Parent {pages_obj_num} 0 R "
                f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
                f"/MediaBox [0 0 612 792] /Contents {content_obj_nums[i]} 0 R >>"
            ).encode("latin-1")
        )  # page object

    for i, lines in enumerate(pages):
        stream = _content_stream(lines)
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )  # content object

    header = b"%PDF-1.4\n"
    body_parts: list[bytes] = []
    offsets: list[int] = [0]  # offsets[0] is unused (object 0 is the free-list head)
    cursor = len(header)

    for obj_num in range(1, len(objects)):
        offsets.append(cursor)
        obj_bytes = f"{obj_num} 0 obj\n".encode("latin-1") + objects[obj_num] + b"\nendobj\n"
        body_parts.append(obj_bytes)
        cursor += len(obj_bytes)

    body = b"".join(body_parts)
    xref_offset = len(header) + len(body)

    num_objects = len(objects)  # includes the unused index 0 slot
    xref_lines = [f"xref", f"0 {num_objects}", "0000000000 65535 f "]
    for obj_num in range(1, num_objects):
        xref_lines.append(f"{offsets[obj_num]:010d} 00000 n ")
    xref = ("\n".join(xref_lines) + "\n").encode("latin-1")

    trailer = (
        f"trailer\n<< /Size {num_objects} /Root {catalog_obj_num} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")

    return header + body + xref + trailer


if __name__ == "__main__":
    pdf_bytes = build_pdf(PAGE_TEXTS)
    output_path = Path(__file__).parent / "permafrost.pdf"
    output_path.write_bytes(pdf_bytes)
    print(f"Wrote {output_path} ({len(pdf_bytes)} bytes, {len(PAGE_TEXTS)} pages)")
