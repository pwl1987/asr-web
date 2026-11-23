"""Markdown → DOCX（会议材料导出用，覆盖常见标题/列表/粗体/表格/引用）。"""
import io
import re

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


def _set_east_asian_font(style, name: str) -> None:
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    fonts.set(qn("w:eastAsia"), name)


def _add_inline(paragraph, text: str) -> None:
    """粗体/行内代码/链接的极简渲染。"""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    for part in re.split(r"(\*\*.+?\*\*|`.+?`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
        else:
            paragraph.add_run(part)


def _add_table(doc: Document, lines: list[str]) -> None:
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
            continue  # markdown 表格分隔行
        rows.append(cells)
    if not rows:
        return
    cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=cols)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j in range(cols):
            cell = table.cell(i, j)
            cell.text = ""
            _add_inline(cell.paragraphs[0], row[j] if j < len(row) else "")


def markdown_to_docx(md: str, title: str | None = None) -> bytes:
    doc = Document()
    _set_east_asian_font(doc.styles["Normal"], "微软雅黑")
    doc.styles["Normal"].font.size = Pt(10.5)
    if title:
        doc.core_properties.title = title

    lines = (md or "").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        line = raw.strip()
        if line.startswith("|") and line.endswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            _add_table(doc, block)
            continue
        i += 1
        if not line:
            doc.add_paragraph()
            continue
        if line.startswith("```") or re.fullmatch(r"[-*_]{3,}", line):
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            doc.add_heading(m.group(2).strip(), level=min(level, 4))
            continue
        if re.match(r"^[-*+]\s+", line):
            _add_inline(doc.add_paragraph(style="List Bullet"), re.sub(r"^[-*+]\s+", "", line))
            continue
        if re.match(r"^\d+[.)]\s+", line):
            _add_inline(doc.add_paragraph(style="List Number"),
                        re.sub(r"^\d+[.)]\s+", "", line))
            continue
        if line.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(18)
            _add_inline(p, line.lstrip("> ").strip())
            continue
        _add_inline(doc.add_paragraph(), line)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
