"""Layout reconstruction shared by digital extraction and OCR.

Plain ``page.get_text()`` returns text in content-stream order, which for many
invoices scrambles tables (see the QualityHosting or Coolblue samples: quantity,
description and amount of one row come out far apart). We instead take word
boxes (from PyMuPDF or Tesseract), group them into visual rows by their vertical
position, and join each row left-to-right, inserting extra spaces for wide gaps.
The result keeps "description ... qty ... value" of a table row on one line,
which preserves the line-item relationship for the LLM and the validators.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WordBox:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    conf: float | None = None  # OCR confidence (0-100), None for digital text

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 0.1)


def group_into_lines(words: list[WordBox]) -> list[list[WordBox]]:
    """Cluster words into visual rows by vertical overlap."""
    words = [w for w in words if w.text and w.text.strip()]
    if not words:
        return []
    words.sort(key=lambda w: (w.yc, w.x0))
    lines: list[list[WordBox]] = []
    current: list[WordBox] = [words[0]]
    cur_yc, cur_h = words[0].yc, words[0].h
    for w in words[1:]:
        # Same row if vertical centres are within ~half a line height.
        if abs(w.yc - cur_yc) <= 0.5 * max(cur_h, w.h):
            current.append(w)
            n = len(current)
            cur_yc = (cur_yc * (n - 1) + w.yc) / n
            cur_h = max(cur_h, w.h)
        else:
            lines.append(current)
            current, cur_yc, cur_h = [w], w.yc, w.h
    lines.append(current)
    return [sorted(line, key=lambda w: w.x0) for line in lines]


def render_line(line: list[WordBox]) -> str:
    """Join a row's words, turning large horizontal gaps into runs of spaces."""
    if not line:
        return ""
    char_w = sum((w.x1 - w.x0) / max(len(w.text), 1) for w in line) / len(line)
    char_w = max(char_w, 0.5)
    out = line[0].text
    for prev, w in zip(line, line[1:]):
        gap = w.x0 - prev.x1
        if gap > 1.5 * char_w:
            # >= 2 spaces marks a column break; keep rough proportional spacing.
            out += " " * min(max(2, round(gap / char_w)), 12)
        else:
            out += " "
        out += w.text
    return out


def line_segments(line: list[WordBox]) -> list[tuple[float, float, str]]:
    """Merge a row's words into cell-like segments split at column gaps."""
    if not line:
        return []
    char_w = max(sum((w.x1 - w.x0) / max(len(w.text), 1) for w in line) / len(line), 0.5)
    segs: list[list] = [[line[0].x0, line[0].x1, line[0].text]]
    for prev, w in zip(line, line[1:]):
        if w.x0 - prev.x1 > 1.5 * char_w:
            segs.append([w.x0, w.x1, w.text])
        else:
            segs[-1][1] = w.x1
            segs[-1][2] += " " + w.text
    return [(a, b, t) for a, b, t in segs]


def layout(words: list[WordBox]) -> tuple[str, list[list[tuple[float, float, str]]]]:
    """Return (layout text, rows of segments) for a page's words."""
    lines = group_into_lines(words)
    return "\n".join(render_line(l) for l in lines), [line_segments(l) for l in lines]


def words_to_text(words: list[WordBox]) -> str:
    return layout(words)[0]
