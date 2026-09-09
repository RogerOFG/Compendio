"""
book-extractor -- microservicio FastAPI para Compendio.

Reutiliza tres módulos REALES del repo book-to-skill (github.com/virgiliojr94/book-to-skill),
sin traducir ni reescribir su lógica:
  1. book_to_skill/parsers/pdf.py   -> extracción de PDF (pdftotext -layout / pypdf / pdfminer,
                                        con respaldo entre sí, limpieza de encabezados/pies de
                                        página repetidos y de-hyphenation).
  2. book_to_skill/sanitize.py      -> quita caracteres Unicode invisibles usados para esconder
                                        inyección de prompts en un documento (espacios de ancho
                                        cero, controles de direccionalidad, bloque de "tags", etc).
  3. book_to_skill/utils.py         -> detect_structure() / _chapter_number() (detección de
                                        capítulos en espanol/ingles/latin, romanos, chino, tailandés,
                                        hindi, bengalí, ruso, coreano, persa).

Lo único que NO viene del repo es el troceo por párrafos y el ensamblado de la respuesta HTTP --
book-to-skill no trocea para embeddings (usa grep+sed on-demand desde un agente), así que esa
parte es nueva, escrita para encajar con lo que Compendio ya espera.

Endpoints:
  POST /extract  (multipart, campo "file")  -> { text, method, pages, removed_invisible }
  POST /chunk    (json: {"text": "..."})    -> { chunks: ["...", ...], method }
  GET  /health                              -> { ok: true }
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import statistics
import sys
import tempfile
from collections import Counter

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="book-extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://n8n.americana.edu.co"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ============================================================================
# 1. book_to_skill/parsers/pdf.py -- extracción de PDF (verbatim)
# ============================================================================

_ROMAN_1_99 = r"(?=[ivxl])(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})"
_PDF_PAGE_NUM = re.compile(rf"^\s*(?:\d{{1,4}}|{_ROMAN_1_99})\s*$", re.IGNORECASE)
_PDF_HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")


def clean_pdftotext(text: str) -> str:
    pages = text.split("\f")
    if len(pages) >= 3:
        edge = Counter()
        for p in pages:
            nb = [ln.strip() for ln in p.splitlines() if ln.strip()]
            if nb:
                edge[nb[0]] += 1
                if len(nb) > 1:
                    edge[nb[-1]] += 1
        boiler = {ln for ln, c in edge.items() if c > len(pages) / 2}
        kept = []
        for p in pages:
            lines = p.splitlines()
            nb_idx = [i for i, ln in enumerate(lines) if ln.strip()]
            first = nb_idx[0] if nb_idx else None
            last = nb_idx[-1] if nb_idx else None
            for i, ln in enumerate(lines):
                if i in (first, last):
                    s = ln.strip()
                    if s in boiler or _PDF_PAGE_NUM.match(s):
                        continue
                kept.append(ln)
        text = "\n".join(kept)
    else:
        text = text.replace("\f", "\n")
    return _PDF_HYPHEN_WRAP.sub(r"\1\2", text)


def extract_with_pdftotext(pdf_path: str) -> str | None:
    if not shutil.which("pdftotext"):
        return None
    try:
        pdf_path = os.path.abspath(pdf_path)
        result = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", pdf_path, "-"],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )
        if result.returncode == 0 and result.stdout.strip():
            return clean_pdftotext(result.stdout)
    except Exception as e:
        print(f"  [warn] extract_with_pdftotext failed: {type(e).__name__}: {e}", file=sys.stderr)
    return None


def looks_image_only(pdf_path: str, pages: int = 5) -> bool:
    if not shutil.which("pdftotext"):
        return False
    try:
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", str(pages), "-enc", "UTF-8", os.path.abspath(pdf_path), "-"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        return result.returncode == 0 and not result.stdout.strip()
    except Exception:
        return False


def extract_with_pypdf(pdf_path: str) -> str | None:
    try:
        import pypdf
        text_parts = []
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    text_parts.append("")
        return clean_pdftotext("\f".join(text_parts))
    except ImportError:
        return None
    except Exception as e:
        print(f"  [warn] extract_with_pypdf failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def extract_with_pdfminer(pdf_path: str) -> str | None:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        return clean_pdftotext(text) if text else text
    except ImportError:
        return None
    except Exception as e:
        print(f"  [warn] extract_with_pdfminer failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def count_pages(pdf_path: str) -> int:
    if shutil.which("pdfinfo"):
        try:
            result = subprocess.run(["pdfinfo", os.path.abspath(pdf_path)], capture_output=True, text=True, timeout=15)
            for line in result.stdout.splitlines():
                if line.startswith("Pages:"):
                    return int(line.split(":")[1].strip())
        except Exception:
            pass
    try:
        import pypdf
        with open(pdf_path, "rb") as f:
            return len(pypdf.PdfReader(f).pages)
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        if text:
            return text.count("\f") + (0 if text.endswith("\f") else 1)
    except Exception:
        pass
    return 0


# ============================================================================
# 2. book_to_skill/sanitize.py -- caracteres invisibles / anti prompt-injection (verbatim)
# ============================================================================

_ZERO_WIDTH_CODEPOINTS = frozenset({
    0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x034F, 0x180E,
    0x2061, 0x2062, 0x2063, 0x2064,
})
_BIDI_CONTROL_CODEPOINTS = frozenset({
    0x200E, 0x200F, 0x061C, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
    0x2066, 0x2067, 0x2068, 0x2069,
})
_INVISIBLE_LETTER_CODEPOINTS = frozenset({0x115F, 0x1160, 0x3164, 0xFFA0})
_ANNOTATION_FORMAT_CODEPOINTS = frozenset({
    0x206A, 0x206B, 0x206C, 0x206D, 0x206E, 0x206F, 0xFFF9, 0xFFFA, 0xFFFB,
})
_INVISIBLE_CODEPOINTS = (
    _ZERO_WIDTH_CODEPOINTS | _BIDI_CONTROL_CODEPOINTS
    | _INVISIBLE_LETTER_CODEPOINTS | _ANNOTATION_FORMAT_CODEPOINTS
)
_TAG_BLOCK_START = 0xE0000
_TAG_BLOCK_END = 0xE007F
_VARIATION_SELECTOR_RANGES = ((0xFE00, 0xFE0F), (0xE0100, 0xE01EF))
_DEPRECATED_FORMAT_RANGE = (0x206A, 0x206F)
_ANNOTATION_CODEPOINTS = frozenset({0xFFF9, 0xFFFA, 0xFFFB})
_MUSICAL_FORMAT_RANGE = (0x1D173, 0x1D17A)


def is_invisible_codepoint(codepoint: int) -> bool:
    if codepoint in _INVISIBLE_CODEPOINTS or codepoint in _ANNOTATION_CODEPOINTS:
        return True
    if _DEPRECATED_FORMAT_RANGE[0] <= codepoint <= _DEPRECATED_FORMAT_RANGE[1]:
        return True
    if _TAG_BLOCK_START <= codepoint <= _TAG_BLOCK_END:
        return True
    if _MUSICAL_FORMAT_RANGE[0] <= codepoint <= _MUSICAL_FORMAT_RANGE[1]:
        return True
    return any(low <= codepoint <= high for low, high in _VARIATION_SELECTOR_RANGES)


def sanitize_extracted_text(text: str) -> tuple[str, int]:
    kept: list[str] = []
    removed = 0
    for character in text:
        if is_invisible_codepoint(ord(character)):
            removed += 1
            continue
        kept.append(character)
    return "".join(kept), removed


# ============================================================================
# 3. book_to_skill/utils.py -- detect_structure() / _chapter_number() (verbatim)
# ============================================================================

_EXPLICIT_CHAPTER = re.compile(
    r"^\s*(?:chapter|unit|lesson|module|lecture|part|chapitre|kapitel|cap[ií]tulo|capitolo|hoofdstuk|chương|ch\.?)\s*(?:(\d{1,2})|(?P<roman>[IVXLCDMivxlcdm]{1,7}))\b(?P<rest>.*)$",
    re.IGNORECASE,
)
_HEADING_TAIL = re.compile(r"^\s*$|^\s*[.:\-—–]|^\s+(?![a-z])")
_ROMAN_HEAD = re.compile(r"^\s*([IVXLCDM]+)\s*[:.]\s+[A-ZÀ-Þ0-9\"“(]")
_LC_MD_ROMAN = re.compile(r"^\s*#{1,6}\s+([ivxlcdm]+)\s*[:.]\s+[A-Za-zÀ-Þ\"“(]")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_MD_HEADING_PREFIX = re.compile(r"^(#{1,6}|={1,6})\s+")
_CN_NUM_VALUES = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_NUM_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_NUM_CLASS = "〇零一二两三四五六七八九十百千"
_KANGXI_NUMERAL_TRANS = {
    0x2F00: ord("一"), 0x2F06: ord("二"), 0x2F0B: ord("八"), 0x2F17: ord("十"),
}
_FW_DIGITS = "０-９"
_CN_CHAPTER = re.compile(rf"^\s*第\s*([0-9{_FW_DIGITS}{_CN_NUM_CLASS}]+)\s*[章回卷节篇讲]")
_MD_CN_HEADING = re.compile(rf"^#{{1,6}}\s+第?\s*([{_FW_DIGITS}{_CN_NUM_CLASS}]+)\s*[·、.:：章回卷节篇讲]")
_TH_DIGITS = "๐-๙"
_TH_DIGIT_MAP = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_TH_CHAPTER = re.compile(rf"^\s*(?:#{{1,6}}\s+)?(?:บทที่|ตอนที่|ภาคที่|บท|ตอน|ภาค)\s*([0-9{_TH_DIGITS}]+)\b")
_HI_DIGITS = "०-९"
_HI_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")
_HI_CHAPTER = re.compile(rf"^\s*(?:#{{1,6}}\s+)?अध्याय\s*([0-9{_HI_DIGITS}]+)\b")
_BN_DIGITS = "০-৯"
_BN_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_BN_CHAPTER = re.compile(rf"^\s*(?:#{{1,6}}\s+)?অধ্যায়\s*([0-9{_BN_DIGITS}]+)\b")
_RU_CHAPTER = re.compile(r"^\s*(?:#{1,6}\s+)?глава\s+([0-9]+)\b", re.IGNORECASE)
_KO_CHAPTER = re.compile(r"^\s*(?:#{1,6}\s+)?제\s*([0-9]+)\s*[장편절관](?:\s*의\s*[0-9]+)?(?:\s*$|[.:\-]|\s+\S)")
_FA_DIGITS = "۰-۹٠-٩"
_FA_ONES = ("اول", "دوم", "سوم", "چهارم", "پنجم", "ششم", "هفتم", "هشتم", "نهم", "دهم")
_FA_COMPOUND_ONES = ("یکم", "دوم", "سوم", "چهارم", "پنجم", "ششم", "هفتم", "هشتم", "نهم")
_FA_TEENS = ("یازدهم", "دوازدهم", "سیزدهم", "چهاردهم", "پانزدهم", "شانزدهم", "هفدهم", "هجدهم", "نوزدهم")
_FA_ONES_SET = frozenset(_FA_ONES)
_FA_SHORT_ORDINAL_TAIL = re.compile(r"^(?:$|\s|[.:\-—–：]|‌)")


def _fa_ordinal_map() -> dict[str, int]:
    m: dict[str, int] = {}
    for i, w in enumerate(_FA_ONES, 1):
        m[w] = i
    for i, w in enumerate(_FA_TEENS, 11):
        m[w] = i
    m["هیجدهم"] = 18
    m["بیستم"] = 20
    m["سی ام"] = 30
    m["سی‌ام"] = 30
    for i, w in enumerate(_FA_COMPOUND_ONES, 1):
        m[f"بیست و {w}"] = 20 + i
        m[f"سی و {w}"] = 30 + i
    return m


_FA_ORDINALS = _fa_ordinal_map()
_FA_ORDINAL_KEYS = sorted(_FA_ORDINALS, key=len, reverse=True)
_FA_LABEL_REST = re.compile(r"^\s*(?:فصل|بخش)\s+(.*)$")
_FA_DIGIT_HEAD = re.compile(rf"^([0-9{_FA_DIGITS}]+)(.*)$")
_FA_DIGIT_TAIL = re.compile(r"^(?:\s*$|[.:\-—–：]|\s+\S)")


def _fa_chapter_number(s: str) -> int | None:
    m = _FA_LABEL_REST.match(s)
    if not m:
        return None
    rest = m.group(1)
    dm = _FA_DIGIT_HEAD.match(rest)
    if dm:
        n = int(dm.group(1))
        if 1 <= n <= 99 and _FA_DIGIT_TAIL.match(dm.group(2)) is not None:
            return n
        return None
    for key in _FA_ORDINAL_KEYS:
        if not rest.startswith(key):
            continue
        tail = rest[len(key):]
        if key in _FA_ONES_SET and _FA_SHORT_ORDINAL_TAIL.match(tail) is None:
            return None
        return _FA_ORDINALS[key]
    return None


_TOC_HEADERS = (
    "table of contents", "contents", "índice", "sumário", "sumario",
    "table des matières", "inhaltsverzeichnis", "indice", "sommario", "inhoudsopgave",
)
_TOC_CJK_PATTERN = r"目[ \t　]*(?:录|錄|次)"
_TOC_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:"
    + "|".join([*(re.escape(h) for h in _TOC_HEADERS), _TOC_CJK_PATTERN])
    + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ATX_HEADING = re.compile(r"^(#{1,6}|={1,6})\s+(.+?)\s*#*$")
_SETEXT_UNDERLINE = re.compile(r"^(={2,}|-{2,})$")
_CODE_FENCE = re.compile(r"^(`{3,}|~{3,})")


def _closed_fence_line_numbers(lines: list[str]) -> set[int]:
    inside: set[int] = set()
    opener: tuple[str, int] | None = None
    for index, line in enumerate(lines):
        match = _CODE_FENCE.match(line.strip())
        if not match:
            continue
        marker = match.group(1)
        if opener is None:
            opener = (marker[0], index)
        elif marker[0] == opener[0]:
            inside.update(range(opener[1], index + 1))
            opener = None
    return inside


_MIN_NUMBERED_TITLES = 3
_MIN_NUMBERED_BODY_CHARS = 200


def _numbered_titles_are_structural(entries, heading_lines, lines) -> bool:
    if len(entries) < _MIN_NUMBERED_TITLES:
        return False
    ordered = sorted(heading_lines)
    bodies = []
    for _, index in entries:
        after = [ln for ln in ordered if ln > index]
        end = after[0] if after else len(lines)
        bodies.append(sum(len(ln) for ln in lines[index + 1:end]))
    return statistics.median(bodies) >= _MIN_NUMBERED_BODY_CHARS


def _structural_chapter_count(text: str) -> int:
    lines = text.splitlines()
    levels: dict[int, set[str]] = {}
    numbered: dict[int, list[tuple[str, int]]] = {}
    heading_lines: list[int] = []
    fenced = _closed_fence_line_numbers(lines)
    prev = ""
    for index, line in enumerate(lines):
        if index in fenced:
            prev = ""
            continue
        s = line.strip()
        if (
            _SETEXT_UNDERLINE.match(s) and prev and not _SETEXT_UNDERLINE.match(prev)
            and len(s) >= len(prev) and re.search(r"\w", prev)
        ):
            depth = 1 if s[0] == "=" else 2
            levels.setdefault(depth, set()).add(prev.lower())
            heading_lines.append(index)
            prev = ""
            continue
        m = _ATX_HEADING.match(s)
        if m:
            title = m.group(2).strip().lower()
            depth = len(m.group(1))
            if title and re.search(r"\w", title):
                heading_lines.append(index)
                if title[0].isdigit():
                    numbered.setdefault(depth, []).append((title, index))
                else:
                    levels.setdefault(depth, set()).add(title)
            prev = ""
            continue
        prev = s
    for depth, entries in numbered.items():
        if _numbered_titles_are_structural(entries, heading_lines, lines):
            levels.setdefault(depth, set()).update(title for title, _ in entries)
    if not levels:
        return 0
    for depth in sorted(levels):
        if len(levels[depth]) >= 2:
            return len(levels[depth])
    return sum(len(titles) for titles in levels.values())


def _cn_numeral_to_int(s: str) -> int | None:
    if s.isdigit():
        n = int(s)
        return n if 1 <= n <= 999 else None
    section = current = 0
    for ch in s:
        if ch in _CN_NUM_VALUES:
            current = _CN_NUM_VALUES[ch]
        elif ch in _CN_NUM_UNITS:
            section += (current or 1) * _CN_NUM_UNITS[ch]
            current = 0
        else:
            return None
    total = section + current
    return total if 1 <= total <= 999 else None


def _int_to_roman(n: int) -> str:
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
             (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
             (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for val, sym in table:
        while n >= val:
            out.append(sym)
            n -= val
    return "".join(out)


def _roman_to_int(s: str) -> int | None:
    s = s.upper()
    total = prev = 0
    for ch in reversed(s):
        v = _ROMAN_VALUES.get(ch)
        if v is None:
            return None
        total += -v if v < prev else v
        prev = max(prev, v)
    if total == 0 or total > 200:
        return None
    return total if _int_to_roman(total) == s else None


def _match_chapter_number(line: str, allow_plain: bool = True) -> int | None:
    s = line.strip().translate(_KANGXI_NUMERAL_TRANS)
    if len(s) > 80:
        return None
    if allow_plain:
        plain = re.match(r"^([1-9]\d{0,2})\s{2,}\S", s)
        if plain:
            return int(plain.group(1))
    m = _EXPLICIT_CHAPTER.match(s)
    if m and _HEADING_TAIL.match(m.group("rest")):
        if m.group(1):
            return int(m.group(1))
        return _roman_to_int(m.group("roman").upper())
    rm = _ROMAN_HEAD.match(s) or _LC_MD_ROMAN.match(s)
    if rm:
        return _roman_to_int(rm.group(1))
    cm = _CN_CHAPTER.match(s) or _MD_CN_HEADING.match(s)
    if cm:
        return _cn_numeral_to_int(cm.group(1))
    tm = _TH_CHAPTER.match(s)
    if tm:
        return int(tm.group(1).translate(_TH_DIGIT_MAP))
    hm = _HI_CHAPTER.match(s)
    if hm:
        return int(hm.group(1).translate(_HI_DIGIT_MAP))
    bm = _BN_CHAPTER.match(s)
    if bm:
        return int(bm.group(1).translate(_BN_DIGIT_MAP))
    rum = _RU_CHAPTER.match(s)
    if rum:
        return int(rum.group(1))
    km = _KO_CHAPTER.match(s)
    if km:
        return int(km.group(1))
    fa = _fa_chapter_number(s)
    if fa is not None:
        return fa
    return None


def _chapter_number(line: str, allow_plain: bool = True) -> int | None:
    match = _match_chapter_number(line, allow_plain)
    if match is not None:
        return match
    s = line.strip()
    md = _MD_HEADING_PREFIX.match(s)
    if md:
        return _match_chapter_number(s[md.end():], allow_plain)
    return None


def detect_structure(text: str) -> dict:
    lines = text.splitlines()
    headings = []
    numbers = set()
    for line in lines:
        num = _chapter_number(line)
        if num is not None:
            numbers.add(num)
            headings.append(line.strip())
    numeric_count = len(numbers)
    if numeric_count >= 2:
        chapters_detected = numeric_count
        chapters_method = "numeric"
    else:
        structural_count = _structural_chapter_count(text)
        chapters_detected = max(numeric_count, structural_count)
        chapters_method = (
            "structural" if structural_count > numeric_count
            else "numeric" if numeric_count
            else "none"
        )
    has_toc = bool(_TOC_PATTERN.search(text[:30000]))
    return {
        "chapters_detected": chapters_detected,
        "chapters_method": chapters_method,
        "chapter_headings_sample": headings[:10],
        "has_toc": has_toc,
    }


# ============================================================================
# NUEVO (no viene del repo): cortar por capítulos + trocear por párrafos.
# book-to-skill no trocea para embeddings -- su agente navega el texto completo
# con grep+sed bajo demanda. Compendio sí necesita fragmentos para el chat, así
# que esta parte se escribió para encajar con eso, reusando _chapter_number()
# como el "patrón de grep" para encontrar los límites de cada capítulo.
# ============================================================================

_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def split_into_chapters(text: str) -> list[dict] | None:
    lines = text.split("\n")
    # Prioriza señales fuertes e inequívocas (p.ej. "Capítulo N", romanos, CJK...).
    # El patrón débil "N␣␣texto" (pensado para tablas de contenido limpias) confunde
    # con facilidad pies de página de PDF que repiten el título del capítulo junto al
    # número de página (p.ej. "20  la importancia del contexto"), así que solo se usa
    # como último recurso si no hay ninguna señal fuerte en todo el documento.
    heading_idx = [i for i, line in enumerate(lines) if _chapter_number(line, allow_plain=False) is not None]
    if len(heading_idx) < 2:
        # El patrón débil solo exige un carácter no-blanco tras los espacios (puede ser
        # otro dígito), así que además exigimos que la línea tenga al menos una letra --
        # descarta ruido de tablas/gráficos con solo números (p.ej. "8   8   8" de un
        # eje numerado extraído de una figura) sin dejar de aceptar títulos reales sin
        # la palabra "Capítulo" (p.ej. "1  Una mirada al interior").
        heading_idx = [
            i for i, line in enumerate(lines)
            if _chapter_number(line, allow_plain=True) is not None and _HAS_LETTER.search(line)
        ]
    if len(heading_idx) < 2:
        return None
    chapters = []
    for idx, start in enumerate(heading_idx):
        end = heading_idx[idx + 1] if idx + 1 < len(heading_idx) else len(lines)
        chapters.append({"heading": lines[start].strip(), "text": "\n".join(lines[start:end]).strip()})
    return chapters


def find_structural_heading_lines(text: str) -> list[int] | None:
    """Respaldo Markdown: mismo escaneo que _structural_chapter_count, pero devuelve
    las líneas de corte en vez de solo el conteo."""
    lines = text.split("\n")
    level_lines: dict[int, list[int]] = {}
    fenced = _closed_fence_line_numbers(lines)
    prev, prev_index = "", -1
    for index, line in enumerate(lines):
        if index in fenced:
            prev = ""
            continue
        s = line.strip()
        if (_SETEXT_UNDERLINE.match(s) and prev and not _SETEXT_UNDERLINE.match(prev)
                and len(s) >= len(prev) and re.search(r"\w", prev)):
            depth = 1 if s[0] == "=" else 2
            level_lines.setdefault(depth, []).append(prev_index)
            prev = ""
            continue
        m = _ATX_HEADING.match(s)
        if m:
            title = m.group(2).strip()
            depth = len(m.group(1))
            if title and re.search(r"\w", title) and not title[0].isdigit():
                level_lines.setdefault(depth, []).append(index)
            prev = ""
            continue
        prev, prev_index = s, index
    for depth in sorted(level_lines):
        if len(level_lines[depth]) >= 2:
            return sorted(level_lines[depth])
    return None


def split_by_markdown_headings(text: str) -> list[dict] | None:
    lines = text.split("\n")
    heading_idx = find_structural_heading_lines(text)
    if not heading_idx:
        return None
    chapters = []
    for idx, start in enumerate(heading_idx):
        end = heading_idx[idx + 1] if idx + 1 < len(heading_idx) else len(lines)
        chapters.append({"heading": lines[start].strip(), "text": "\n".join(lines[start:end]).strip()})
    return chapters


def chunk_by_paragraphs(block_text: str, max_len: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", block_text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paragraphs:
        if len(p) > max_len:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(p), max_len):
                chunks.append(p[i:i + max_len])
            continue
        if cur and (len(cur) + 2 + len(p)) > max_len:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


MAX_CHUNK = 1600


def chunk_text(text: str) -> tuple[list[str], str]:
    chapters = split_into_chapters(text) or split_by_markdown_headings(text)
    if chapters:
        method = "chapters" if split_into_chapters(text) else "markdown"
        chunks: list[str] = []
        for ch in chapters:
            chunks.extend(chunk_by_paragraphs(ch["text"], MAX_CHUNK))
        return chunks, method
    # Respaldo: sin estructura confiable -> tamaño fijo con solape (comportamiento anterior).
    chunks = []
    chunk_size, overlap = 1600, 200
    i = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        chunks.append(text[i:end])
        if end == len(text):
            break
        i += (chunk_size - overlap)
    return chunks, "fixed-size"


# ============================================================================
# FastAPI endpoints
# ============================================================================

class ChunkRequest(BaseModel):
    text: str


class ChaptersRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se aceptan archivos .pdf en este endpoint.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if looks_image_only(tmp_path):
            raise HTTPException(422, "El PDF no tiene texto extraíble (parece escaneado/imagen). Sube uno con texto real u OCR aparte.")

        text, method = None, None
        for fn, name in (
            (extract_with_pdftotext, "pdftotext"),
            (extract_with_pypdf, "pypdf"),
            (extract_with_pdfminer, "pdfminer"),
        ):
            text = fn(tmp_path)
            if text and text.strip():
                method = name
                break

        if not text or not text.strip():
            raise HTTPException(422, "No se pudo extraer texto de este PDF con ninguno de los 3 métodos disponibles.")

        clean_text, removed = sanitize_extracted_text(text)
        pages = count_pages(tmp_path)
        return {"text": clean_text, "method": method, "pages": pages, "removed_invisible": removed}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/chunk")
def chunk(req: ChunkRequest):
    clean_text, removed = sanitize_extracted_text(req.text)
    chunks, method = chunk_text(clean_text)
    structure = detect_structure(clean_text)
    return {
        "chunks": chunks,
        "method": method,
        "removed_invisible": removed,
        "chapters_detected": structure["chapters_detected"],
    }


@app.post("/chapters")
def chapters_endpoint(req: ChaptersRequest):
    """Devuelve los capítulos reales del texto (encabezado + cuerpo completo de cada uno),
    para que un agente de IA los resuma sin tener que adivinar cuántos hay ni dónde empiezan."""
    clean_text, removed = sanitize_extracted_text(req.text)
    chapters = split_into_chapters(clean_text)
    method = "chapters"
    if not chapters:
        chapters = split_by_markdown_headings(clean_text)
        method = "markdown"
    if not chapters:
        chapters = []
        method = "none"
    return {"chapters": chapters, "method": method, "removed_invisible": removed}
