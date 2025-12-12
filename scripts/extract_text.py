#extract_text.py
from __future__ import annotations

import os, json, mimetypes, tempfile, io, re
from typing import Union, Tuple, Dict, Any, Optional

import gradio as gr

# pdfplumber
try:
    import pdfplumber
    PDF_OK = True
except ModuleNotFoundError:
    PDF_OK = False

# pypdf
try:
    from pypdf import PdfReader 
    _HAS_PYPDF = True
except Exception:
    _HAS_PYPDF = False

# OCR
try:
    import fitz 
    from PIL import Image 
    import pytesseract
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False
    
def _openable(obj: object) -> bool:
    """Return True if obj looks like a file-like (has .read and .name)."""
    return hasattr(obj, "read") and hasattr(obj, "name")


def _to_bytes_and_name(file_obj_or_path: Union[str, os.PathLike, object]) -> Tuple[bytes, str]:
    """Normalize any input (path or file-like object) into (bytes, name)."""
    if _openable(file_obj_or_path):
        return file_obj_or_path.read(), getattr(file_obj_or_path, "name", "uploaded")
    # Treat as path
    path = os.fspath(file_obj_or_path)
    with open(path, "rb") as f:
        data = f.read()
    return data, path


def extract_text(file_obj_or_path: Union[str, os.PathLike, object]) -> str:
    """
    Extract UTF-8 text from .txt or .pdf.
    Accepts file-like objects OR filesystem paths.
    """
    data, name = _to_bytes_and_name(file_obj_or_path)
    mime, _ = mimetypes.guess_type(name)

    # PDF
    if mime == "application/pdf":
        if not PDF_OK:
            raise gr.Error("pdfplumber not installed—install it (`pip install pdfplumber`) or upload a .txt file instead.")
        # pdfplumber prefers a path; write a temp file if needed
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            with pdfplumber.open(tmp_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Everything else: treat as text
    text = (
        data.decode("utf-8", errors="ignore")
        .lstrip("\ufeff")     # strip BOM
        .replace("\xa0", " ") # non-breaking spaces
    )
    return text

def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\ufeff", "").replace("\u00A0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "".join(ch for ch in s if (ch == "\n") or ch.isprintable())
    lines = [ln.rstrip() for ln in s.split("\n")]
    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _file_to_bytes(upload: Any) -> Tuple[Optional[bytes], Optional[str]]:
    if not upload:
        return None, None
    if isinstance(upload, dict):
        if isinstance(upload.get("data"), (bytes, bytearray)):
            name = upload.get("orig_name") or upload.get("name") or "uploaded"
            return bytes(upload["data"]), name
        path = upload.get("path") or upload.get("name")
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return f.read(), os.path.basename(path)
            except Exception:
                return None, None
        return None, None
    if isinstance(upload, str) and os.path.exists(upload):
        try:
            with open(upload, "rb") as f:
                return f.read(), os.path.basename(upload)
        except Exception:
            return None, None
    path = getattr(upload, "name", None)
    if isinstance(path, str) and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return f.read(), os.path.basename(path)
        except Exception:
            return None, None
    if hasattr(upload, "read"):
        try:
            data = upload.read()
            return data, os.path.basename(path) if isinstance(path, str) else None
        except Exception:
            return None, None
    return None, None

def _looks_like_pdf_source(s: str) -> bool:
    if not s:
        return False
    if s.lstrip().startswith("%PDF-"):
        return True
    sample = s[:6000]
    tokens = sum(sample.count(t) for t in (" obj", " endobj", "/Type", "/Page", "stream", "endstream", "/Contents"))
    return tokens > 10

def _ocr_pdf(file_bytes: bytes, lang: str = "eng+spa", dpi: int = 260, max_pages: int = 200) -> str:
    if not _HAS_OCR:
        return ""
    texts = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        zoom = max(72, dpi) / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pm = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            try:
                t = pytesseract.image_to_string(img, lang=lang)
            except Exception:
                try:
                    t = pytesseract.image_to_string(img, lang="eng")
                except Exception:
                    t = ""
            if t and t.strip():
                texts.append(t)
        return _clean_text("\n\n".join(texts))
    except Exception:
        return ""

def _pdf_to_text(file_bytes: bytes, *, ocr_dpi: int = 260, ocr_max_pages: int = 200) -> tuple[str, str]:
    """Returns (text, tool_used). Order: pypdf → pdfminer → pdfplumber → OCR."""
    if not file_bytes:
        return "", ""
    def _good(s: str) -> bool:
        if not s or _looks_like_pdf_source(s):
            return False
        printable = sum(ch.isprintable() for ch in s)
        letters = sum(ch.isalpha() for ch in s)
        return printable > 200 and (letters / max(len(s), 1)) > 0.18

    # 1) pypdf
    if _HAS_PYPDF:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    pass
            txt = "\n".join((p.extract_text() or "") for p in reader.pages)
            if _good(txt):
                return _clean_text(txt), "pypdf"
        except Exception:
            pass

    # 2) pdfminer via pdfplumber deps
    try:
        if PDF_OK:
            from pdfminer.high_level import extract_text as miner_extract  # type: ignore
            from pdfminer.layout import LAParams  # type: ignore
            laparams = LAParams(word_margin=0.1, char_margin=2.0, line_margin=0.2, all_texts=True)
            txt = miner_extract(io.BytesIO(file_bytes), laparams=laparams) or ""
            if _good(txt):
                return _clean_text(txt), "pdfminer.six"
    except Exception:
        pass

    # 3) pdfplumber
    if PDF_OK:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = [p.extract_text(x_tolerance=2, y_tolerance=2) or "" for p in pdf.pages]
            txt = "\n".join(pages)
            if _good(txt):
                return _clean_text(txt), "pdfplumber"
        except Exception:
            pass

    # 4) OCR
    if _HAS_OCR:
        ocr_txt = _ocr_pdf(file_bytes, dpi=ocr_dpi, max_pages=ocr_max_pages)
        if _good(ocr_txt):
            return ocr_txt, "OCR (PyMuPDF + Tesseract)"

    return "", ""

def _save_textfile(text: str, prefix: str, suffix: str = ".txt") -> Optional[str]:
    if not text:
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fd, path = tempfile.mkstemp(prefix=f"{prefix}-{ts}-", suffix=suffix)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

from datetime import datetime

def extract_to_box(upload) -> tuple[str, str, Optional[str]]:
    data, name = _file_to_bytes(upload)
    if data is None:
        return "", "_No file selected._", None

    base = os.path.basename((name or "uploaded"))
    is_pdf = (data[:5] == b"%PDF-") or base.lower().endswith(".pdf")

    try:
        tool = ""
        if is_pdf:
            text, tool = _pdf_to_text(data)
        else:
            text = data.decode("utf-8", errors="ignore")
            if _looks_like_pdf_source(text):
                text = ""
            tool = "plain-text"
        text = _clean_text(text)
    except Exception as e:
        return "", f"**Error:** Could not read *{base}*: `{e}`", None

    if text.strip():
        chars = f"{len(text):,}"
        tool_note = f" via **{tool}**" if tool else ""
        dl = _save_textfile(text, prefix="extracted")
        return text, f"Extracted **{chars}** characters from *{base}*{tool_note}. You can edit the text below.", dl
    else:
        hints = []
        if is_pdf and not _HAS_OCR:
            hints.append("OCR backend not available (PyMuPDF/PIL/pytesseract missing)")
        if is_pdf and not (_HAS_PYPDF or PDF_OK):
            hints.append("No PDF extractors available (install pypdf/pdfplumber)")
        extra = ("\n" + "; ".join(hints)) if hints else ""
        return "", f"Could not extract readable text from *{base}*. You can paste text into the box manually.{extra}", None


def _save_json_to_tmp(payload: Dict[str, Any] | Any) -> str:
    """Save payload to a temp JSON file and return its path."""
    fd, path = tempfile.mkstemp(prefix="law_coding_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def _ensure_text(upload_or_text: Union[str, os.PathLike, object]) -> str:
    if upload_or_text is None or (isinstance(upload_or_text, str) and not upload_or_text.strip()):
        raise gr.Error("Please paste text or upload a file.")
    if isinstance(upload_or_text, str) and not os.path.isfile(upload_or_text):
        return upload_or_text.strip()
    if isinstance(upload_or_text, str) and os.path.isfile(upload_or_text):
        return extract_text(upload_or_text)
    if _openable(upload_or_text):
        return extract_text(upload_or_text)
    raise gr.Error("Unsupported input type—expect text or a .txt/.pdf file.")