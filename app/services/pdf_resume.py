"""Read-only, page-aware resume extraction with local OCR fallback."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import threading
import unicodedata

from pypdf import PdfReader

from app.config import settings


_render_lock = threading.Lock()  # PDFium is not thread safe.
_ocr_lock = threading.Lock()


@dataclass(frozen=True)
class PdfExtraction:
    text: str
    pages: list[dict]

    def report(self) -> dict:
        return {
            "version": "pdf-resume-v1",
            "page_count": len(self.pages),
            "ocr_pages": [p["page_number"] for p in self.pages if p["method"] == "ocr"],
            "pages": self.pages,
            "policy": "local_read_only",
        }


def quality_issues(text: str) -> list[str]:
    compact = "".join(text.split())
    issues = []
    if sum(c.isalnum() for c in compact) < 40:
        issues.append("sparse_text")
    invalid = sum(c == "\ufffd" or unicodedata.category(c) in {"Co", "Cc", "Cn"} for c in compact)
    if compact and invalid / len(compact) > 0.02:
        issues.append("garbled_text")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 10 and sum(len(line) <= 2 for line in lines) / len(lines) > 0.5:
        issues.append("fragmented_reading_order")
    return issues


@lru_cache(maxsize=1)
def _ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise ValueError("扫描简历需要本地 OCR 依赖，请安装完整 requirements.txt 后重试。") from exc
    # Bundled Chinese/English ONNX models; no external document upload.
    return RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)


def render_page(path: Path, page_index: int):
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ValueError("扫描简历需要 PDF 渲染依赖，请安装完整 requirements.txt 后重试。") from exc
    with _render_lock:
        pdf = pdfium.PdfDocument(str(path))
        try:
            page = pdf[page_index]
            try:
                width, height = page.get_size()
                scale = settings.pdf_ocr_dpi / 72
                if width <= 0 or height <= 0 or width * height * scale * scale > settings.pdf_ocr_max_pixels:
                    raise ValueError(f"第 {page_index + 1} 页尺寸过大，无法安全执行 OCR。")
                bitmap = page.render(scale=scale)
                try:
                    return bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            pdf.close()


def recognize_page(image) -> list[dict]:
    import numpy as np

    with _ocr_lock:
        engine = _ocr_engine()
        pixels = np.asarray(image)
        result, _ = engine(pixels, text_score=0.0)
        # The direction classifier can flip an already upright line. Compare
        # an unrotated pass when confidence is poor instead of accepting it.
        if not result or any(float(row[2]) < 0.8 for row in result):
            unrotated, _ = engine(pixels, use_cls=False, text_score=0.0)

            def confidence(rows):
                return sum(float(row[2]) for row in (rows or [])) / max(1, len(rows or []))

            if confidence(unrotated) > confidence(result):
                result = unrotated
    return [
        {"text": str(text).strip(), "confidence": float(score),
         "bbox": [[float(x), float(y)] for x, y in box]}
        for box, text, score in (result or []) if str(text).strip()
    ]


def extract_resume_pdf(path: Path) -> PdfExtraction:
    reader = PdfReader(str(path))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF 已加密，请上传解除密码保护后的副本。")
    if not 0 < len(reader.pages) <= settings.pdf_resume_max_pages:
        raise ValueError(f"简历 PDF 页数应在 1–{settings.pdf_resume_max_pages} 页之间。")
    pages = []
    for number, page in enumerate(reader.pages, 1):
        extraction_failed = False
        try:
            native = (page.extract_text() or "").strip()
        except Exception:
            native = ""
            extraction_failed = True
        issues = quality_issues(native)
        if extraction_failed:
            issues.append("text_extraction_failed")
        record = {"page_number": number, "method": "text", "text": native,
                  "native_text": native, "quality_issues": issues, "warnings": [], "blocks": []}
        if issues:
            if not settings.pdf_ocr_enabled:
                raise ValueError(f"第 {number} 页需要 OCR，但 PDF_OCR_ENABLED 已关闭。")
            try:
                image = render_page(path, number - 1)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"第 {number} 页渲染失败，请上传可正常打开的 PDF。") from exc
            try:
                import numpy as np

                pixels = np.asarray(image.convert("L"))
                blank = float(pixels.mean()) > 245 and float(pixels.std()) < 2
                blocks = [] if blank else recognize_page(image)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"第 {number} 页本地 OCR 失败，请检查依赖或换用清晰的 PDF。") from exc
            finally:
                image.close()
            if blank and not native:
                record.update(method="blank", text="")
            else:
                recognized = "\n".join(block["text"] for block in blocks)
                if not recognized or any(i != "sparse_text" for i in quality_issues(recognized)):
                    raise ValueError(f"第 {number} 页无法可靠读取，请上传更清晰的扫描件或文字版 PDF。")
                if sum(b["confidence"] for b in blocks) / max(1, len(blocks)) < 0.65:
                    raise ValueError(f"第 {number} 页 OCR 置信度过低，请上传更清晰的 PDF。")
                record.update(method="ocr", text=recognized, blocks=blocks)
                record["warnings"].append("OCR 文字及多栏阅读顺序需人工核对。")
                if any(b["confidence"] < 0.8 for b in blocks):
                    record["warnings"].append("存在低置信度文字，请重点核对数字、日期和联系方式。")
        pages.append(record)
    text = "\n\n".join(p["text"] for p in pages if p["text"]).strip()
    if sum(c.isalnum() for c in text) < 100:
        raise ValueError("PDF 经文本提取和 OCR 后有效内容仍不足，请上传完整且清晰的简历。")
    return PdfExtraction(text=text, pages=pages)
