from __future__ import annotations

from unittest.mock import Mock
from pathlib import Path

import pytest
import reportlab
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from app.config import settings
from app.services import pdf_resume


LINES = [
    "Candidate: Demo. Graduation Year: 2027. Bachelor in Computer Science.",
    "Project: Python API with FastAPI, RAG knowledge base and Chroma retrieval.",
    "Implemented document ingestion and evidence search for job applications.",
]


def make_pdf(path, kinds):
    canvas = Canvas(str(path), pagesize=(600, 850))
    for kind in kinds:
        if kind == "text":
            for i, line in enumerate(LINES):
                canvas.drawString(30, 800 - i * 22, line)
        elif kind == "scan":
            image = Image.new("RGB", (1200, 1700), "white")
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(str(Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"), 25)
            for i, line in enumerate(LINES):
                draw.text((45, 100 + i * 50), line, font=font, fill="black")
            canvas.drawImage(ImageReader(image), 0, 0, 600, 850)
        elif kind == "unreadable":
            canvas.setFillColorRGB(0, 0, 0)
            canvas.rect(0, 0, 600, 850, fill=1)
        canvas.showPage()
    canvas.save()


def blocks(text=None, confidence=0.97):
    return [{"text": text or "\n".join(LINES), "confidence": confidence, "bbox": [[0, 0], [1, 0], [1, 1], [0, 1]]}]


def test_native_pdf_never_loads_ocr(tmp_path, monkeypatch):
    path = tmp_path / "native.pdf"
    make_pdf(path, ["text"])
    original = path.read_bytes()
    ocr = Mock(side_effect=AssertionError("must not OCR native text"))
    monkeypatch.setattr(pdf_resume, "recognize_page", ocr)
    result = pdf_resume.extract_resume_pdf(path)
    assert "FastAPI" in result.text
    assert result.report()["ocr_pages"] == []
    assert path.read_bytes() == original
    ocr.assert_not_called()


def test_mixed_pdf_reads_scan_even_when_first_page_has_enough_text(tmp_path, monkeypatch):
    path = tmp_path / "mixed.pdf"
    make_pdf(path, ["text", "scan", "blank"])
    ocr = Mock(return_value=blocks("第二页扫描项目经历：Python OCR", 0.7))
    monkeypatch.setattr(pdf_resume, "recognize_page", ocr)
    result = pdf_resume.extract_resume_pdf(path)
    assert "第二页扫描项目经历" in result.text
    assert [p["method"] for p in result.pages] == ["text", "ocr", "blank"]
    assert result.report()["ocr_pages"] == [2]
    assert len(result.pages[1]["warnings"]) == 2
    ocr.assert_called_once()


def test_bad_scan_does_not_silently_drop_page(tmp_path, monkeypatch):
    path = tmp_path / "bad.pdf"
    make_pdf(path, ["text", "unreadable"])
    monkeypatch.setattr(pdf_resume, "recognize_page", lambda image: [])
    with pytest.raises(ValueError, match="第 2 页"):
        pdf_resume.extract_resume_pdf(path)


def test_disabled_ocr_rejects_scan(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    make_pdf(path, ["scan"])
    monkeypatch.setattr(settings, "pdf_ocr_enabled", False)
    with pytest.raises(ValueError, match="PDF_OCR_ENABLED"):
        pdf_resume.extract_resume_pdf(path)


def test_quality_detects_garbled_and_fragmented_text():
    assert "garbled_text" in pdf_resume.quality_issues("\ufffd" * 30 + "Python" * 10)
    assert "fragmented_reading_order" in pdf_resume.quality_issues("\n".join("a" * 100))


def test_page_and_render_limits(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    make_pdf(path, ["scan", "text"])
    monkeypatch.setattr(settings, "pdf_resume_max_pages", 1)
    with pytest.raises(ValueError, match="页数"):
        pdf_resume.extract_resume_pdf(path)
    monkeypatch.setattr(settings, "pdf_resume_max_pages", 20)
    monkeypatch.setattr(settings, "pdf_ocr_max_pixels", 100)
    with pytest.raises(ValueError, match="尺寸"):
        pdf_resume.extract_resume_pdf(path)


def test_ocr_failure_preserves_existing_resume_via_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setattr(settings, "source_corpus_dir", tmp_path / "sources")
    monkeypatch.setattr(settings, "api_token", None)
    directory = settings.source_corpus_dir / "candidate_profiles" / "demo"
    directory.mkdir(parents=True)
    original = directory / "resume_source.pdf"
    original.write_bytes(b"previous-valid-resume")
    path = tmp_path / "bad.pdf"
    make_pdf(path, ["unreadable"])
    monkeypatch.setattr(pdf_resume, "recognize_page", lambda image: [])
    response = TestClient(app).post("/api/domestic/profile/pdf", params={"candidate_id": "demo"},
                                    files={"file": ("scan.pdf", path.read_bytes(), "application/pdf")})
    assert response.status_code == 422
    assert original.read_bytes() == b"previous-valid-resume"


def test_page_provenance_reaches_candidate_index(tmp_path):
    from app.knowledge.profiles import CandidateProfileStore

    store = CandidateProfileStore(tmp_path / "sources", tmp_path / "vectors")
    try:
        store.ingest_text("demo", "\n".join(LINES), pages=[
            {"page_number": 2, "method": "ocr", "text": "\n".join(LINES)}])
        docs = store.retrieve("demo", "Python")
        assert docs and all(d.metadata["page_number"] == 2 for d in docs)
        assert all(d.metadata["extraction_method"] == "ocr" for d in docs)
    finally:
        store.close()


def test_real_local_ocr_on_scanned_pdf(tmp_path):
    """Real bundled ONNX inference; no LLM, network or mocked OCR."""
    path = tmp_path / "scan.pdf"
    make_pdf(path, ["scan"])
    result = pdf_resume.extract_resume_pdf(path)
    assert result.report()["ocr_pages"] == [1]
    assert "Python" in result.text
    assert "2027" in result.text
    assert result.pages[0]["blocks"]


def test_real_chinese_scan_import_keeps_report_and_original(tmp_path, monkeypatch):
    from app.domestic.profile import ingest_pdf_profile

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not font_path.is_file():
        pytest.skip("Chinese integration fixture needs Microsoft YaHei font")
    image = Image.new("RGB", (1200, 1700), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 32)
    lines = [
        "个人简历：测试候选人，毕业年份为2027年。",
        "教育经历：计算机科学与技术专业，本科学历。",
        "项目经历：使用Python开发知识库与智能问答系统。",
        "工作内容：负责文档处理、语义检索以及接口开发。",
        "技术能力：熟悉数据库设计和软件工程，具备团队协作经验。",
        "求职意向：人工智能应用开发工程师，期望城市为上海。",
        "实习经历：完成需求分析、测试验证和技术文档整理。",
    ]
    for i, line in enumerate(lines):
        draw.text((50, 100 + 65 * i), line, font=font, fill="black")
    path = tmp_path / "chinese.pdf"
    canvas = Canvas(str(path), pagesize=(600, 850))
    canvas.drawImage(ImageReader(image), 0, 0, 600, 850)
    canvas.save()
    original = path.read_bytes()
    result = ingest_pdf_profile(candidate_id="demo", source_pdf=path,
                                source_corpus_dir=tmp_path / "sources",
                                vector_db_dir=tmp_path / "vectors", collection_name="ocr_test")
    assert result.managed_pdf_path.read_bytes() == original
    text = result.extracted_text_path.read_text(encoding="utf-8")
    assert "2027" in text and "项目经历" in text and "Python" in text
    assert result.profile["pdf_extraction"]["ocr_pages"] == [1]
    assert result.chunks_added > 0
    assert (result.profile_json_path.parent / "extraction_report.json").is_file()


def test_direction_classifier_fallback(monkeypatch):
    engine = Mock(side_effect=[([([[0, 0]], "bad", 0.3)], None),
                               ([([[0, 0]], "Python", 0.98)], None)])
    monkeypatch.setattr(pdf_resume, "_ocr_engine", lambda: engine)
    result = pdf_resume.recognize_page(Image.new("RGB", (10, 10)))
    assert result[0]["text"] == "Python"
    assert engine.call_args.kwargs["use_cls"] is False
