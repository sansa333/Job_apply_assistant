from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "demo" / "synthetic_resume.json"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "synthetic_demo_resume.pdf"


def register_demo_font() -> str:
    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    )
    for candidate in candidates:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont("DemoSansCN", str(candidate)))
            return "DemoSansCN"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def build_resume(source: Path, output: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("data_classification") != "synthetic_demo_only":
        raise ValueError("Only explicitly synthetic demo data may be rendered")

    output.parent.mkdir(parents=True, exist_ok=True)
    font_name = register_demo_font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=17,
        textColor=colors.HexColor("#233044"),
        wordWrap="CJK",
    )
    heading = ParagraphStyle(
        "HeadingCN",
        parent=body,
        fontSize=13,
        leading=20,
        textColor=colors.HexColor("#155E75"),
        spaceBefore=7,
        spaceAfter=4,
    )
    name_style = ParagraphStyle(
        "NameCN",
        parent=body,
        fontSize=22,
        leading=28,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0F172A"),
    )

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Synthetic AI application resume",
        author="AI Job Apply Assistant demo generator",
    )
    story = [
        Paragraph(data["name"], name_style),
        Paragraph(data["contact"], body),
        Paragraph("目标方向：" + " / ".join(data["target_roles"]), body),
        Spacer(1, 4 * mm),
    ]

    def section(title: str) -> None:
        story.append(
            Table(
                [[Paragraph(title, heading)]],
                colWidths=[174 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E0F2FE")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#7DD3FC")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 1),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ]
                ),
            )
        )

    section("教育背景")
    for item in data["education"]:
        story.append(
            Paragraph(
                f'{item["period"]}　{item["school"]}　{item["degree"]}',
                body,
            )
        )

    section("技能")
    story.append(Paragraph("、".join(data["skills"]), body))

    section("项目经历")
    for project in data["projects"]:
        story.append(Paragraph(f'{project["period"]}　{project["name"]}', heading))
        for detail in project["details"]:
            story.append(Paragraph("- " + detail, body))

    section("数据声明")
    story.append(Paragraph(data["notice"], body))
    document.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fully synthetic demo resume PDF")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_resume(args.source, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
