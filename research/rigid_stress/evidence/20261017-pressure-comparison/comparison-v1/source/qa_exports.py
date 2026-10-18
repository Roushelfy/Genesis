"""Check final-size editable exports and rasterize PDFs for visual inspection."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops, ImageStat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", nargs="+", default=["performance", "model-error-cost", "reward-phase"])
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    results = []
    for name in args.names:
        path = args.figures / name
        original = json.loads(path.with_name(name + "-qa.json").read_text())
        assert not original["text_qa_issues"]
        root = ET.fromstring(path.with_suffix(".svg").read_bytes())
        text_count = len(root.findall(".//{http://www.w3.org/2000/svg}text"))
        assert text_count > 10
        width_mm = float(root.attrib["width"].removesuffix("pt")) * 25.4 / 72
        height_mm = float(root.attrib["height"].removesuffix("pt")) * 25.4 / 72
        assert abs(width_mm - 183) < 1e-5 and height_mm <= 170
        pdf = pymupdf.open(path.with_suffix(".pdf"))
        assert len(pdf) == 1
        fonts = []
        for font in pdf.get_page_fonts(0, full=True):
            basename, extension, kind, content = pdf.extract_font(font[0])
            assert kind != "Type3" and extension == "ttf" and len(content) > 0
            fonts.append({"name": basename, "extension": extension, "type": kind, "embedded_bytes": len(content)})
        assert fonts
        page = pdf[0]
        assert abs(page.rect.width * 25.4 / 72 - 183) < 0.001
        target = args.output / (name + "-pdf-preview.png")
        page.get_pixmap(dpi=150, alpha=False).save(target)
        pdf.close()
        png = Image.open(path.with_suffix(".png"))
        preview = Image.open(path.with_name(name + "_preview.png")).convert("RGB")
        pdf_preview = Image.open(target).convert("RGB")
        assert png.info["dpi"][0] >= 449.9 and abs(png.width / 450 * 25.4 - 183) < 0.1
        # PDF pixel bounds round upward. Resize only this QA comparison, never
        # source data or the scientific exports.
        matched = pdf_preview.resize(preview.size, Image.Resampling.BILINEAR)
        difference = ImageStat.Stat(ImageChops.difference(preview, matched))
        mean_difference = sum(difference.mean) / 3
        assert mean_difference < 5
        results.append(
            {
                "figure": name,
                "svg_width_mm": width_mm,
                "svg_height_mm": height_mm,
                "svg_editable_text_elements": text_count,
                "pdf_fonts": fonts,
                "pymupdf_version": pymupdf.version,
                "pdf_all_fonts_embedded": True,
                "pdf_has_type3": False,
                "png_dimensions": png.size,
                "png_dpi": png.info["dpi"],
                "pdf_vs_png_preview_mean_abs_difference_0_255": mean_difference,
                "exports_sha256": {
                    suffix: hashlib.sha256(path.with_suffix(suffix).read_bytes()).hexdigest()
                    for suffix in (".svg", ".pdf", ".png")
                },
                "manual_visual_review": "pending",
            }
        )
    (args.output / "export-qa.json").write_text(
        json.dumps({"figures": results, "passed_automated_checks": True}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
