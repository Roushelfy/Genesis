"""Inspect actual PDF/PNG/SVG exports and rasterize each PDF for visual review."""

import argparse
import json
import sys
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--pdf-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.pdf_library))
    import pymupdf

    rows = []
    text_qa = json.loads((args.figures / "rendered-text-qa.json").read_text())
    assert all(not row["text_issues"] for row in text_qa), text_qa
    for pdf in sorted(args.figures.glob("*.pdf")):
        with pymupdf.open(pdf) as document:
            assert len(document) == 1
            page = document[0]
            width, height = page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72
            assert abs(width - 183) < 0.01 and height <= 170
            fonts = page.get_fonts(full=True)
            assert fonts and all("Type3" not in str(font) for font in fonts)
            for font in fonts:
                assert document.extract_font(font[0])[3], font
            page.get_pixmap(dpi=150).save(str(args.output / f"{pdf.stem}-pdf-preview.png"))
            page.get_pixmap(dpi=450).save(str(args.output / f"{pdf.stem}-pdf-450.png"))
        svg = ElementTree.parse(pdf.with_suffix(".svg"))
        texts = svg.findall(".//{http://www.w3.org/2000/svg}text")
        assert len(texts) > 20
        with Image.open(pdf.with_suffix(".png")) as image:
            dpi = image.info.get("dpi")
            assert dpi and all(abs(value - 450) < 0.1 for value in dpi)
            assert image.mode in ("RGB", "RGBA")
            png_size = image.size
        rows.append(
            {
                "figure": pdf.stem,
                "width_mm": width,
                "height_mm": height,
                "fonts": fonts,
                "fonts_embedded": True,
                "type3_fonts": False,
                "svg_editable_text_count": len(texts),
                "png_pixels": png_size,
                "png_dpi": dpi,
            }
        )
    (args.output / "result.json").write_text(json.dumps({"kind": "actual-export-qa", "figures": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
