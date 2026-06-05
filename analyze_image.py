#!/usr/bin/env python3
"""
Image analysis tool for Claude Code's read-image skill.
Extracts metadata, performs OCR, and generates color analysis.

Usage:
    python analyze_image.py <image_path> [--ocr-lang <lang>]
"""
import json
import sys
import os
from pathlib import Path

try:
    from PIL import Image, ExifTags
    import numpy as np
except ImportError:
    print(json.dumps({"error": "Pillow not installed. Run: pip install Pillow numpy"}))
    sys.exit(1)

def analyze_image(image_path: str, ocr_lang: str = "eng"):
    """Analyze an image and return structured results."""
    path = Path(image_path)
    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    result = {"file": str(path), "ocr_lang": ocr_lang}

    # ── Open image ──
    img = Image.open(path)
    result["metadata"] = {
        "format": img.format or "unknown",
        "mode": img.mode,
        "width": img.width,
        "height": img.height,
        "aspect_ratio": round(img.width / img.height, 3),
        "file_size_bytes": path.stat().st_size,
        "file_size_kb": round(path.stat().st_size / 1024, 1),
    }

    # ── EXIF data ──
    exif_data = {}
    if hasattr(img, "_getexif") and img._getexif():
        for tag_id, value in img._getexif().items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            # Skip binary/too-long values
            if isinstance(value, (str, int, float)):
                exif_data[tag] = value
            elif isinstance(value, bytes):
                try:
                    exif_data[tag] = value.decode("utf-8", errors="replace")[:200]
                except:
                    pass
    if exif_data:
        result["exif"] = exif_data

    # ── Convert to RGB for analysis ──
    if img.mode == "RGBA":
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
    elif img.mode != "RGB":
        rgb_img = img.convert("RGB")
    else:
        rgb_img = img

    # ── Color analysis ──
    # Dominant colors via palette quantization
    palette = rgb_img.quantize(colors=6).convert("RGB")
    color_counts = palette.getcolors()
    if color_counts:
        color_counts.sort(reverse=True)
        dominant = []
        for count, color in color_counts[:5]:
            dominant.append({
                "rgb": f"rgb({color[0]},{color[1]},{color[2]})",
                "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                "pixel_count": count
            })
        result["colors"] = {"dominant": dominant}

    # ── Brightness / contrast ──
    gray = rgb_img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    result["luminance"] = {
        "mean": round(float(arr.mean()), 1),
        "std": round(float(arr.std()), 1),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "dark_pct": round(float((arr < 64).sum() / arr.size * 100), 1),
        "bright_pct": round(float((arr > 192).sum() / arr.size * 100), 1),
    }

    # ── Check if mostly dark (terminal screenshot heuristic) ──
    result["heuristics"] = {
        "likely_terminal_screenshot": result["luminance"]["mean"] < 64 and result["luminance"]["dark_pct"] > 50,
        "likely_photo": result["luminance"]["std"] > 60,
    }

    # ── OCR ──
    try:
        import pytesseract
        # Find tesseract executable
        tesseract_cmd = None
        for candidate in [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]:
            if os.path.exists(candidate):
                tesseract_cmd = candidate
                break
        # Also check PATH
        if not tesseract_cmd:
            import shutil
            tesseract_cmd = shutil.which("tesseract")

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

            # Check if requested language is available
            langs = pytesseract.get_languages(config="")
            if ocr_lang not in langs and "_".join(ocr_lang.split("+")) not in langs:
                # Try fallback
                available = [l for l in langs if l != "osd"]
                fallback = available[0] if available else "eng"
                result["ocr"] = {
                    "note": f"Language '{ocr_lang}' not available, fell back to '{fallback}'. Available: {langs}",
                    "language_used": fallback,
                }
                ocr_lang = fallback
            else:
                result["ocr"] = {"language_used": ocr_lang}

            text = pytesseract.image_to_string(rgb_img, lang=ocr_lang)
            # Clean and limit output
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            result["ocr"]["lines"] = len(lines)
            result["ocr"]["total_chars"] = len(text)
            result["ocr"]["text"] = text[:3000]  # Cap at 3000 chars
        else:
            result["ocr"] = {"error": "Tesseract not found on system"}
    except ImportError:
        result["ocr"] = {"error": "pytesseract not installed. Run: pip install pytesseract"}
    except Exception as e:
        result["ocr"] = {"error": str(e)}

    return result

if __name__ == "__main__":
    # Fix Windows console encoding for Chinese output
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    if not args:
        print(json.dumps({"error": "Usage: python analyze_image.py <image_path> [--ocr-lang <lang>]"}))
        sys.exit(1)

    ocr_lang = "eng"
    img_path = args[0]
    if "--ocr-lang" in args:
        idx = args.index("--ocr-lang")
        if idx + 1 < len(args):
            ocr_lang = args[idx + 1]

    result = analyze_image(img_path, ocr_lang)
    print(json.dumps(result, ensure_ascii=False, indent=2))
