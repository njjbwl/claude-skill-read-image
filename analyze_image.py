#!/usr/bin/env python3
"""
Image analysis tool for Claude Code's read-image skill.
Extracts metadata, performs OCR (with preprocessing), color analysis,
and optional vision API integration (SiliconFlow Qwen3-VL-32B).

Usage:
    python analyze_image.py <image_path> [options]

Options:
    --ocr-lang <lang>    OCR language (default: eng, use chi_sim+eng for Chinese)
    --vision             Enable SiliconFlow vision API (requires SILICONFLOW_API_KEY)
    --check              Check dependencies and report status
    --json               Output as JSON (default)
    --pretty             Pretty-print JSON output
"""
import json
import sys
import os
import io
import base64
import argparse
from pathlib import Path

# ═══════════════════════════════════════════════
# Dependency detection (graceful degradation)
# ═══════════════════════════════════════════════

DEP = {
    "pillow": False, "numpy": False,
    "pytesseract": False, "tesseract_bin": False,
    "requests": False, "sklearn": False,
    "vision_api_key": False,
}

# Pillow
try:
    from PIL import Image, ExifTags
    DEP["pillow"] = True
except ImportError:
    Image = None
    ExifTags = None

# numpy
try:
    import numpy as np
    DEP["numpy"] = True
except ImportError:
    np = None

# pytesseract + tesseract binary
try:
    import pytesseract as _pt
    DEP["pytesseract"] = True
    # Search for tesseract executable
    _candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for _c in _candidates:
        if os.path.exists(_c):
            _pt.pytesseract.tesseract_cmd = _c
            DEP["tesseract_bin"] = True
            break
    if not DEP["tesseract_bin"]:
        import shutil
        if shutil.which("tesseract"):
            DEP["tesseract_bin"] = True
except ImportError:
    _pt = None

# requests (for Vision API)
try:
    import requests as _req
    DEP["requests"] = True
except ImportError:
    _req = None

# sklearn (optional, for k-means color analysis)
try:
    from sklearn.cluster import KMeans as _KM
    from sklearn.utils import check_random_state
    DEP["sklearn"] = True
except ImportError:
    _KM = None

# Vision API key
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "") or ""
DEP["vision_api_key"] = bool(SILICONFLOW_API_KEY)


# ═══════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════

def _to_rgb(img):
    """Convert any image to RGB, handling RGBA with white background."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    elif img.mode != "RGB":
        return img.convert("RGB")
    return img


def _downscale(img, max_dim=2000):
    """Downscale image if it exceeds max_dim on the longest side."""
    if max(img.width, img.height) <= max_dim:
        return img, False
    ratio = max_dim / max(img.width, img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS), True


def _percent_str(part, total):
    return round(part / total * 100, 1) if total else 0.0


# ═══════════════════════════════════════════════
# Module 1: Metadata & EXIF
# ═══════════════════════════════════════════════

def extract_metadata(img, path):
    md = {
        "format": img.format or "unknown",
        "mode": img.mode,
        "width": img.width,
        "height": img.height,
        "aspect_ratio": round(img.width / img.height, 3),
        "file_size_bytes": path.stat().st_size,
        "file_size_kb": round(path.stat().st_size / 1024, 1),
    }
    if hasattr(img, "info") and img.info.get("dpi"):
        md["dpi"] = img.info["dpi"]
    return md


def extract_exif(img):
    if not hasattr(img, "_getexif") or not img._getexif():
        return None
    exif = {}
    for tag_id, value in img._getexif().items():
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        if isinstance(value, (str, int, float)):
            exif[tag] = value
        elif isinstance(value, bytes):
            try:
                exif[tag] = value.decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
    return exif if exif else None


# ═══════════════════════════════════════════════
# Module 2: Color analysis (k-means with fallback)
# ═══════════════════════════════════════════════

def analyze_colors(rgb_img):
    """Dominant colors via k-means (sklearn) or palette quantization (fallback)."""
    result = {}
    total_pixels = rgb_img.width * rgb_img.height

    # Small thumbnail for speed
    thumb = rgb_img.copy()
    if max(thumb.width, thumb.height) > 200:
        ratio = 200 / max(thumb.width, thumb.height)
        thumb = thumb.resize((int(thumb.width * ratio), int(thumb.height * ratio)), Image.LANCZOS)

    n_colors = 5

    if DEP["sklearn"] and DEP["numpy"]:
        # k-means clustering
        arr = np.array(thumb).reshape(-1, 3)
        kmeans = _KM(n_clusters=n_colors, n_init=3, random_state=42, max_iter=300)
        kmeans.fit(arr)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.astype(int)

        counts = np.bincount(labels, minlength=n_colors)
        # Sort by count descending
        order = np.argsort(-counts)

        dominant = []
        for idx in order:
            c = centers[idx]
            count = int(counts[idx])
            pct = _percent_str(count, len(labels))
            dominant.append({
                "rgb": f"rgb({c[0]},{c[1]},{c[2]})",
                "hex": f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}",
                "pixel_count": count,
                "percentage": pct,
            })
        result["dominant"] = dominant
        result["method"] = "kmeans"
    elif DEP["pillow"]:
        # Fallback: palette quantization
        palette = thumb.quantize(colors=n_colors + 1).convert("RGB")
        color_counts = palette.getcolors()
        if color_counts:
            color_counts.sort(reverse=True)
            dominant = []
            total = sum(c for c, _ in color_counts[:n_colors])
            for count, color in color_counts[:n_colors]:
                dominant.append({
                    "rgb": f"rgb({color[0]},{color[1]},{color[2]})",
                    "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                    "pixel_count": count,
                    "percentage": _percent_str(count, total),
                })
            result["dominant"] = dominant
            result["method"] = "quantize"
    else:
        return None

    return result


# ═══════════════════════════════════════════════
# Module 3: Luminance & heuristics
# ═══════════════════════════════════════════════

def analyze_luminance(rgb_img):
    if not DEP["numpy"]:
        return None
    gray = rgb_img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    return {
        "mean": round(float(arr.mean()), 1),
        "std": round(float(arr.std()), 1),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "dark_pct": round(float((arr < 64).sum() / arr.size * 100), 1),
        "bright_pct": round(float((arr > 192).sum() / arr.size * 100), 1),
    }


def _hex_brightness(hex_color):
    """Perceived brightness from hex color (0-255)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r * 0.299 + g * 0.587 + b * 0.114)


def heuristics(md, lum, colors=None, ocr=None):
    """
    Multi-signal image classification.
    Combines luminance, color palette, OCR, and metadata.
    Returns dict of boolean classifications for the AI to interpret.
    """
    h = {}
    sig = {}

    # ── Luminance ──
    if lum:
        sig["dark"] = lum["mean"] < 60
        sig["bright"] = lum["mean"] > 180
        sig["high_contrast"] = lum["std"] > 40
        sig["low_contrast"] = lum["std"] < 25
        sig["bimodal"] = lum["dark_pct"] > 30 and (lum["max"] - lum["mean"]) > 60
        sig["mostly_dark_area"] = lum["dark_pct"] > 50

    # ── Colors ──
    if colors and colors.get("dominant"):
        dom = colors["dominant"]
        top_pct = dom[0]["percentage"] if dom else 0
        sig["single_dominant"] = top_pct > 80
        sig["colorful"] = len(dom) >= 4 and top_pct < 50
        dark_c = sum(1 for c in dom[:3] if _hex_brightness(c["hex"]) < 80)
        bright_c = sum(1 for c in dom[:3] if _hex_brightness(c["hex"]) > 180)
        sig["mostly_dark"] = dark_c >= 2
        sig["mostly_bright"] = bright_c >= 2

    # ── OCR ──
    if ocr and "error" not in ocr:
        sig["has_text"] = ocr.get("lines", 0) > 0
        sig["dense_text"] = ocr.get("lines", 0) > 15
    else:
        sig["has_text"] = False

    # ── Metadata ──
    if md:
        mp = (md.get("width", 0) * md.get("height", 0)) / 1e6
        ar = md.get("aspect_ratio", 0)
        sig["large"] = mp > 5
        sig["widescreen"] = ar and abs(ar - 16/9) < 0.15
        sig["portrait"] = ar and 0 < ar < 0.9
        sig["tall_portrait"] = ar and 0 < ar < 0.56

    # ── Classification rules ──
    h["terminal_screenshot"] = bool(
        sig.get("dark") and sig.get("bimodal")
        and sig.get("has_text") and not sig.get("large")
    )
    h["code_screenshot"] = bool(
        not sig.get("dark") and sig.get("has_text")
        and sig.get("mostly_bright") and sig.get("high_contrast")
        and not sig.get("large")
    )
    h["document_scan"] = bool(
        sig.get("bright") and sig.get("has_text")
        and sig.get("single_dominant") and not sig.get("colorful")
    )
    h["photo"] = bool(
        sig.get("large")
        or (sig.get("colorful") and not sig.get("has_text"))
        or (sig.get("high_contrast") and not sig.get("has_text")
            and not sig.get("mostly_dark"))
    )
    h["chart_diagram"] = bool(
        sig.get("colorful") and sig.get("has_text")
        and not sig.get("large")
    )
    h["mobile_screenshot"] = bool(
        sig.get("tall_portrait") and sig.get("has_text")
        and not sig.get("large")
    )

    h["text_screenshot"] = bool(
        sig.get("has_text") and not sig.get("photo")
        and not sig.get("large")
    )

    if sig.get("widescreen"):
        h["widescreen"] = True
    if sig.get("portrait") and md.get("width", 0) < md.get("height", 0):
        h["portrait"] = True

    return h if any(v for v in h.values()) else None


# ═══════════════════════════════════════════════
# Module 4: OCR with preprocessing
# ═══════════════════════════════════════════════

def _ocr_deskew(img):
    if not DEP["pytesseract"] or not DEP["tesseract_bin"]:
        return img
    try:
        osd = _pt.image_to_osd(img, output_type=_pt.Output.DICT)
        angle = float(osd.get("rotate", 0))
        if abs(angle) > 0.5:
            return img.rotate(-angle, expand=True, fillcolor=(255, 255, 255))
    except Exception:
        pass
    return img


def _ocr_preprocess(rgb_img):
    if not DEP["numpy"]:
        return rgb_img

    arr = np.array(rgb_img.convert("L"), dtype=np.uint8)
    mean_brightness = float(arr.mean())

    if mean_brightness < 80:
        result_img = rgb_img
    elif mean_brightness > 240:
        gray_arr = np.array(rgb_img.convert("L"), dtype=np.uint8)
        threshold = max(mean_brightness * 0.85, 128)
        binary = (gray_arr > threshold).astype(np.uint8) * 255
        result_img = Image.fromarray(binary).convert("RGB")
    else:
        gray_img = rgb_img.convert("L")
        threshold = max(mean_brightness * 0.8, 100)
        binary = (np.array(gray_img, dtype=np.uint8) > threshold).astype(np.uint8) * 255
        result_img = Image.fromarray(binary).convert("RGB")

    w, h = result_img.size
    if min(h, w) < 500:
        scale = max(1.0, 800 / min(h, w))
        if scale > 1.2:
            new_size = (int(w * scale), int(h * scale))
            result_img = result_img.resize(new_size, Image.LANCZOS)

    return result_img


def run_ocr(rgb_img, ocr_lang="eng"):
    if not DEP["pytesseract"]:
        return {"error": "pytesseract not installed. Run: pip install pytesseract"}
    if not DEP["tesseract_bin"]:
        return {"error": "Tesseract binary not found on system"}

    try:
        try:
            available = _pt.get_languages(config="")
        except Exception:
            available = ["eng"]

        requested_langs = ocr_lang.replace("+", " ").split()
        missing = [l for l in requested_langs if l not in available]
        if len(missing) == len(requested_langs):
            fallback = [l for l in available if l != "osd"]
            lang_used = fallback[0] if fallback else "eng"
        else:
            lang_used = ocr_lang

        processed = _ocr_preprocess(rgb_img)
        processed = _ocr_deskew(processed)

        data = _pt.image_to_data(processed, lang=lang_used, output_type=_pt.Output.DICT)
        text = _pt.image_to_string(processed, lang=lang_used)

        words = []
        text_lines = [l.strip() for l in text.split("\n") if l.strip()]

        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
            if word and conf > 0:
                words.append({
                    "text": word,
                    "confidence": conf,
                    "bbox": {
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "w": data["width"][i],
                        "h": data["height"][i],
                    },
                    "line": data["line_num"][i],
                    "block": data["block_num"][i],
                })

        max_chars = 3000
        truncated = len(text) > max_chars

        ocr_result = {
            "language_used": lang_used,
            "language_requested": ocr_lang,
            "lines": len(text_lines),
            "words": len(words),
            "total_chars": len(text),
            "text": text[:max_chars],
            "truncated": truncated,
            "word_details": words[:200],
        }

        if words:
            ocr_result["avg_confidence"] = round(
                sum(w["confidence"] for w in words) / len(words), 1
            )

        return ocr_result

    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════
# Module 5: Vision API (SiliconFlow)
# ═══════════════════════════════════════════════

VISION_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
VISION_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"


def call_vision(image_path: str, prompt: str = None) -> dict:
    if not DEP["requests"]:
        return {"error": "requests library not installed. Run: pip install requests"}
    if not DEP["vision_api_key"]:
        return {"error": "SILICONFLOW_API_KEY not set in environment"}
    if not DEP["pillow"]:
        return {"error": "Pillow not installed"}

    if prompt is None:
        prompt = (
            "请详细描述这张图片的内容，包括：\n"
            "1. 画面中有什么（场景、物体、人物、布局）\n"
            "2. 所有可见的文字内容（逐行列出）\n"
            "3. 整体颜色色调和氛围\n"
            "4. 图片类型（截图/照片/图表/插画等）"
        )

    try:
        img = Image.open(image_path)
        fmt = img.format or "PNG"
        buf = io.BytesIO()
        save_fmt = fmt
        if fmt.upper() in ("PNG", "JPEG", "JPG", "WEBP"):
            img.save(buf, format=fmt)
        else:
            img.save(buf, format="PNG")
            save_fmt = "PNG"

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        ext = save_fmt.lower().replace("jpeg", "jpg")
        data_url = f"data:image/{ext};base64,{b64}"

        resp = _req.post(
            VISION_ENDPOINT,
            headers={
                "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": VISION_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                "max_tokens": 1024,
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not content:
            return {"error": "Empty response from vision API"}

        return {
            "description": content[:2000],
            "model": VISION_MODEL,
            "usage": data.get("usage", {}),
        }

    except _req.exceptions.Timeout:
        return {"error": "Vision API request timed out (60s)"}
    except _req.exceptions.RequestException as e:
        return {"error": f"Vision API request failed: {e}"}
    except Exception as e:
        return {"error": f"Vision API error: {e}"}


# ═══════════════════════════════════════════════
# Main analysis pipeline
# ═══════════════════════════════════════════════

def analyze_image(image_path: str, ocr_lang: str = "eng", enable_vision: bool = False):
    path = Path(image_path)
    result = {
        "file": str(path.resolve()),
        "ocr_lang": ocr_lang,
        "dependencies": {k: v for k, v in DEP.items()},
    }

    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    if not DEP["pillow"]:
        return {
            "error": "Pillow not installed. Run: pip install Pillow numpy",
            "file": str(path.resolve()),
        }

    try:
        img = Image.open(path)
    except Exception as e:
        return {"error": f"Cannot open image: {e}", "file": str(path.resolve())}

    result["metadata"] = extract_metadata(img, path)

    exif = extract_exif(img)
    if exif:
        result["exif"] = exif

    rgb_img = _to_rgb(img)
    rgb_img, was_downscaled = _downscale(rgb_img, max_dim=2000)
    if was_downscaled:
        result["downscaled"] = {"original_size": f"{img.width}x{img.height}", "new_size": f"{rgb_img.width}x{rgb_img.height}"}

    colors = analyze_colors(rgb_img)
    if colors:
        result["colors"] = colors

    lum = analyze_luminance(rgb_img)
    if lum:
        result["luminance"] = lum

    ocr_result = run_ocr(rgb_img, ocr_lang)
    result["ocr"] = ocr_result

    h = heuristics(
        result.get("metadata", {}), lum,
        colors=result.get("colors"), ocr=ocr_result,
    )
    if h:
        result["heuristics"] = h

    if enable_vision:
        if DEP["vision_api_key"] and DEP["requests"]:
            result["vision"] = call_vision(image_path)
        else:
            result["vision"] = {
                "note": "Vision API disabled. Set SILICONFLOW_API_KEY and install requests.",
                "available": DEP["vision_api_key"] and DEP["requests"],
            }

    return result


# ═══════════════════════════════════════════════
# Dependency check
# ═══════════════════════════════════════════════

def check_dependencies():
    checks = {}

    if DEP["pillow"]:
        checks["pillow"] = {"status": "ok", "version": Image.__version__ if hasattr(Image, "__version__") else "?"}
    else:
        checks["pillow"] = {"status": "missing", "fix": "pip install Pillow"}

    if DEP["numpy"]:
        checks["numpy"] = {"status": "ok", "version": np.__version__ if hasattr(np, "__version__") else "?"}
    else:
        checks["numpy"] = {"status": "missing", "fix": "pip install numpy"}

    if DEP["pytesseract"]:
        checks["pytesseract"] = {"status": "ok"}
    else:
        checks["pytesseract"] = {"status": "missing", "fix": "pip install pytesseract"}

    if DEP["tesseract_bin"]:
        try:
            import subprocess
            ver = subprocess.check_output(["tesseract", "--version"], stderr=subprocess.STDOUT, timeout=5)
            first_line = ver.decode("utf-8", errors="replace").split("\n")[0].strip()
            checks["tesseract_binary"] = {"status": "ok", "version": first_line}
        except Exception:
            checks["tesseract_binary"] = {"status": "ok", "version": "?"}
    else:
        checks["tesseract_binary"] = {"status": "missing", "fix": "Install Tesseract-OCR from https://github.com/UB-Mannheim/tesseract/wiki"}

    if DEP["pytesseract"] and DEP["tesseract_bin"]:
        try:
            langs = _pt.get_languages(config="")
            checks["tesseract_languages"] = {"status": "ok", "available": langs}
        except Exception as e:
            checks["tesseract_languages"] = {"status": "error", "detail": str(e)}

    if DEP["requests"]:
        checks["requests"] = {"status": "ok"}
    else:
        checks["requests"] = {"status": "missing", "fix": "pip install requests", "note": "Required for Vision API"}

    if DEP["sklearn"]:
        checks["sklearn"] = {"status": "ok"}
    else:
        checks["sklearn"] = {"status": "missing (optional)", "fix": "pip install scikit-learn", "note": "Falls back to quantize for color analysis"}

    if DEP["vision_api_key"]:
        checks["siliconflow_api_key"] = {"status": "ok"}
    else:
        checks["siliconflow_api_key"] = {"status": "not set", "fix": "export SILICONFLOW_API_KEY=your_key", "note": "Required for Vision API"}

    if DEP["vision_api_key"] and DEP["requests"]:
        try:
            test_resp = _req.get(
                "https://api.siliconflow.cn/v1/models",
                headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
                timeout=10,
            )
            if test_resp.status_code == 200:
                models = test_resp.json().get("data", [])
                has_vision = any("VL" in m.get("id", "") for m in models)
                checks["siliconflow_api"] = {
                    "status": "ok",
                    "models_count": len(models),
                    "vision_model_available": has_vision,
                }
            else:
                checks["siliconflow_api"] = {"status": "error", "code": test_resp.status_code, "detail": test_resp.text[:200]}
        except Exception as e:
            checks["siliconflow_api"] = {"status": "error", "detail": str(e)}

    return checks


def install_deps():
    import subprocess
    required = []
    optional = []

    if not DEP["pillow"]:
        required.append("Pillow")
    if not DEP["numpy"]:
        required.append("numpy")
    if not DEP["pytesseract"]:
        required.append("pytesseract")
    if not DEP["requests"]:
        optional.append("requests")
    if not DEP["sklearn"]:
        optional.append("scikit-learn")

    if not required and not optional:
        print(json.dumps({"status": "ok", "message": "All dependencies already installed."}, ensure_ascii=False, indent=2))
        return

    all_pkgs = required + optional
    print(json.dumps({"status": "installing", "required": required, "optional": optional}, ensure_ascii=False, indent=2))
    print("---")

    try:
        cmd = [sys.executable, "-m", "pip", "install"] + all_pkgs
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0:
            print(json.dumps({"status": "ok", "installed": all_pkgs}, ensure_ascii=False, indent=2))
        else:
            print(json.dumps({"status": "error", "detail": result.stderr[-500:]}, ensure_ascii=False, indent=2))
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print(json.dumps({"status": "error", "detail": "pip install timed out (180s)"}, ensure_ascii=False, indent=2))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"status": "error", "detail": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Image analysis tool for Claude Code")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--ocr-lang", default="eng", help="OCR language (default: eng, use chi_sim+eng for Chinese)")
    parser.add_argument("--vision", action="store_true", help="Enable SiliconFlow vision API for image understanding")
    parser.add_argument("--check", action="store_true", help="Check dependencies and exit")
    parser.add_argument("--install-deps", action="store_true", help="Install missing Python dependencies")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    args = parser.parse_args()

    if args.install_deps:
        install_deps()
        sys.exit(0)

    if args.check:
        checks = check_dependencies()
        output = json.dumps(checks, ensure_ascii=False, indent=2 if args.pretty else None)
        print(output)
        critical = ["pillow"]
        has_errors = any(checks.get(c, {}).get("status") == "missing" for c in critical)
        sys.exit(1 if has_errors else 0)

    if not args.image:
        parser.print_help()
        sys.exit(1)

    result = analyze_image(args.image, ocr_lang=args.ocr_lang, enable_vision=args.vision)
    output = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
    print(output)

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()