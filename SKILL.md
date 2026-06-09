---
name: read-image
version: 2.1.1
description: Use when the user asks you to look at, describe, read text from, or analyze
  an image/screenshot/photo. Enables Claude to "see" images by using local Python tools
  (Pillow + Tesseract OCR) for metadata/color/text extraction, with optional SiliconFlow
  Qwen3-VL-32B vision API integration for advanced image understanding.
triggers:
  - 看图
  - 看图片
  - read image
  - describe image
  - screenshot analysis
  - OCR
  - 提取图片文字
  - image analysis
---

# Read Image v2.1.1 — 图片分析技能

通过 Python + Pillow + Tesseract OCR + SiliconFlow Qwen3-VL-32B 分析图片。

## v2.1.1 新增

| 改进 | 说明 |
|------|------|
| OCR 多语言修复 | 修复 chi_sim+eng 格式被错误降级的问题 |
| 预处理阈值优化 | 亮色背景二值化不再洗掉文字笔画 |

## 使用

```bash
python analyze_image.py <图片路径>
python analyze_image.py <图片路径> --ocr-lang chi_sim+eng
python analyze_image.py <图片路径> --vision
python analyze_image.py --check --pretty
```

## 输出字段

metadata, colors, luminance, ocr, heuristics, vision, dependencies

## 安装

```bash
python analyze_image.py --install-deps
```

Tesseract: https://github.com/UB-Mannheim/tesseract/wiki