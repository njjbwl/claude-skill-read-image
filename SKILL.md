---
name: read-image
version: 1.0.0
description: Use when the user asks you to look at, describe, read text from, or analyze
  an image/screenshot/photo. Enables Claude to "see" images by using local Python tools
  (Pillow + Tesseract OCR) for metadata/color/text extraction, with optional DeepSeek
  vision API integration for advanced image understanding. Also use when you need to
  extract information from images that you cannot directly view. Triggers: "看图",
  "看图片", "read image", "describe image", "OCR this", "what's in this image",
  "提取图片文字", "DeepSeek 看图".
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

# Read Image — 图片分析技能（Claude + DeepSeek 看图）

通过 Python + Pillow + Tesseract OCR 分析图片，提取元数据、颜色、文字和统计信息。
可选接入 DeepSeek 视觉 API 实现高级图像内容理解。

## 什么时候用

- 用户说「帮我看这张图」「OCR 这张图」「这张图里有什么」
- 需要从截图/图片中提取文字信息
- 需要分析图片的尺寸、颜色、亮度等属性
- 需要分析终端截图来理解用户看到的界面

## 快速使用

```bash
# 基础分析（英文OCR）
python ~/.claude/skills/read-image/analyze_image.py <图片路径>

# 中文OCR（需要先安装chi_sim语言包）
python ~/.claude/skills/read-image/analyze_image.py <图片路径> --ocr-lang chi_sim+eng

# 截图快速分析
# 1. 用 computer-use MCP 截屏保存
# 2. 用本脚本分析
```

## 分析结果说明

脚本输出 JSON，包含：

| 字段 | 说明 |
|------|------|
| `metadata` | 格式、尺寸、宽高比、文件大小 |
| `exif` | 拍摄设备、GPS 等元数据（如有） |
| `colors.dominant` | 主色调（最多 5 种），含 RGB/Hex |
| `luminance` | 亮度均值/标准差/最暗最亮/明暗比例 |
| `heuristics` | 是否为终端截图/照片的判断 |
| `ocr.text` | OCR 识别出的文字（最多 3000 字符） |

## 给 AI 的使用指南

当你需要「看图」时：

1. **用户给了图片路径** → 直接运行 `analyze_image.py`
2. **需要截屏** → 用 `computer-use` MCP 截屏并保存到文件，再分析
3. **结果解读**：
   - 终端截图 → `likely_terminal_screenshot` 为 true，关注 OCR 文字
   - 照片 → 关注 `colors`、`luminance`、`exif`
   - OCR 文字 → 直接读 `ocr.text`，告诉用户

## DeepSeek 视觉 API（可选）

如需更高级的图像理解，可配置 DeepSeek 视觉 API：

```bash
export DEEPSEEK_API_KEY=your_key_here
python ~/.claude/skills/read-image/analyze_image.py <图片路径> --deepseek
```

需要先安装：`pip install openai`

## 安装中文 OCR（可选）

如果用户需要识别图片中的中文：

```bash
# 下载中文语言包放到 Tesseract 目录
curl -L -o "/c/Program Files/Tesseract-OCR/tessdata/chi_sim.traineddata" \
  https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata
```

安装后使用 `--ocr-lang chi_sim+eng`。

## 已知限制

- 不能像人一样「看懂」图片内容（无法识别物体、人脸、场景）
- OCR 对清晰文本效果好，对手写体/艺术字效果差
- 中文 OCR 需要额外安装语言包
- DeepSeek 视觉 API 需要自行配置 API key
- 不支持视频分析
