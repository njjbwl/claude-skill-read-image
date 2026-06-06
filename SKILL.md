---
name: read-image
version: 2.1.0
description: Use when the user asks you to look at, describe, read text from, or analyze
  an image/screenshot/photo. Enables Claude to "see" images by using local Python tools
  (Pillow + Tesseract OCR) for metadata/color/text extraction, with optional SiliconFlow
  Qwen3-VL-32B vision API integration for advanced image understanding. Also use when
  you need to extract information from images that you cannot directly view. Triggers: "看图",
  "看图片", "read image", "describe image", "OCR this", "what's in this image",
  "提取图片文字", "SiliconFlow 看图".
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

# Read Image v2.1 — 图片分析技能

通过 Python + Pillow + Tesseract OCR + SiliconFlow Qwen3-VL-32B 分析图片。

## v2.1 新增

| 改进 | 说明 |
|------|------|
| 🧠 **多维启发式判断** | 综合亮度/颜色/OCR/尺寸判断 7 种图片类型 |
| 📦 **一键安装** | `--install-deps` 自动装全 Python 依赖 |
| 🖼️ **图片分类** | 区分终端截图/代码截图/文档扫描/照片/图表/UI/手机截图 |

## 什么时候用

- 用户说「帮我看这张图」「OCR 这张图」「这张图里有什么」
- 需要从截图/图片中提取文字信息
- 需要分析图片的尺寸、颜色、亮度等属性
- 需要分析终端截图来理解用户看到的界面
- 需要真正理解图片内容（场景、物体、布局）→ 加 `--vision`

## 使用

```bash
# 基础分析（英文OCR）
python ~/.claude/skills/read-image/analyze_image.py <图片路径>

# 中文OCR
python ~/.claude/skills/read-image/analyze_image.py <图片路径> --ocr-lang chi_sim+eng

# Vision API 看图（需要 SILICONFLOW_API_KEY 环境变量）
python ~/.claude/skills/read-image/analyze_image.py <图片路径> --vision

# 自检所有依赖
python ~/.claude/skills/read-image/analyze_image.py --check --pretty

# 一键安装缺失的 Python 依赖
python ~/.claude/skills/read-image/analyze_image.py --install-deps
```

## 输出字段

| 字段 | 说明 |
|------|------|
| `metadata` | 格式、尺寸、宽高比、文件大小、DPI |
| `exif` | 拍摄设备、GPS 等元数据（如有） |
| `downscaled` | 如果大图自动缩放了，显示原始/新尺寸 |
| `colors.dominant` | 主色调（k-means 聚类，含 RGB/Hex/百分比） |
| `colors.method` | `kmeans` 或 `quantize`（取决于 sklearn 是否安装） |
| `luminance` | 亮度均值/标准差/最暗最亮/明暗比例 |
| `heuristics` | 图片类型分类结果（见下表） |
| `ocr.text` | OCR 识别出的文字（最多 3000 字符，含截断标记） |
| `ocr.lines` | 识别出的文本行数 |
| `ocr.words` | 识别的单词数 |
| `ocr.word_details` | 逐词详情（含 bounding box 和置信度） |
| `ocr.avg_confidence` | OCR 平均置信度 |
| `vision.description` | Vision API 返回的图片内容描述 |
| `dependencies` | 各依赖组件的可用状态 |

### heuristics 字段详解

| 字段 | 触发条件 | 含义 |
|------|---------|------|
| `terminal_screenshot` | 暗色背景 + 亮色文字 + 有 OCR 文本 | 深色终端截图，关注 `ocr.text` |
| `code_screenshot` | 亮色背景 + 高对比度 + 有 OCR 文本 | 代码/浅色终端截图 |
| `document_scan` | 亮色背景 + 单一主色调 + 有文本 | 扫描文档/白底黑字 |
| `photo` | 大文件/色彩丰富无文字/高对比度无文字 | 照片，关注 `colors`、`luminance`、`exif` |
| `chart_diagram` | 色彩丰富 + 有文本 + 非照片 | 图表/示意图 |
| `mobile_screenshot` | 竖屏比例 + 有文本 | 手机截图 |
| `text_screenshot` | 有文本 + 非照片 | 任意含文字的截图（综合） |
| `widescreen` | 宽高比 ≈ 16:9 | 宽屏画面 |
| `portrait` | 高度 > 宽度 | 竖屏画面 |

## 依赖容错

脚本会自动检测可用库，缺的自动降级：

| 缺什么 | 影响 | 降级行为 |
|--------|------|---------|
| Pillow | ❌ 完全不能用 | 报错退出 |
| numpy | ⚠️ 部分 | 亮度分析不可用 |
| pytesseract | ⚠️ OCR | OCR 段跳过 |
| Tesseract 二进制 | ⚠️ OCR | OCR 段跳过 |
| scikit-learn | ⚠️ 颜色 | 用 quantize 代替 k-means |
| requests | ⚠️ Vision API | Vision API 不可用 |
| SILICONFLOW_API_KEY | ⚠️ Vision API | Vision API 不可用 |

## 自检与安装

```bash
# 自检
python ~/.claude/skills/read-image/analyze_image.py --check --pretty

# 一键安装所有 Python 依赖（含可选包）
python ~/.claude/skills/read-image/analyze_image.py --install-deps
```

自检输出示例：
```json
{
  "pillow": { "status": "ok", "version": "11.3.0" },
  "numpy": { "status": "ok", "version": "2.4.4" },
  "tesseract_binary": { "status": "ok", "version": "tesseract 5.3.3" },
  "tesseract_languages": { "status": "ok", "available": ["eng", "chi_sim"] },
  "siliconflow_api": { "status": "ok", "models_count": 93, "vision_model_available": true }
}
```

## 环境变量

| 变量 | 用途 | 配置位置 |
|------|------|---------|
| `SILICONFLOW_API_KEY` | Vision API 密钥 | `~/.claude/settings.json` → `env` |
| `TESSDATA_PREFIX` | 中文 OCR 语言包路径 | `~/.claude/settings.json` → `env` |

Vision API 密钥从 https://cloud.siliconflow.cn/account/ak 获取。
已在 `settings.json` 中永久配置，后续会话自动生效。

## 给 AI 的使用指南

当你需要「看图」时，按以下步骤操作：

### 快速工作流

1. **用户给了图片路径** → 直接运行
   ```bash
   python ~/.claude/skills/read-image/analyze_image.py <路径>
   ```

2. **需要截屏** → 两步走：
   - 用 Playwright/computer-use MCP 截屏保存到临时文件
   - 用本脚本分析

3. **需要理解图片内容**（不仅仅是读字）→ 加 `--vision`

### 结果解读规则

根据 `heuristics` 判断图片类型，聚焦对应的输出字段：

| 检测到 | 优先看 | 说明 |
|--------|--------|------|
| `terminal_screenshot` | `ocr.text` | 终端输出内容，直接读文字 |
| `code_screenshot` | `ocr.text` + `colors` | 代码截图，看文字和语法高亮色 |
| `document_scan` | `ocr.text` | 文档内容 |
| `photo` | `colors` + `luminance` + `exif` | 照片属性，必要时用 `--vision` |
| `chart_diagram` | `ocr.text` + `colors.dominant` | 图表数据和配色 |
| `mobile_screenshot` | `ocr.text` | 手机界面内容 |
| （无分类） | 综合所有字段 | 先看 `metadata` 了解基本尺寸 |

### 截断处理

如果 `ocr.truncated: true`，分段读取后续文字：

```bash
# 第一段已自动截取前 3000 字符
# 读取后续内容：用 --ocr-lang 配合 head/tail
python -c "
import json, sys
# 第二次运行截取从 3000 字符开始
" 
```

（或直接加 `--vision` 让 Vision API 一次性理解完整内容）

## 安装

### Python 依赖

```bash
# 自动安装（推荐）
python ~/.claude/skills/read-image/analyze_image.py --install-deps

# 或手动
pip install Pillow numpy pytesseract requests
pip install scikit-learn  # 可选，颜色分析用 k-means 代替 quantize
```

### Tesseract 二进制

- **Windows**: https://github.com/UB-Mannheim/tesseract/wiki
- **macOS**: `brew install tesseract`
- **Linux**: `apt install tesseract-ocr`

### 中文 OCR 语言包

已安装在 `~/.tesseract/tessdata/`，`TESSDATA_PREFIX` 已配置好。
如需手动更新：

```bash
curl -L -o ~/.tesseract/tessdata/chi_sim.traineddata \
  https://github.com/tesseract-ocr/tessdata_fast/raw/main/chi_sim.traineddata
```

## 已知限制

- OCR 对手写体/艺术字效果差（可用 `--vision` 弥补）
- Vision API 需要网络连接
- Vision API 不支持视频分析
- 启发式判断对极端特殊的图片可能不准（此时应加 `--vision`）
