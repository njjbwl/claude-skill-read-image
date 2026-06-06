# 🖼️ Claude Code Read Image Skill v2.1 — Claude Code 看图技能

> 让 Claude Code 真正「看懂」图片——通过本地 Python 工具（Pillow + Tesseract OCR）+ SiliconFlow Qwen3-VL-32B 视觉 API，实现图片元数据提取、颜色分析、OCR 文字识别和图像内容理解。
> Let Claude Code truly "see" images — local Python analysis (Pillow + Tesseract OCR) plus SiliconFlow Qwen3-VL-32B vision API for comprehensive image understanding.

## 能力 / Capabilities

| 能力 | 说明 | v2.1 改进 |
|------|------|-----------|
| 📐 **元数据** | 图片格式、尺寸、宽高比、DPI、EXIF | |
| 🎨 **颜色分析** | k-means 主色调聚类，含 RGB/Hex/占比% | ⬆️ 从 quantize(6) 升级为 k-means |
| 🔦 **亮度分析** | 平均亮度、标准差、明暗比例 | |
| 🖥️ **截图分类** | 多维启发式判断（终端截图/代码截图/文档/照片/图表/手机截图） | ⬆️ 从二元判断升级为 7 类 |
| 📝 **OCR 英文** | Tesseract 提取文字，含 bounding box + 置信度 | ⬆️ 新增结构化输出 |
| 🀄 **OCR 中文** | 支持 chi_sim+eng 双语识别 | ⬆️ 新增 |
| 📷 **EXIF 数据** | 拍摄设备、GPS 信息等 | |
| 🧠 **Vision API** | SiliconFlow Qwen3-VL-32B 真正理解图片内容 | ⬆️ 从 DeepSeek 更换为 Qwen3-VL |
| ✅ **自检模式** | `--check` 一键验证所有依赖 | ⬆️ 新增 |
| 📦 **一键安装** | `--install-deps` 自动补全 Python 依赖 | ⬆️ 新增 |
| 🛡️ **依赖容错** | 缺库自动降级，不崩 | ⬆️ 新增 |

## 安装 / Installation

### 前置要求 / Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)
- Python 3.8+ with `pip`
- Tesseract OCR（可选，仅 OCR 需要）

### 步骤 / Steps

```bash
# 1. 克隆本仓库到技能目录
git clone https://github.com/njjbwl/claude-skill-read-image.git ~/.claude/skills/read-image

# 2. 一键安装 Python 依赖
cd ~/.claude/skills/read-image
python analyze_image.py --install-deps

# 3. 验证安装
python analyze_image.py --check --pretty
```

### Vision API 配置（可选但推荐）

1. 注册 [SiliconFlow（硅基流动）](https://siliconflow.cn) → 免费送 ¥14 额度
2. 在 [API Key 管理](https://cloud.siliconflow.cn/account/ak) 新建密钥
3. 配置到 Claude Code 全局设置：

```json
// ~/.claude/settings.json
{
  "env": {
    "SILICONFLOW_API_KEY": "sk-你的密钥"
  }
}
```

### 中文 OCR（可选）

已内置在技能目录中，无需额外操作。
如需手动更新语言包：

```bash
curl -L -o ~/.tesseract/tessdata/chi_sim.traineddata \
  https://github.com/tesseract-ocr/tessdata_fast/raw/main/chi_sim.traineddata
```

并进行设置：

```json
// ~/.claude/settings.json
{
  "env": {
    "TESSDATA_PREFIX": "C:/Users/你的用户名/.tesseract/tessdata"
  }
}
```

## 使用方法 / Usage

```bash
# 基础分析（英文OCR）
python ~/.claude/skills/read-image/analyze_image.py image.png

# 中文 OCR
python ~/.claude/skills/read-image/analyze_image.py image.png --ocr-lang chi_sim+eng

# Vision API 看图
python ~/.claude/skills/read-image/analyze_image.py image.png --vision

# 自检依赖
python ~/.claude/skills/read-image/analyze_image.py --check --pretty
```

在 Claude 会话中说：
- 「帮我看这张图」
- 「OCR 这个截图」
- 「这张图里有什么」
- 「用 vision 看一下这张图」
- 「这是什么类型的图片」

## 输出示例 / Output Example

```json
{
  "metadata": { "format": "PNG", "width": 1920, "height": 1080 },
  "colors": {
    "dominant": [
      { "hex": "#1e1e28", "percentage": 94.1 },
      { "hex": "#00ff64", "percentage": 2.6 }
    ],
    "method": "kmeans"
  },
  "luminance": { "mean": 32.0, "std": 10.1 },
  "heuristics": {
    "terminal_screenshot": true,
    "text_screenshot": true
  },
  "ocr": {
    "text": "Hello World!",
    "lines": 4,
    "words": 15,
    "avg_confidence": 82.7,
    "word_details": [
      { "text": "Hello", "confidence": 95, "bbox": { "x": 20, "y": 20, "w": 40, "h": 15 } }
    ]
  },
  "vision": {
    "description": "这张图片是一张深色背景的终端截图...",
    "model": "Qwen/Qwen3-VL-32B-Instruct"
  }
}
```

## 启发式分类 / Heuristics

脚本自动识别图片类型，帮助 AI 聚焦关键信息：

| 分类 | 说明 | 聚焦字段 |
|------|------|---------|
| `terminal_screenshot` | 深色终端截图 | `ocr.text` |
| `code_screenshot` | 亮色代码截图 | `ocr.text` + `colors` |
| `document_scan` | 白底文档扫描 | `ocr.text` |
| `photo` | 自然照片 | `colors` + `luminance` + `exif` |
| `chart_diagram` | 图表/示意图 | `ocr.text` + `colors` |
| `mobile_screenshot` | 手机截图 | `ocr.text` |
| `text_screenshot` | 含文字的截图（综合） | `ocr.text` |

## 文件结构 / File Structure

```
~/.claude/skills/read-image/
├── SKILL.md              # 技能定义（给 AI 的指令）
├── analyze_image.py      # Python 分析脚本（690 行）
└── README.md             # 本文档
```

## 架构 / Architecture

```
┌─ analyze_image.py ─────────────────────────────────┐
│                                                     │
│  1. Metadata & EXIF  ← Pillow                      │
│  2. Downscale (if >2000px)  ← Pillow LANCZOS       │
│  3. Color Analysis  ← k-means/sklearn or quantize  │
│  4. Luminance  ← numpy                             │
│  5. OCR  ← Tesseract (with preprocessing)          │
│  6. Heuristics  ← multi-signal classification      │
│  7. Vision API  ← SiliconFlow Qwen3-VL-32B         │
│                                                     │
│  Fallback: every module degrades gracefully         │
└─────────────────────────────────────────────────────┘
```

## 版本历史 / Changelog

| 版本 | 日期 | 变更 |
|------|------|------|
| v2.1 | 2026-06-07 | 多维启发式判断、`--install-deps`、改进图片分类 |
| v2.0 | 2026-06-06 | Vision API 集成、k-means 颜色、OCR 预处理、降级容错 |
| v1.0 | 2026-06-05 | 初始版本：Pillow + Tesseract 基础分析 |

## 许可 / License

MIT — 随意使用、修改、分发。
