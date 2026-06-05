# 🖼️ Claude Code Read Image Skill — Claude 接入 DeepSeek 看图技能

> 让 Claude Code 也能"看图"——通过本地 Python 工具（Pillow + Tesseract OCR）+ DeepSeek 视觉 API，实现图片元数据提取、颜色分析、OCR 文字识别和图像内容理解。
> Let Claude Code "see" images — local Python analysis (Pillow + Tesseract OCR) plus DeepSeek vision API for metadata extraction, color analysis, OCR, and image content understanding.

## 能力 / Capabilities

当用户说「帮我看这张图」，Claude 可以：

| 能力 | 说明 |
|------|------|
| 📐 元数据 | 图片格式、尺寸、宽高比、文件大小 |
| 🎨 颜色分析 | 主色调提取（含 RGB/Hex 值） |
| 🔦 亮度分析 | 平均亮度、标准差、明暗比例 |
| 🖥️ 截图识别 | 自动判断是否为终端截图 |
| 📝 OCR 英文 | 用 Tesseract 提取图片中的英文字 |
| 📷 EXIF 数据 | 拍摄设备、GPS 信息等 |

## 安装 / Installation

### 前置要求 / Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)
- Python 3.8+ with `pip`
- Tesseract OCR（可选，仅 OCR 需要）

### 步骤 / Steps

```bash
# 1. 安装 Python 依赖
pip install Pillow numpy

# 2. 安装 Tesseract OCR（Windows）
choco install tesseract

# 3. 克隆本仓库
git clone https://github.com/njjbwl/claude-skill-read-image.git ~/.claude/skills/read-image

# 4. 安装 pytesseract（可选，仅 OCR 需要）
pip install pytesseract

# 5. 在会话中调用
# 输入: /read-image
```

### 中文 OCR（可选）

```bash
# 下载中文语言包
curl -L -o "/c/Program Files/Tesseract-OCR/tessdata/chi_sim.traineddata" \
  https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata
```

## 使用方法 / Usage

```bash
# 基础分析（英文 OCR）
python ~/.claude/skills/read-image/analyze_image.py image.png

# 中英文 OCR
python ~/.claude/skills/read-image/analyze_image.py image.png --ocr-lang chi_sim+eng
```

或在 Claude 会话中说：
- 「帮我看这张图」
- 「OCR 这个截图」
- 「这张图里有什么文字」

## 文件结构 / File Structure

```
~/.claude/skills/read-image/
├── SKILL.md              # 技能定义
├── analyze_image.py      # Python 分析脚本
└── README.md             # 本文件
```

## 工作原理 / How It Works

`read-image` 技能使用多层次架构来分析图片：

1. **Pillow + NumPy** — 图片元数据、EXIF、颜色、亮度分析
2. **Tesseract OCR** — 从图片中提取文字
3. **DeepSeek 视觉 API（可选）** — 高级图像内容理解

Claude 收到「看图」请求后，自动调用相应工具，然后将分析结果用自然语言告诉你。

## 许可 / License

MIT — 随意使用、修改、分发。
