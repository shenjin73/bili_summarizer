# Bili Summarizer

一个把 B 站视频变成中文 AI 摘要的命令行工具。

输入一个 B 站视频链接或 BV 号，工具会自动抓取视频的元数据、简介、CC 字幕和热门评论，交给大语言模型生成一份结构化的中文 Markdown 总结。如果视频没有 CC 字幕，会自动下载音频并用 Whisper 转写，保证绝大多数视频都能被总结。

> ⚠️ 生成的摘要仅供学习参考，不构成任何投资建议。

## 功能特性

- 🎬 **自动抓取**视频元数据、简介、CC 字幕、前 20 条热门评论
- 🎤 **双重字幕来源**：优先使用官方 CC 字幕；没有字幕时自动下载音频，用 Whisper 转写
- 🤖 **多 LLM 后端**：支持 Anthropic / DeepSeek / MiniMax / OpenAI 及任意 OpenAI 兼容 API，按环境变量自动检测
- 📝 **结构化总结**：中文 Markdown 输出，包含视频概述、核心主题、关键观点、重要细节、评论视角、总体评价与参考价值评分
- 💾 **按 UP 主归档**：所有文件保存在 `videos/<UP主>/<BV号>/` 目录下，方便日后检索

## 工作原理

```
main.py ──→ fetcher.py ──→ transcriber.py ──→ summarizer.py ──→ storage.py
            抓取 B 站数据     Whisper 转写        LLM 总结          保存文件
```

1. **解析 BV 号** —— 从链接中提取 `BV...` 视频 ID（支持完整链接与短链）
2. **抓取元数据** —— 标题、UP 主、简介、时长、播放/点赞/投币等数据
3. **获取字幕**
   - 先查官方 CC 字幕（优先选择中文语言的字幕）
   - 若无字幕，下载音频（DASH 最优音质流）→ Whisper 转写 → 清理音频文件
4. **获取评论** —— 前 20 条热门评论（按热度排序）
5. **保存原始数据** —— metadata / description / subtitles / comments 全部落盘
6. **LLM 总结** —— 把以上内容拼接后交给大模型，生成总结并保存

## 安装

需要 **Python 3.10+**。

```bash
git clone <repo-url>
cd bili_summarizer

# 基础依赖
pip install -r requirements.txt

# 如需 Whisper 转写（无 CC 字幕的视频），额外安装（需 Apple Silicon）：
pip install mlx-whisper
```

> `mlx-whisper` 依赖 `ffmpeg` 解码音频。若机器上还没有，请先用 `brew install ffmpeg` 安装。

## 配置

### 1. 设置 LLM API Key

复制 `.env.example` 为 `.env`（或直接在 shell 中 export），配置任意一种模型的 Key 即可：

```bash
# Anthropic（Claude）
export ANTHROPIC_API_KEY=sk-ant-xxxx

# 或 DeepSeek
export DEEPSEEK_API_KEY=sk-xxxx

# 或 MiniMax
export MINIMAX_API_KEY=xxxx

# 或 OpenAI
export OPENAI_API_KEY=sk-xxxx

# 或任意 OpenAI 兼容 API（需同时指定模型名与 Base URL）
export LLM_API_KEY=sk-xxxx
export LLM_MODEL=your-model
export LLM_BASE_URL=https://your-endpoint/v1
```

**Provider 检测顺序**：Anthropic → DeepSeek → MiniMax → OpenAI → 自定义。

- 强制指定 Provider：`export LLM_PROVIDER=deepseek`
- 覆盖默认模型：`export LLM_MODEL=<模型名>`
- 覆盖 API 地址：`export LLM_BASE_URL=<地址>`

| Provider   | 环境变量              | 默认模型             |
|------------|----------------------|----------------------|
| Anthropic  | `ANTHROPIC_API_KEY`  | `claude-sonnet-4-6`  |
| DeepSeek   | `DEEPSEEK_API_KEY`   | `deepseek-chat`      |
| MiniMax    | `MINIMAX_API_KEY`    | `abab6.5s-chat`      |
| OpenAI     | `OPENAI_API_KEY`     | `gpt-4o`             |
| 自定义     | `LLM_API_KEY` + `LLM_MODEL` + `LLM_BASE_URL` | 必须自定 |

> 未配置任何 Key 时，工具仍会抓取并保存所有数据，只是跳过总结步骤。

### 2. HuggingFace 镜像（可选）

Whisper 模型默认从 `hf-mirror.com` 下载（方便国内网络）。如需使用官方源：

```bash
export HF_ENDPOINT=https://huggingface.co
```

## 使用方法

```bash
cd bili_summarizer

# 传完整链接
python3 -m bili_summarizer.main https://www.bilibili.com/video/BV1CNLQ6REu5/

# 传短链
python3 -m bili_summarizer.main https://b23.tv/xxxxx

# 直接传 BV 号
python3 -m bili_summarizer.main BV1CNLQ6REu5
```

运行过程中会打印每一步的进度，结束时把 AI 总结打印在终端。

## 输出结构

每个视频的数据存放在 `videos/<UP主名>/<BV号>/` 目录下：

```
videos/
└── <UP主名>/
    └── <BV号>/
        ├── metadata.json      # 原始视频元数据（含标题、UP主、统计数据）
        ├── description.txt    # 视频简介纯文本
        ├── subtitles.json     # CC 字幕原始 JSON（按时间轴分段）
        ├── subtitles.txt      # 字幕纯文本（Whisper 转写也输出到此处）
        ├── comments.json      # 前 20 条热门评论
        └── summary.md         # AI 生成的 Markdown 总结 ★
```

`summary.md` 包含以下板块：

- **视频概述**、**核心主题**、**关键观点与论证**
- **重要细节**（具体数据、案例、方法论）
- **评论视角**（如有评论数据）
- **总体评价** 与 **参考价值评估**（数据支撑度、可复现性、观点独特性等 1-5 星评分）

## 说明与限制

- **字幕缺失的降级**：当视频没有 CC 字幕且音频转写失败时，总结会基于简介和评论生成，并在开头标注「⚠️ 本总结基于视频简介和评论生成」。
- **内容截断**：字幕与评论拼接后超过约 80,000 字符的部分会被截断，避免超出模型上下文。
- **接口依赖**：B 站未开放的接口随时可能调整（如 WBI 签名机制），若抓取失败请更新 `fetcher.py` 或等待修复。
- **音频转写**：仅支持 Apple Silicon（依赖 `mlx-whisper`），默认模型为 `whisper-medium`，可在 `transcriber.py` 中调整。

## 项目结构

```
bili_summarizer/
├── main.py          # 入口，编排整个流程
├── fetcher.py       # B 站 API 封装（WBI 签名、元数据、字幕、评论、音频下载）
├── transcriber.py   # mlx-whisper 音频转写
├── summarizer.py    # 多 Provider LLM 总结
└── storage.py       # 按 <UP主>/<BV号> 归档保存文件
```

## License

MIT
