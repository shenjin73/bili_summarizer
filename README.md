# Bili Summarizer

一个把 B 站视频变成中文 AI 摘要的命令行工具。

输入一个 B 站视频链接或 BV 号，工具会自动抓取视频的元数据、简介、CC 字幕和热门评论，交给大语言模型生成一份结构化的中文 Markdown 总结。如果视频没有 CC 字幕，会自动下载音频并用 Whisper 转写，保证绝大多数视频都能被总结。

> ⚠️ 生成的摘要仅供学习参考，不构成任何投资建议。

## 功能特性

- 🎬 **自动抓取**视频元数据、简介、CC 字幕、前 20 条热门评论
- 🎤 **双重字幕来源**：优先使用官方 CC 字幕；没有字幕时自动下载音频，用 Whisper 转写
- 🖼️ **画面分析**：每 10 秒截图一帧，感知哈希去重（静态幻灯片只保留一张），再用视觉大模型 OCR 识别画面文字并描述图表/图标，补全语音字幕覆盖不到的信息
- 🤖 **多 LLM 后端**：支持 Anthropic / DeepSeek / MiniMax / OpenAI 及任意 OpenAI 兼容 API（含本地 LM Studio），按环境变量自动检测
- 📝 **结构化总结**：中文 Markdown 输出，包含视频概述、核心主题、关键观点、重要细节、评论视角、总体评价与参考价值评分
- 💾 **按 UP 主归档**：所有文件保存在 `videos/<UP主>/<BV号>/` 目录下，方便日后检索

## 工作原理

```
main.py ──→ fetcher.py ──→ transcriber.py ──→ frames.py ──→ summarizer.py ──→ storage.py
            抓取 B 站数据     Whisper 转写       截图+OCR        LLM 总结          保存文件
```

1. **解析 BV 号** —— 从链接中提取 `BV...` 视频 ID（支持完整链接与短链）
2. **抓取元数据** —— 标题、UP 主、简介、时长、播放/点赞/投币等数据
3. **获取字幕**
   - 先查官方 CC 字幕（优先选择中文语言的字幕）
   - 若无字幕，下载音频（DASH 最优音质流）→ Whisper 转写 → 清理音频文件
4. **画面分析**（可用 `--no-frames` 跳过）
   - 下载视频流（默认 720p，DASH）→ ffmpeg 每 10 秒截一帧
   - **感知哈希去重**：与上一张保留帧的 dHash 汉明距离 ≤ 阈值的视为重复画面，删除（静态幻灯片/重复标题页只留一张）
   - **视觉模型 OCR**：对每张唯一截图调用视觉大模型，提取画面文字（标题、数据标签、代码等）并描述图表/图标
   - 清理视频文件，保留唯一截图于 `frames/`
5. **获取评论** —— 前 20 条热门评论（按热度排序）
6. **保存原始数据** —— metadata / description / subtitles / ocr_text / comments 全部落盘
7. **LLM 总结** —— 把以上内容（含画面 OCR）拼接后交给大模型，生成总结并保存

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

复制 `.env.example` 为 `.env`（或直接在 shell 中 export），配置任意一种模型的 Key 即可。**`.env` 会在包导入时自动加载**；已存在的 shell 环境变量优先，不会被覆盖：

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

# 或本地模型（默认优先；LM Studio 需先加载模型并开启本地服务器）
export LOCAL_API_KEY=lm-studio-local   # 占位即可，本地服务器不校验
export LOCAL_MODEL=qwen3.8-27b-mlx     # LM Studio 中的模型 ID
export LOCAL_BASE_URL=http://127.0.0.1:1234/v1
# 高级：多个本地端点（逗号分隔的链，按顺序尝试，全部不在线才切云端）
# export LOCAL_BASE_URLS=http://127.0.0.1:8000/v1,http://127.0.0.1:1234/v1
# export LOCAL_MODELS=model-a,model-b   # 按位置对应；单个值应用到所有端点
# export LOCAL_API_KEYS=key-a,lm-studio-local
```

> 💡 **本地视觉模型推荐**：画面 OCR 需要一个支持图片输入的模型。`qwen3.8-27b`（MLX 版）同时支持文本与视觉，一个模型即可承担「总结 + OCR」两个角色。若用纯文本模型做总结、另配视觉模型做 OCR，可用 `VISION_*` 变量单独指定（见下）。

**Provider 检测顺序**：本地 `LOCAL_*` → DeepSeek → 其余云端 → 自定义 `LLM_*`。

**本地离线自动降级**：启动时健康检查本地端点（`GET /models`，5 秒超时）；不在线或模型未加载时，若设置了 `DEEPSEEK_API_KEY`，总结与画面 OCR 都会自动切换到 DeepSeek 视觉模型（`deepseek-flash`，支持图片输入）并打印提示。仅当本地与 DeepSeek 都不可用时 OCR 才跳过，总结基于字幕生成。调用中途本地服务挂掉时也会自动切到 DeepSeek 重试；OCR 批次中若过半帧因连接失败，失败帧会在下一个可用视觉端点上重跑一次。

- 强制指定 Provider：`export LLM_PROVIDER=deepseek`（或 `local`）
- 覆盖云端 Provider 的默认模型：`export LLM_MODEL=<模型名>`
- 覆盖云端 Provider 的 API 地址：`export LLM_BASE_URL=<地址>`

| Provider   | 环境变量              | 默认模型             |
|------------|----------------------|----------------------|
| 本地       | `LOCAL_MODEL` + `LOCAL_BASE_URL`（+可选 `LOCAL_API_KEY`）；多端点可用复数形式 | 必须自定，如 `qwen3.8-27b-mlx` |
| Anthropic  | `ANTHROPIC_API_KEY`  | `claude-sonnet-4-6`  |
| DeepSeek   | `DEEPSEEK_API_KEY`   | `deepseek-flash` |
| MiniMax    | `MINIMAX_API_KEY`    | `abab6.5s-chat`      |
| OpenAI     | `OPENAI_API_KEY`     | `gpt-4o`             |
| 自定义     | `LLM_API_KEY` + `LLM_MODEL` + `LLM_BASE_URL` | 必须自定 |

> 未配置任何 Key 时，工具仍会抓取并保存所有数据，只是跳过总结步骤。

### 2. 画面分析参数（可选）

截图 / OCR 环节的行为可用环境变量微调：

| 变量 | 默认 | 说明 |
|------|------|------|
| `FRAME_INTERVAL` | `10` | 截图间隔（秒） |
| `FRAME_DEDUP_THRESHOLD` | `10` | 去重阈值：与上一张保留帧的 dHash 汉明距离 ≤ 该值视为重复（0-64，调大更激进） |
| `FRAME_QN` | `64` | 下载视频流清晰度：16=360p / 32=480p / 64=720p / 80=1080p |
| `FRAME_MAX_OCR` | `80` | 唯一截图超过该数量时均匀采样，控制 OCR 耗时（长视频建议保留默认） |
| `OCR_CONCURRENCY` | 本地 `2` / 云端 `6` | OCR 并发线程数（本地 LM Studio 串行处理能力有限，默认压低；云端可调大） |

### 3. 总结输入限长（可选）

总结 prompt 的每个板块**独立限长**（超长时保留头尾、丢弃中间），避免超长字幕把 OCR / 评论板块整体挤出：

| 变量 | 默认 | 说明 |
|------|------|------|
| `SUMMARY_MAX_SUBTITLE_CHARS` | `50000` | 字幕板块上限（字符） |
| `SUMMARY_MAX_OCR_CHARS` | `15000` | 画面 OCR 板块上限（字符） |
| `SUMMARY_MAX_DESC_CHARS` | `4000` | 简介板块上限（字符） |
| `SUMMARY_MAX_COMMENT_CHARS` | `3000` | 评论板块上限（字符） |

### 4. Whisper 转写（可选）

| 变量 | 默认 | 说明 |
|------|------|------|
| `WHISPER_MODEL` | `large-v3-turbo` | mlx-whisper 模型变体（对应 `mlx-community/whisper-<值>`）。turbo 约为 medium 的数倍速度，精度接近 large-v3 |

转写结果会**立即写入 `subtitles.txt`**；重跑同一视频时若该文件已存在且非空，直接复用并跳过 Whisper（断点续跑）。

### 5. HuggingFace 镜像（可选）

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

# 跳过画面分析（只走字幕路线，更快、不下载视频）
python3 -m bili_summarizer.main BV1CNLQ6REu5 --no-frames

# 自定义截图间隔（每 20 秒一帧）
python3 -m bili_summarizer.main BV1CNLQ6REu5 --interval 20
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
        ├── ocr_text.txt       # 画面 OCR 结果（每行 [时间戳] + 文字/图表描述）
        ├── frames.json        # 逐帧 OCR 清单（时间戳、文件名、识别文本）
        ├── frames/            # 去重后的唯一截图（frame_001.jpg ...）
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
- **音频转写**：仅支持 Apple Silicon（依赖 `mlx-whisper`），默认模型为 `whisper-large-v3-turbo`，可用 `WHISPER_MODEL` 环境变量覆盖。
- **画面分析**：需要本机安装 `ffmpeg`（抽帧用）；会额外下载一份视频流（默认 720p，10 分钟视频约几十 MB），分析完成后自动删除。OCR 依赖视觉模型——本地 LM Studio（默认）或任意 OpenAI 兼容的视觉 API；本地不可用时自动切到 DeepSeek 视觉模型，两者都不可用才跳过 OCR（不影响字幕总结）。

## 项目结构

```
bili_summarizer/
├── main.py          # 入口，编排整个流程（含 --no-frames / --interval）
├── fetcher.py       # B 站 API 封装（WBI 签名、元数据、字幕、评论、音视频下载）
├── transcriber.py   # mlx-whisper 音频转写
├── frames.py        # 画面分析：ffmpeg 抽帧、感知哈希去重、视觉模型 OCR
├── summarizer.py    # 多 Provider LLM 总结（合并字幕 + 画面 OCR）
└── storage.py       # 按 <UP主>/<BV号> 归档保存文件
```

## 常见问题

**Whisper 转写报 `Invalid port: ':1]'`**

`no_proxy` 里若含裸写或带方括号的 IPv6（如 `::1`、`[::1]`），URL 解析器会按冒号切分并在该条目上抛错，导致 HuggingFace 拉取中断——**即使模型权重早已缓存**，转写也会失败。工具会自动剔除这类条目（保留 `localhost` / `127.0.0.1`），无需手工处理；想从源头消除告警可执行：

```bash
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
```

**本地 LM Studio 不在线，OCR 走了云端**

这是预期的降级路径：本地端点健康检查失败后，总结与 OCR 会自动切到 DeepSeek（`deepseek-flash`，支持图片输入），并打印提示。若报「视觉模型不可用」，请检查 `DEEPSEEK_API_KEY` 是否有效。注意：健康检查会核对模型名是否出现在端点的 `/models` 列表里，**云端模型改名或下线时会被判为不可用**（报错会同时给出本地与云端两侧原因）。

**OCR 太慢 / 调用量太大**

用 `FRAME_MAX_OCR` 限制参与 OCR 的唯一截图数、`OCR_CONCURRENCY` 调整并发，或直接 `--no-frames` 只走字幕路线。云端视觉模型按图计费，每张图有约 1024 token 的上限（大于约 1300×1300 的图会被自动缩放）。

## License

MIT
