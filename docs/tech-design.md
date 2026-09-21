# 知识库构建工具 · 技术方案

版本：v0.2
日期：2026-09-21
关联文档：[product-design.md](product-design.md)（产品设计与验收标准）

> **v0.2 定位调整**：本项目从「知识库问答助手」改定位为「**知识库构建工具**」——
> 面向的不是终端用户，而是 Agent 系统的检索层。主体是两条命令行
> （`rag.ingest` 建库 / `rag.query` 验证检索），Streamlit 界面降级为可选的人工核对工具。
> 理由：这套检索能力的主战场是 [MockMate](https://github.com/spike5321/mockmate)，
> 单独做一个平行的问答应用只会让人问"它和 MockMate 的 `kb/` 什么关系"。
> 同时补齐了切分策略与幂等更新三处修正（见 §4 D6）。

## 1. 总体架构

整体是一个**单机单进程的分层架构**：UI 是薄壳，全部业务逻辑收敛在 `rag/` 包内，模型调用收敛在 `rag/core.py` 一处。

```
┌─────────────────────────────────────────────┐
│  CLI（python -m rag.ingest / rag.query）      │   ← 主体入口
│  app.py（Streamlit 界面）  demo.py（Mock 演示）│   ← 可选：人工核对
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  rag/ 包（核心逻辑，不依赖任何 UI 库）           │
│  ├─ core.py      模型调用唯一入口（embed/chat）│
│  ├─ ingest.py    解析 → 切分 → 向量化 → 入库   │
│  ├─ query.py     只检索、不生成（验证建库质量）  │
│  ├─ pipeline.py  检索 → 拼提示词 → 生成 + 引用 │
│  └─ mock.py      模拟数据层（Demo/测试用）      │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  数据层（本地文件）                            │
│  ├─ data/docs/   用户原始文档                  │
│  └─ db/          Chroma 持久化（collection:    │
│                  knowledge_base）             │
└─────────────────────────────────────────────┘
        │                          │
        ▼                          ▼
  智谱 embedding-3           智谱 glm-4.5-flash
  或本地 bge-small-zh        （生成回答，仅界面用）
  （向量化，无 Key 可用本地）
```

问答主链路（对应产品文档 F2/F3，仅在用界面时走完整链路）：

```
用户提问 → embed(question) → Chroma 检索 top-4
        → 拼装提示词（参考资料 + [1][2] 编号）
        → glm-4.5-flash 生成 → (回答, 引用片段列表) → 界面展示溯源
```

与 MockMate 的关系（同一套算法两个形态）：

| | 独立形态（本仓库） | 内嵌形态（MockMate） |
|---|---|---|
| 存储层 | `rag/ingest.py` | `mockmate/kb/store.py` |
| 向量由谁算 | `rag/core.py` | Agent 的 LLM 客户端（`embed_fn` 注入） |
| 由谁调用 | 命令行 | Agent 工具 `search_jd_kb` |

两边是 vendoring 而非 pip 依赖：个人项目不值得维护发版与版本漂移，一致性靠
`tests/` 的回归测试 + `scripts/compare_chunking.py` 的对照脚本保证。

## 2. 技术选型与理由

| 项 | 选择 | 理由 | 备选（何时换） |
|---|---|---|---|
| 语言 | Python 3.11+ | 生态最全，团队熟悉 | — |
| LLM | 智谱 glm-4.5-flash | 免费、快（glm-4-flash 已于 2026-09 被智谱下线，实测确认）；已支持工具调用，v2.0 Agent 化无需换模型 | DeepSeek/Qwen（只需改 `core.py`） |
| Embedding | 智谱 embedding-3，账号无权限时自动降级本地 `BAAI/bge-small-zh-v1.5`（fastembed） | 与 chat 同供应商，共用一个 API Key；本地兜底保证链路始终可跑（切换 provider 后需重建 `db/`） | — |
| 向量库 | Chroma（PersistentClient 本地持久化） | 嵌入式零运维；单机十万级向量绰绰有余 | Qdrant/pgvector（见 §6 触发条件） |
| 界面 | Streamlit | 聊天组件 + 侧边栏开箱即用；`st.write_stream` 原生支持 v1.1 流式 | FastAPI + 前端（见 §6 触发条件） |
| 文档解析 | pypdf + 内置读文本 | 覆盖 PDF/TXT/MD 三种格式，无重依赖 | pymupdf（PDF 解析质量不佳时） |
| 测试 | pytest | 标准选择；外部 API 通过 mock 层隔离 | — |
| 配置 | python-dotenv | API Key 走 `.env`，不硬编码 | — |

## 3. 模块职责与代码边界

### 三条硬边界（架构上最重要的约定）

1. **模型调用只出现在 `rag/core.py`**。换模型/换供应商/加缓存/加重试，只动这一个文件，`ingest.py`、`pipeline.py` 不感知供应商细节。
2. **`rag/` 包不 import 任何 UI 库**（streamlit 等）。`pipeline.answer()` 返回 `(回答, 引用片段列表)` 这样的纯数据，界面只负责渲染。换 UI 壳时 `rag/` 一行不动。
3. **UI 层不写业务逻辑**。切分、检索、提示词拼装全部在 `rag/` 内，`app.py` 只做上传保存、调用、展示。

### 现有模块的关键参数

| 模块 | 职责 | 关键参数/约定 |
|---|---|---|
| `rag/core.py` | embed / chat 唯一出口 | embedding 单批最多 64 条，内部分批；模型名可经 `CHAT_MODEL` 覆盖；provider 由 `EMBEDDING_PROVIDER` 决定（`auto` 无 Key 时走本地） |
| `rag/ingest.py` | 解析、切分、向量化、入库 | `CHUNK_SIZE=500` 字符、`CHUNK_OVERLAP=80`；**先按 markdown 标题切小节**（`#`~`####`），小节内按段落聚合，切不出段落时按行；片段 id 为 `{文件名}::{序号}::{内容hash[:8]}`；metadata 记录 source/chunk；状态 `added/updated/unchanged/empty/error`；CLI：`python -m rag.ingest [路径] [--rebuild]` |
| `rag/query.py` | 只检索、不生成 | 验证"库建得对不对"用；`-k` 指定返回片段数，`--full` 打印完整片段。刻意不接大模型，避免把切分问题和生成问题混在一起 |
| `rag/pipeline.py` | 检索 + 生成主链路 | `TOP_K=4`；相似度分数 = `1 - Chroma 距离`；提示词要求"资料不足时明确说明，不编造" |
| `rag/mock.py` | 模拟数据层 | 供 `demo.py` 演示与单元测试使用，与真实链路同接口形状（`(回答, 引用列表)`） |
| `app.py` | Streamlit 薄壳（可选） | 会话历史存 `st.session_state`；引用来源用 expander 展示来源文件名、分数、摘录；入库结果按 `added/updated/unchanged/error` 分类提示 |

## 4. 关键设计决策（ADR 摘要）

**D1 不引入 LangChain / LlamaIndex，管线手写。**
MVP 链路只有"切分 → 向量化 → 检索 → 拼装 → 生成"五步，手写代码量小、可控、好调试，还避免了重框架的版本依赖地狱。LangChain 的价值在复杂编排和现成集成，MVP 都用不上。

**D2 供应商封装收敛到 `core.py`。**
目前智谱免费额度足够 MVP；未来对比 DeepSeek/Qwen 或接入缓存/重试/流式，改动面收敛在一个文件。

**D3 向量库用嵌入式 Chroma，不部署独立服务。**
个人知识库场景数据量小（千级文档、十万级片段以内），独立向量数据库（Milvus/Qdrant）的运维成本完全不划算。Chroma 持久化到 `db/` 目录即可满足"全程本地运行"的非功能要求。

**D4 界面是可替换的薄壳。**
Streamlit 是 MVP 求快的选择，天花板在于多用户/复杂交互。因为 `rag/` 与 UI 已分离，未来迁移到 FastAPI + 前端时核心逻辑零改动，只重写展示层。

**D5 外部依赖可 mock。**
`rag/mock.py` 提供与真实链路同构的模拟实现，让 UI Demo 与单元测试不依赖网络和 API Key；后续对 `embed`/`chat` 的测试同样走 mock/注入。

**D6 按 markdown 标题切小节，而不是按字数累加（v0.2 新增）。**
原实现按 `\n\n` 分段后累加到 500 字，边界是"任意位置的 500 字"。实测同一片段里同时包含
「缓存穿透 / 缓存击穿 / 缓存雪崩」三个知识点，语义被稀释、相似度全线偏低；修正后同一批
语料 4 片 → 9 片，三组"串味"关系全部归零（数据见 README「切分质量：实测」）。
片段前还会带上所属小节标题，让下游知道这段话在讲哪个知识点。
代价：**只对 markdown 有效**，PDF/DOCX 没有标题结构，只能退回按行/段落切。

**D7 幂等更新靠内容指纹，不靠文件名（v0.2 新增）。**
原实现是「文件名已存在 → 整篇跳过且不报错」，于是文档改了内容重新入库**永远不生效**，
而调用方收到的是一个"成功跳过"的结果，从日志上完全看不出异常。现在片段 id 带内容指纹，
指纹一致才跳过，变了就覆盖。状态因此从 `duplicate` 拆成 `unchanged` / `updated` ——
旧名字把"用户传了两次"和"文件改了但被静默跳过"混为一谈，而后者正是这个 bug。

**D8 provider 的默认值偏向"能跑起来"，但显式指定时绝不静默降级（v0.2 新增）。**
`auto` 模式没配 Key 就直接走本地模型，而不是先发一个注定失败的请求再降级。
但显式设 `EMBEDDING_PROVIDER=zhipu` 时失败即报错：两个 provider 的向量维度不同
（2048 vs 512），静默切换会让检索结果错得看不出来。

## 5. 测试策略

```bash
python -m pytest tests -q      # 35 项，离线、不联网、约 7 秒
```

- **纯函数单测**：切分逻辑 `split_text`（标题切小节、段落聚合、超长硬切与重叠、
  无空行退回按行）、`build_context` 编号拼装、`collect_files` 目录展开——不触网，直接测。
- **回归测试**（v0.2 新增，钉住 D6/D7/D8）：
  `test_sections_are_not_mixed`（同一片段不许出现两个知识点）、
  `test_changed_file_is_really_updated`（改了内容必须真替换旧片段）、
  `test_no_blank_lines_falls_back_to_line_split`（无空行文本不许退化成整篇一片）、
  `test_explicit_zhipu_does_not_fall_back_without_key`（显式 provider 不许静默降级）。
- **provider 状态隔离**：`tests/test_core.py` 有 autouse fixture 重置 `_provider` 与
  Key —— 否则某个用例把 provider 定成 `local` 后，后面的用例会真的去加载本地模型。
- **mock 层单测**：`tests/test_mock.py` 已覆盖模拟入库/检索/回答的核心行为。
- **外部 API 隔离**：涉及 `embed`/`chat` 的测试一律 mock（monkeypatch 或注入），CI/本地不烧 API 额度。
- **入库落盘隔离**：入库用例用 `monkeypatch` 把 `DB_DIR` 指到 `tmp_path`，不污染真实 `db/`。
- 真实链路的端到端验证（真实 Key + 真实文档）按产品文档 §7 验收标准人工执行，不进自动化。
- **切分对照**：`scripts/compare_chunking.py` 内置修正前的算法（从 git 历史原样抄下），
  可在任意文档上做新旧 A/B —— 这是调切分参数时唯一可信的反馈。

## 6. 演进路径与换栈触发条件

技术栈按 Roadmap 分阶段演进，**每个阶段都不推翻 MVP 选型，只做增量**：

| 阶段 | 技术增量 | 说明 |
|---|---|---|
| v1.1 流式输出、重试 | `core.chat()` 增加 stream 模式（zhipuai SDK 流式迭代器）；UI 用 `st.write_stream`；`core.py` 内加重试（手写或 tenacity） | 不换任何组件 |
| v1.2 混合检索 + Rerank + 评测 | BM25 用 `rank-bm25` 内存索引，与向量结果做 **RRF 融合**——不上 Elasticsearch，这个文档规模用 ES 是杀鸡用牛刀；Rerank 评估 bge-reranker；RAGAS 出评测报告 | 不换任何组件 |
| v2.0 Agent 化 | glm-4-flash 已支持 function call，先用**普通 Python 工具循环**实现；LangGraph 只在编排出现分支/并行/人工介入等真实复杂度时再评估引入 | 框架后置，避免提前背复杂度 |

**换栈触发条件**（满足才换，否则不动）：

| 信号 | 动作 |
|---|---|
| 需要用户系统 / 文档管理后台 / 多人并发 | Streamlit → FastAPI + 前端（`rag/` 不动，只换壳） |
| 向量规模 > 50 万，或多机共享知识库 | Chroma → Qdrant / pgvector（只改 `ingest.py`/`pipeline.py` 的存取层） |
| PDF 解析质量不达标（扫描件、复杂表格） | pypdf → pymupdf，或补 OCR 链路 |
| LLM 供应商更换或需要多模型路由 | 只改 `rag/core.py` |
