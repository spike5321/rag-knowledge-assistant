# 知识库问答助手 · 技术方案

版本：v0.1
日期：2026-09-05
关联文档：[product-design.md](product-design.md)（产品设计与验收标准）

## 1. 总体架构

整体是一个**单机单进程的分层架构**：UI 是薄壳，全部业务逻辑收敛在 `rag/` 包内，模型调用收敛在 `rag/core.py` 一处。

```
┌─────────────────────────────────────────────┐
│  app.py（Streamlit 界面）   demo.py（Mock 演示）│   ← 界面层：只做展示与交互
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  rag/ 包（核心逻辑，不依赖任何 UI 库）           │
│  ├─ core.py      模型调用唯一入口（embed/chat）│
│  ├─ ingest.py    解析 → 切分 → 向量化 → 入库   │
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
  智谱 embedding-3           智谱 glm-4-flash
  （向量化）                  （生成回答）
```

问答主链路（对应产品文档 F2/F3）：

```
用户提问 → embed(question) → Chroma 检索 top-4
        → 拼装提示词（参考资料 + [1][2] 编号）
        → glm-4-flash 生成 → (回答, 引用片段列表) → 界面展示溯源
```

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
| `rag/core.py` | embed / chat 唯一出口 | embedding 单批最多 64 条，内部分批；模型名可经 `CHAT_MODEL` 环境变量覆盖 |
| `rag/ingest.py` | 解析、切分、向量化、入库 | `CHUNK_SIZE=500` 字符、`CHUNK_OVERLAP=80`；片段 id 为 `{文件名}::{序号}`；metadata 记录 source/chunk；支持 CLI 批量导入（`python -m rag.ingest`） |
| `rag/pipeline.py` | 检索 + 生成主链路 | `TOP_K=4`；相似度分数 = `1 - Chroma 距离`；提示词要求"资料不足时明确说明，不编造" |
| `rag/mock.py` | 模拟数据层 | 供 `demo.py` 演示与单元测试使用，与真实链路同接口形状（`(回答, 引用列表)`） |
| `app.py` | Streamlit 薄壳 | 会话历史存 `st.session_state`；引用来源用 expander 展示来源文件名、分数、摘录 |

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

## 5. 测试策略

- **纯函数单测**：切分逻辑 `split_text`（段落聚合、超长硬切、重叠）、`build_context` 编号拼装——不触网，直接测。
- **mock 层单测**：`tests/test_mock.py` 已覆盖模拟入库/检索/回答的核心行为。
- **外部 API 隔离**：涉及 `embed`/`chat` 的测试一律 mock（monkeypatch 或注入），CI/本地不烧 API 额度。
- 真实链路的端到端验证（真实 Key + 真实文档）按产品文档 §7 验收标准人工执行，不进自动化。

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
