# 知识库问答助手（RAG MVP）

基于智谱 GLM + Chroma 的个人知识库问答应用，支持 PDF/TXT/MD 文档上传、检索增强问答与引用溯源。

## 快速开始

1. 创建环境并安装依赖：

```bash
conda activate rag-agent
pip install -r requirements.txt
```

2. 在项目根目录创建 `.env`（智谱 API Key 申请：https://open.bigmodel.cn ）：

```
ZHIPU_API_KEY=你的key
# 可选，默认 glm-4-flash（免费）
CHAT_MODEL=glm-4-flash
```

3. 启动界面：

```bash
streamlit run app.py
```

在侧边栏上传文档 → 点击"入库" → 在对话框提问。

## 项目结构

```
app.py            Streamlit 界面
rag/core.py       智谱 API 封装（embedding + chat）
rag/ingest.py     文档解析、切分、向量化入库
rag/pipeline.py   检索 + 生成主链路
data/docs/        知识库文档
db/               Chroma 向量库持久化
```

## Roadmap

- [ ] 混合检索（向量 + BM25）+ Rerank
- [ ] 查询改写 / HyDE
- [ ] 工具调用（联网搜索），升级为 Agent
- [ ] 迁移到 LangGraph 图编排
- [ ] RAGAS 检索质量评测
