# 知识库问答助手（RAG MVP）

基于智谱 GLM + Chroma 的个人知识库问答应用，支持 PDF/TXT/MD 文档上传、检索增强问答与引用溯源。

## 快速开始

1. 创建环境并安装依赖：

```bash
conda create -n rag-agent python=3.13 -y
conda activate rag-agent
# 国内网络建议走清华镜像安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. 在项目根目录创建 `.env`（可直接复制 `.env.example` 修改，智谱 API Key 申请：https://open.bigmodel.cn ）：

```
ZHIPU_API_KEY=你的key
# 可选，默认 glm-4.5-flash（免费；glm-4-flash 已被智谱下线）
CHAT_MODEL=glm-4.5-flash
# 可选，向量化 provider：auto（默认，智谱 embedding-3 优先，账号无权限时自动降级本地 bge 模型）
# EMBEDDING_PROVIDER=auto
```

3. 启动界面：

```bash
streamlit run app.py
```

在侧边栏上传文档 → 点击"入库" → 在对话框提问。重复上传同名文档会提示已存在并跳过；也可以用命令行批量导入本地文件夹：

```bash
python -m rag.ingest path/to/your/docs
```

## 项目结构

```
app.py            Streamlit 界面
demo.py           界面 Demo（模拟数据，不调用 API）
rag/core.py       模型调用唯一入口（智谱 embedding/chat + 本地向量化兜底）
rag/ingest.py     文档解析、切分、向量化入库
rag/pipeline.py   检索 + 生成主链路
rag/mock.py       模拟数据层（Demo/测试用）
scripts/          示例文档生成与端到端验收脚本
data/docs/        知识库文档
db/               Chroma 向量库持久化
```

## Roadmap

- [ ] 混合检索（向量 + BM25）+ Rerank
- [ ] 查询改写 / HyDE
- [ ] 工具调用（联网搜索），升级为 Agent
- [ ] 迁移到 LangGraph 图编排
- [ ] RAGAS 检索质量评测
