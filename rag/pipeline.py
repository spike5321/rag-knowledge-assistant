"""检索 + 生成：问答主链路。"""
from rag.core import chat, embed_query
from rag.ingest import get_collection

TOP_K = 4

PROMPT_TEMPLATE = """你是一个严谨的知识库问答助手。请仅根据下面的参考资料回答用户问题。

要求：
1. 只使用参考资料中的信息，不要编造；资料不足以回答时明确说明。
2. 回答末尾用 [1] [2] 这样的编号标注引用了哪些资料片段。

参考资料：
{context}

用户问题：{question}
"""


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    col = get_collection()
    vector = embed_query(question)
    res = col.query(query_embeddings=[vector], n_results=k)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": doc, "source": meta["source"], "score": 1 - dist})
    return hits


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] （来源: {h['source']}）\n{h['text']}" for i, h in enumerate(hits)
    )


def answer(question: str) -> tuple[str, list[dict]]:
    """返回 (回答, 引用片段列表)，供界面展示溯源。"""
    hits = retrieve(question)
    if not hits:
        return "知识库为空或没有检索到相关内容，请先运行文档入库。", []
    prompt = PROMPT_TEMPLATE.format(context=build_context(hits), question=question)
    reply = chat([{"role": "user", "content": prompt}])
    return reply, hits
