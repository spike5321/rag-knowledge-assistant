"""知识库问答助手 - Streamlit 界面。

运行: streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from rag.ingest import ingest
from rag.pipeline import answer

st.set_page_config(page_title="知识库问答助手", page_icon="📚")
st.title("📚 知识库问答助手")

# 侧边栏：上传文档并入库
with st.sidebar:
    st.header("文档管理")
    uploads = st.file_uploader(
        "上传文档（支持 PDF / TXT / MD，可多选）",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploads and st.button("入库", type="primary"):
        save_dir = Path("data/docs")
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in uploads:
            p = save_dir / f.name
            p.write_bytes(f.getvalue())
            paths.append(p)
        with st.spinner("正在解析、切分并向量化..."):
            results = ingest(paths)
        added = [r for r in results if r["status"] == "added"]
        duplicates = [r["source"] for r in results if r["status"] == "duplicate"]
        if added:
            st.success(f"入库完成，新增 {sum(r['chunks'] for r in added)} 个片段")
        if duplicates:
            st.warning(f"以下文档已存在，已跳过重复入库：{'、'.join(duplicates)}")

# 主界面：对话
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 引用来源"):
                for s in msg["sources"]:
                    st.caption(f"**{s['source']}**（相似度 {s['score']:.2f}）")
                    st.text(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

if question := st.chat_input("基于你的知识库提问..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("检索资料并生成回答..."):
            try:
                reply, hits = answer(question)
            except Exception as e:  # 给初学者友好的错误提示
                reply, hits = f"出错啦：{e}", []
        st.markdown(reply)
        if hits:
            with st.expander("📎 引用来源"):
                for s in hits:
                    st.caption(f"**{s['source']}**（相似度 {s['score']:.2f}）")
                    st.text(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": hits})
