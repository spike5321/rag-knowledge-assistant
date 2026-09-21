"""知识库构建工具 · 界面 Demo（模拟数据版）。

后端功能全部由 rag/mock.py 模拟，不调用真实 API，
用于验证界面与操作流程。真实版本见 app.py。

运行: streamlit run demo.py
"""
import time

import streamlit as st

from rag.mock import MOCK_KNOWLEDGE_BASE, mock_answer, mock_ingest

st.set_page_config(page_title="知识库构建工具 · Demo", page_icon="📚")
st.title("📚 知识库构建工具")
st.caption("界面演示版：数据为模拟数据，用于验证交互流程")

# 演示用：把模拟知识库文档当作"可上传"的文件
DEMO_FILES = list(MOCK_KNOWLEDGE_BASE.keys())

# 侧边栏：文档管理与入库
with st.sidebar:
    st.header("📄 文档管理")

    uploaded = st.file_uploader(
        "上传文档（支持 PDF / TXT / MD，可多选）",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        disabled=True,  # Demo 版不支持真实上传，仅展示交互位置
    )
    st.caption("⬆️ 上传组件在 Demo 中禁用，请用下方示例文档体验入库流程")

    st.divider()
    st.subheader("示例文档")
    selected = {}
    for name in DEMO_FILES:
        selected[name] = st.checkbox(name, value=True)

    if st.button("入库", type="primary", use_container_width=True):
        names = [n for n, on in selected.items() if on]
        if not names:
            st.warning("请先勾选至少一份示例文档")
        else:
            bar = st.progress(0, text="准备入库...")
            steps = ["解析文档", "切分片段", "向量化", "写入向量库"]
            for i, step in enumerate(steps):
                time.sleep(0.4)
                bar.progress((i + 1) / len(steps), text=f"{step}...")
            results = mock_ingest(names)
            bar.progress(1.0, text="入库完成")
            st.session_state["ingested"] = results
            st.success("入库完成：" + "、".join(f"{r['source']}（{r['chunks']} 片段）" for r in results))

    if results := st.session_state.get("ingested"):
        st.divider()
        st.subheader("已入库文档")
        for r in results:
            st.markdown(f"- **{r['source']}** · {r['chunks']} 个片段")

# 主界面：对话
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 引用来源（{len(msg['sources'])} 条）"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** · 片段 {s['chunk'] + 1} · 相似度 {s['score']:.2f}")
                    st.text(s["text"])
                    st.divider()

if not st.session_state.get("ingested"):
    st.info("👈 先在侧边栏勾选示例文档并点击「入库」，再开始提问")
elif question := st.chat_input("基于你的知识库提问..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("检索资料并生成回答..."):
            reply, hits = mock_answer(question)
        st.markdown(reply)
        if hits:
            with st.expander(f"📎 引用来源（{len(hits)} 条）"):
                for s in hits:
                    st.markdown(f"**{s['source']}** · 片段 {s['chunk'] + 1} · 相似度 {s['score']:.2f}")
                    st.text(s["text"])
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": hits})
