# -*- coding: utf-8 -*-
"""只做检索、不调大模型 —— 用来验证"库建得对不对"。

用法：

    python -m rag.query "缓存击穿怎么解决"
    python -m rag.query "报销标准" -k 3

它和 Streamlit 界面（app.py）的区别是：这里**不生成回答**，只把命中的片段
和相似度打出来。好处有两个：

1. **不需要大模型**也能判断切分和向量化有没有问题 —— 检索质量是入库质量
   的直接体现，而生成质量会把两者混在一起看不清；
2. 换切分策略 / 换 embedding 模型之后，可以直接对比同一问题的命中片段，
   这是调参时唯一可信的反馈。

向量化走 rag.core，provider 由 EMBEDDING_PROVIDER 决定；没配 ZHIPU_API_KEY
时会用本地模型（首次运行需要下载约 100MB）。
"""
from __future__ import annotations

import argparse
import sys

from rag.pipeline import retrieve


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag.query",
        description="在已建好的知识库里检索片段（不生成回答）",
    )
    parser.add_argument("question", help="要检索的问题")
    parser.add_argument("-k", type=int, default=4, help="返回片段数，默认 4")
    parser.add_argument("--full", action="store_true", help="打印完整片段而不是截断预览")
    args = parser.parse_args()

    hits = retrieve(args.question, k=args.k)
    if not hits:
        print("没有检索到内容。先建库：python -m rag.ingest")
        return 1

    print(f"问题: {args.question}")
    print(f"命中 {len(hits)} 个片段\n")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] 相似度 {h['score']:.3f}   来源 {h['source']}")
        body = h["text"] if args.full else h["text"][:200].replace("\n", " ")
        print(f"    {body}{'' if args.full or len(h['text']) <= 200 else '...'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
