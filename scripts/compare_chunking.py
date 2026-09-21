# -*- coding: utf-8 -*-
"""对照脚本：修正前 vs 修正后的切分逻辑。

这是一次性分析脚本（不属于产品代码），留着是因为 README 里的数字
应该能被任何人自己复现：

    python scripts/compare_chunking.py              # 跑 data/docs/
    python scripts/compare_chunking.py 你的文档.md   # 跑指定文件

`split_text_old()` 是从本仓库 git 历史里**原样抄下来**的，不是凭印象写的
近似版本 —— 它和修正前的 rag/ingest.py 逐行一致，唯一区别是参数写死了。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE, collect_files, read_file, split_text  # noqa: E402


def split_text_old(text: str) -> list[str]:
    """修正前的算法：只按空行切段，再按字数累加到 CHUNK_SIZE。"""
    text = text.replace("\r\n", "\n").strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= CHUNK_SIZE:
            current = f"{current}\n{para}".strip()
            continue
        if current:
            chunks.append(current)
        while len(para) > CHUNK_SIZE:
            chunks.append(para[:CHUNK_SIZE])
            para = para[CHUNK_SIZE - CHUNK_OVERLAP :]
        current = para
    if current:
        chunks.append(current)
    return chunks


def preview(chunk: str, n: int = 46) -> str:
    return chunk[:n].replace("\n", " | ")


def report(name: str, text: str) -> None:
    old = split_text_old(text)
    new = split_text(text)
    blank_paras = len([p for p in text.split("\n\n") if p.strip()])
    lines = len([ln for ln in text.split("\n") if ln.strip()])

    print(f"=== {name}")
    print(f"    原文 {len(text)} 字，空行段落 {blank_paras} 个 / 非空行 {lines} 行")
    print(f"    旧: {len(old)} 个片段  长度 {[len(c) for c in old][:12]}")
    print(f"    新: {len(new)} 个片段  长度 {[len(c) for c in new][:12]}")
    if blank_paras <= 1:
        print("    注意：这份文本几乎没有空行，旧算法切不出段落，会退化成")
        print("          按字符位置硬切（边界落在句子中间）；新算法按行切。")
    if old:
        print(f"    旧版首片段：{preview(old[0])}")
    if new:
        print(f"    新版首片段：{preview(new[0])}")
    print()


def synthetic_no_blank_lines() -> None:
    """构造用例：949 字、60 行、**没有空行**。

    用来观察"边界切在哪里"这件事 —— 真实的 PDF 抽取文本就是这个形状。
    注意预期不要搞错：旧算法在这份数据上**不是**切出 1 个片段，
    而是按字符位置硬切成 500/500/109；差别在边界落在哪，不在片段数量。
    """
    body = "\n".join(f"第{i}行内容大约二十个字左右吧" for i in range(60))
    report("「构造：无空行的长文本」（949 字 / 60 行，用于观察边界）", body)


def main() -> int:
    args = sys.argv[1:]
    if args:
        files = collect_files([Path(a) for a in args])
    else:
        files = collect_files([Path(__file__).resolve().parent.parent / "data" / "docs"])

    if not files:
        print("没有找到文档。用法: python scripts/compare_chunking.py [文件或目录]")
        return 1

    for f in files:
        try:
            report(f.name, read_file(f))
        except Exception as exc:  # noqa: BLE001
            print(f"=== {f.name} 读取失败：{exc}\n")

    synthetic_no_blank_lines()
    print(f"（切分参数: CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
