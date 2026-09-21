# -*- coding: utf-8 -*-
"""知识库构建层：文档 → 切分 → 向量化 → 写入 Chroma。

用法：

    python -m rag.ingest                       # 入库 data/docs/ 下的全部文档
    python -m rag.ingest 我的资料/               # 入库指定目录（递归）
    python -m rag.ingest a.md b.pdf            # 入库指定文件
    python -m rag.ingest --rebuild             # 清空向量库后重建

支持 .md / .txt / .pdf。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
这一层与 MockMate 的 `kb/store.py` 是**同一套逻辑的两个形态**：

  · 独立形态（本文件）—— 命令行工具，自己管向量库、自己算向量；
  · 内嵌形态（MockMate）—— 同一个算法，向量由 Agent 的 LLM 客户端统一算，
    这样 Agent 换模型不必动存储层。

算法只维护一份，改动会两边同步。原先两边不一致，本文件补齐了三处修正：

 ① 按 markdown 标题切小节。原实现只按 "\\n\\n" 切段再累加到 500 字：
      a) pypdf 抽出来的 PDF 常常几乎没有空行，切不出段落，于是退化成按
         字符位置硬切，边界落在句子中间。（实测 949 字无空行文本：旧 3 片，
         硬切在 500/500/109；新 2 片，切在行首。）
      b) 更关键的是边界本身：它是"任意位置的 500 字"。实测同一片段里混着
         「缓存穿透 / 缓存击穿 / 缓存雪崩」三个知识点，语义被稀释，相似度
         全线偏低。换成按标题切之后同样语料 4 片 → 9 片，串味消失。
    现在：先按标题切小节，小节内再按段落聚合；切不出段落时退回按行切。

 ② 覆盖式更新。原实现判定「这个文件名已存在就整篇跳过」，而且不报错 ——
    文档改了内容重新入库永远不生效，还完全静默。现在按内容指纹判断：
    内容没变才跳过，变了就覆盖。

 ③ 片段 id 带内容指纹：从 "文件名::序号" 变成 "文件名::序号::内容hash"。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import chromadb
from pypdf import PdfReader

from rag.core import embed

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "db"                      # 向量库落盘目录（.gitignore 已忽略）
DOCS_DIR = ROOT / "data" / "docs"         # 默认语料目录
COLLECTION_NAME = "knowledge_base"

# 余弦距离在语义检索里比欧氏距离更合适：只看方向不看长度，短句长句能公平比较。
# 另外 cosine 空间下 distance = 1 - 余弦相似度，score 天然落在 0~1，
# 可以直接当"相似度"展示（见 rag/pipeline.py 的 retrieve）。
COLLECTION_META = {"hnsw:space": "cosine"}

CHUNK_SIZE = 500      # 单个片段的目标长度（字）
CHUNK_OVERLAP = 80    # 超长段落硬切时的重叠长度，避免答案正好被切在接缝上
SUPPORTED_SUFFIX = {".md", ".txt", ".pdf"}

# markdown 标题行（# ~ ####）—— 用它当切分边界
_HEADING = re.compile(r"^(#{1,4})\s+(.+)$", re.M)


# ---------------------------------------------------------------------------
# 文档读取与切分
# ---------------------------------------------------------------------------


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    raise ValueError(f"不支持的文件类型: {suffix}")


def _pack_paragraphs(body: str, chunk_size: int, overlap: int) -> list[str]:
    """按空行分段，把相邻小段聚合成不超过 chunk_size 的块。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    # ★ 修正 ①a：切不出段落时退回按单行切。
    #   PDF 抽取的文本常常一行一段、几乎没有空行，于是"段落"只有一个。
    #   原逻辑此时只剩两条路：不足 chunk_size 就整篇算一片；超过就按字符
    #   位置硬切，把句子从中间劈开。退回按行切之后，边界落在行首 ——
    #   片段边界对应的是文本自身的结构，而不是一个字符位置。
    if len(paragraphs) <= 1 and "\n" in body:
        paragraphs = [ln.strip() for ln in body.split("\n") if ln.strip()]

    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
            continue
        if current:
            pieces.append(current)
        # 单段本身就超长（比如整段没有空行的表格）→ 硬切，带重叠
        while len(para) > chunk_size:
            pieces.append(para[:chunk_size])
            para = para[chunk_size - overlap :]
        current = para
    if current:
        pieces.append(current)
    return pieces


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """把一篇文档切成检索片段。

    ★ 修正 ①b：**先按 markdown 标题切小节**，每个小节再按段落聚合。

    为什么这一步很关键？原来的做法只管按字数累加到 500 字，切出来的边界是
    "任意位置的 500 字"。实测就是这样：一个片段里同时包含
    「缓存击穿」和「缓存雪崩」的正文 —— 两个知识点被从中间切开又重新拼到了
    一起，语义被稀释，相似度全线偏低。

    改成按标题切之后，一个片段 = 一个知识点。

    另外每条片段前面会带上所属小节标题（如【2. 缓存击穿】），
    这样下游（不管是拿去做问答还是喂给 Agent）都能知道
    "这段话在讲哪个知识点"，而不是看到一段没头没尾的正文。
    """
    text = text.replace("\r\n", "\n").strip()
    parts = _HEADING.split(text)

    # 切成 (标题, 正文) 列表。parts 结构是 [标题前内容, #, 标题1, 正文1, ##, 标题2, 正文2, ...]
    sections: list[tuple[str, str]] = []
    preamble = parts[0].strip()
    if preamble:
        sections.append(("", preamble))
    for i in range(1, len(parts) - 1, 3):
        title = parts[i + 1].strip()
        body = parts[i + 2].strip()
        if title or body:
            sections.append((title, body))

    chunks: list[str] = []
    for title, body in sections:
        if not body:
            continue
        for piece in _pack_paragraphs(body, chunk_size, overlap):
            chunks.append(f"【{title}】\n{piece}" if title else piece)
    return chunks


def collect_files(targets: list[Path]) -> list[Path]:
    """把"文件或目录"的混合列表展开成文件列表，目录递归展开。"""
    files: list[Path] = []
    for t in targets:
        if t.is_dir():
            files.extend(sorted(p for p in t.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIX))
        elif t.exists():
            files.append(t)
    return files


# ---------------------------------------------------------------------------
# 向量库
# ---------------------------------------------------------------------------


def get_collection():
    """拿向量库里的集合（不存在就建）。

    故意不加模块级缓存：入库是命令行一次性动作，而测试要用 monkeypatch
    换 DB_DIR 来隔离，缓存会让两次测试互相串数据。
    """
    client = chromadb.PersistentClient(str(DB_DIR))
    return client.get_or_create_collection(COLLECTION_NAME, metadata=COLLECTION_META)


def _chunk_id(source: str, index: int, text: str) -> str:
    """★ 修正 ③：id 里带上内容指纹，内容变了 id 就变。"""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{source}::{index}::{digest}"


def reset_collection() -> int:
    """清空整个向量库，返回删掉的片段数。"""
    col = get_collection()
    existing = col.get(include=[]).get("ids") or []
    if existing:
        col.delete(ids=existing)
    return len(existing)


def ingest(paths: list[Path], reset: bool = False) -> list[dict]:
    """入库并返回每个文件的处理结果，供界面/命令行展示。

    每项为 {"source", "chunks", "status"}，status 取值：

    - "added"      新入库
    - "updated"    同名文档内容变了 → 覆盖式更新（★ 修正 ②）
    - "unchanged"  同名文档内容一字未变 → 跳过，不重复花向量化额度
    - "empty"      文档无内容
    - "error"      读文件或向量化失败（记下原因，不中断其他文件）
    """
    if reset:
        reset_collection()

    col = get_collection()
    results: list[dict] = []

    for path in paths:
        try:
            text = read_file(path)
        except Exception as exc:  # noqa: BLE001
            results.append({"source": path.name, "chunks": 0, "status": "error", "error": str(exc)})
            continue

        chunks = split_text(text)
        if not chunks:
            results.append({"source": path.name, "chunks": 0, "status": "empty"})
            continue

        ids = [_chunk_id(path.name, i, c) for i, c in enumerate(chunks)]
        old_ids = set(col.get(where={"source": path.name}, include=[]).get("ids") or [])

        # 内容指纹完全一致 → 真的没变，跳过（省向量化调用）
        if old_ids and old_ids == set(ids):
            results.append({"source": path.name, "chunks": len(chunks), "status": "unchanged"})
            continue

        # 先向量化再删旧数据：万一网络挂了，旧索引还在，不至于查不到东西
        try:
            vectors = embed(chunks)
        except Exception as exc:  # noqa: BLE001
            results.append(
                {"source": path.name, "chunks": len(chunks), "status": "error", "error": str(exc)}
            )
            continue

        # ★ 修正 ②：覆盖式更新。文档改了内容，重新入库会真正生效。
        col.delete(where={"source": path.name})
        col.add(
            ids=ids,
            documents=chunks,
            embeddings=vectors,
            metadatas=[{"source": path.name, "chunk": i} for i in range(len(chunks))],
        )
        results.append(
            {
                "source": path.name,
                "chunks": len(chunks),
                "status": "updated" if old_ids else "added",
            }
        )

    return results


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------

_LABEL = {
    "added": "新增",
    "updated": "更新",
    "unchanged": "跳过（内容未变）",
    "empty": "跳过（无内容）",
    "error": "失败",
}


def _main(argv: list[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__.split("━━━")[0].strip())
        return 0

    rebuild = "--rebuild" in argv
    raw = [a for a in argv if not a.startswith("-")]
    targets = [Path(p) for p in raw] or [DOCS_DIR]

    missing = [str(t) for t in targets if not t.exists()]
    for m in missing:
        print(f"路径不存在，已忽略: {m}")

    files = collect_files(targets)
    if not files:
        print("没有找到可入库的文档，请把 .md/.txt/.pdf 放进 data/docs/")
        return 1

    print(f"扫描到 {len(files)} 个文件" + ("，重建模式将清空现有向量库" if rebuild else ""))
    results = ingest(files, reset=rebuild)

    for r in results:
        label = _LABEL.get(r["status"], r["status"])
        note = f" —— {r['error']}" if r.get("error") else ""
        count = f" {r['chunks']} 个片段" if r["chunks"] else ""
        print(f"  [{label}] {r['source']}{count}{note}")

    added = sum(r["chunks"] for r in results if r["status"] == "added")
    updated = sum(r["chunks"] for r in results if r["status"] == "updated")
    failed = [r["source"] for r in results if r["status"] == "error"]
    total = len(get_collection().get(include=[]).get("ids") or [])

    print()
    print(f"新增 {added} 个片段，更新 {updated} 个片段。向量库现有 {total} 个片段。")
    if failed:
        print(f"有 {len(failed)} 个文件失败: {'、'.join(failed)}")
        return 1
    print("下一步可以先验证检索效果: python -m rag.query \"你的问题\"")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
