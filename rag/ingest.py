"""文档解析、切分、向量化并写入 Chroma。

用法: python -m rag.ingest <文档路径或目录>
支持 .txt / .md / .pdf
"""
import sys
from pathlib import Path

import chromadb
from pypdf import PdfReader

from rag.core import embed

DB_DIR = Path(__file__).resolve().parent.parent / "db"
CHUNK_SIZE = 500      # 每个片段约 500 字符
CHUNK_OVERLAP = 80    # 相邻片段重叠，避免句子被切断后检索不到


def read_file(path: Path) -> str:
    if path.suffix.lower() in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"不支持的文件类型: {path.suffix}")


def split_text(text: str) -> list[str]:
    """按段落聚合到 CHUNK_SIZE，重叠 CHUNK_OVERLAP。"""
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
        # 单段超长时硬切
        while len(para) > CHUNK_SIZE:
            chunks.append(para[:CHUNK_SIZE])
            para = para[CHUNK_SIZE - CHUNK_OVERLAP :]
        current = para
    if current:
        chunks.append(current)
    return chunks


def ingest(paths: list[Path]) -> int:
    col = get_collection()
    total = 0
    for path in paths:
        text = read_file(path)
        chunks = split_text(text)
        if not chunks:
            print(f"跳过（无内容）: {path}")
            continue
        vectors = embed(chunks)
        col.add(
            ids=[f"{path.name}::{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=vectors,
            metadatas=[{"source": path.name, "chunk": i} for i in range(len(chunks))],
        )
        total += len(chunks)
        print(f"已入库 {path.name}: {len(chunks)} 个片段")
    return total


def get_collection():
    client = chromadb.PersistentClient(str(DB_DIR))
    return client.get_or_create_collection("knowledge_base")


if __name__ == "__main__":
    targets = [Path(p) for p in sys.argv[1:]] or [Path("data/docs")]
    files = []
    for t in targets:
        if t.is_dir():
            files.extend(p for p in t.rglob("*") if p.suffix.lower() in (".txt", ".md", ".pdf"))
        else:
            files.append(t)
    if not files:
        print("没有找到可入库的文档，请把 .txt/.md/.pdf 放进 data/docs/")
        sys.exit(1)
    n = ingest(files)
    print(f"完成，共入库 {n} 个片段")
