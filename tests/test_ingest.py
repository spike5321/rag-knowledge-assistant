"""rag/ingest.py 切分与入库逻辑的单元测试（不触网，embed 走 mock）。"""
from pathlib import Path

import pytest

import rag.ingest as ingest_mod
from rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE, read_file, split_text
from rag.pipeline import build_context


class TestSplitText:
    def test_empty_text_returns_no_chunks(self):
        assert split_text("") == []
        assert split_text("   \n\n  ") == []

    def test_short_paragraphs_aggregated_into_one_chunk(self):
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = split_text(text)
        assert len(chunks) == 1
        assert "第一段" in chunks[0] and "第三段" in chunks[0]

    def test_paragraphs_split_when_exceeding_chunk_size(self):
        para = "字" * 300
        chunks = split_text(f"{para}\n\n{para}\n\n{para}")
        # 任意两段相加都超过 500，只能每段一片
        assert len(chunks) == 3
        assert all(len(c) == 300 for c in chunks)

    def test_two_small_paragraphs_fit_one_chunk_but_three_do_not(self):
        para = "字" * 200
        assert len(split_text(f"{para}\n\n{para}")) == 1
        assert len(split_text(f"{para}\n\n{para}\n\n{para}")) == 2

    def test_oversized_paragraph_hard_split(self):
        text = "".join(str(i % 10) for i in range(CHUNK_SIZE * 2 + 200))
        chunks = split_text(text)
        assert len(chunks) >= 3
        assert all(len(c) <= CHUNK_SIZE for c in chunks)
        # 相邻片段有 CHUNK_OVERLAP 重叠
        assert chunks[1].startswith(chunks[0][-CHUNK_OVERLAP:])
        assert chunks[2].startswith(chunks[1][-CHUNK_OVERLAP:])


class TestReadFile:
    def test_read_txt(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello 知识库", encoding="utf-8")
        assert read_file(f) == "hello 知识库"

    def test_read_md(self, tmp_path):
        f = tmp_path / "b.md"
        f.write_text("# 标题\n\n正文", encoding="utf-8")
        assert read_file(f).startswith("# 标题")

    def test_unsupported_suffix_raises(self, tmp_path):
        with pytest.raises(ValueError):
            read_file(tmp_path / "c.docx")


class TestBuildContext:
    def test_numbering_and_sources(self):
        hits = [
            {"text": "片段甲", "source": "手册.pdf"},
            {"text": "片段乙", "source": "调研.md"},
        ]
        ctx = build_context(hits)
        assert "[1]" in ctx and "[2]" in ctx
        assert "手册.pdf" in ctx and "调研.md" in ctx
        assert "片段甲" in ctx and "片段乙" in ctx


class TestIngestDedup:
    @pytest.fixture
    def doc_dir(self, tmp_path, monkeypatch):
        """临时文档目录 + 临时 Chroma 目录，避免污染真实 db/。"""
        monkeypatch.setattr(ingest_mod, "DB_DIR", tmp_path / "db")
        docs = tmp_path / "docs"
        docs.mkdir()
        return docs

    @pytest.fixture(autouse=True)
    def mock_embed(self, monkeypatch):
        monkeypatch.setattr(
            ingest_mod, "embed", lambda texts: [[0.1] * 8 for _ in texts]
        )

    def test_ingest_then_duplicate_skipped(self, doc_dir):
        f = doc_dir / "note.md"
        f.write_text("第一段。\n\n第二段。", encoding="utf-8")

        first = ingest_mod.ingest([f])
        assert first[0]["status"] == "added"
        assert first[0]["chunks"] > 0

        second = ingest_mod.ingest([f])
        assert second[0]["status"] == "duplicate"

    def test_different_files_both_ingested(self, doc_dir):
        f1 = doc_dir / "a.md"
        f2 = doc_dir / "b.md"
        f1.write_text("文档甲的内容。", encoding="utf-8")
        f2.write_text("文档乙的内容。", encoding="utf-8")
        results = ingest_mod.ingest([f1, f2])
        assert all(r["status"] == "added" for r in results)

    def test_empty_file_reported(self, doc_dir):
        f = doc_dir / "empty.md"
        f.write_text("  ", encoding="utf-8")
        results = ingest_mod.ingest([f])
        assert results[0]["status"] == "empty"
