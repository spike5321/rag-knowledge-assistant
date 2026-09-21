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

    def test_ingest_then_unchanged_skipped(self, doc_dir):
        """同名文件、内容一字未变 → 跳过，不重复花向量化额度。

        状态名从原来的 "duplicate" 改成 "unchanged"：旧名字把
        "用户上传了两次" 和 "文件改了但被静默跳过" 混为一谈，
        而后者正是下面那条回归测试要钉住的 bug。
        """
        f = doc_dir / "note.md"
        f.write_text("第一段。\n\n第二段。", encoding="utf-8")

        first = ingest_mod.ingest([f])
        assert first[0]["status"] == "added"
        assert first[0]["chunks"] > 0

        second = ingest_mod.ingest([f])
        assert second[0]["status"] == "unchanged"

    def test_changed_file_is_really_updated(self, doc_dir):
        """★ 回归测试（修正 ②）：文件改了内容，重新入库必须真正生效。

        旧实现在这里会返回 "duplicate" 并静默跳过 —— 向量库里永远是旧内容，
        而调用方还收到一个"成功跳过"的结果，从日志上完全看不出异常。
        """
        f = doc_dir / "note.md"
        f.write_text("旧内容：试用期是三个月。", encoding="utf-8")
        ingest_mod.ingest([f])
        col = ingest_mod.get_collection()
        assert any("三个月" in d for d in col.get(include=["documents"])["documents"])

        f.write_text("新内容：试用期改为六个月。", encoding="utf-8")
        again = ingest_mod.ingest([f])
        assert again[0]["status"] == "updated"

        docs = col.get(include=["documents"])["documents"]
        assert any("六个月" in d for d in docs)
        assert not any("三个月" in d for d in docs), "旧片段没有被替换掉"

    def test_one_bad_file_does_not_break_the_batch(self, doc_dir):
        """一个文件读不了，不应该让整批入库失败。"""
        bad = doc_dir / "bad.docx"
        bad.write_text("x", encoding="utf-8")
        good = doc_dir / "good.md"
        good.write_text("正常内容。", encoding="utf-8")

        results = ingest_mod.ingest([bad, good])
        assert results[0]["status"] == "error"
        assert results[0]["chunks"] == 0
        assert results[1]["status"] == "added"

    def test_rebuild_clears_existing_library(self, doc_dir):
        f = doc_dir / "note.md"
        f.write_text("内容。", encoding="utf-8")
        ingest_mod.ingest([f])

        assert ingest_mod.reset_collection() > 0
        assert ingest_mod.get_collection().get(include=[])["ids"] == []

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


class TestSplitTextHeadings:
    """★ 回归测试（修正 ①）：按 markdown 标题切小节。

    旧实现只按字数累加到 CHUNK_SIZE，切出来的边界是"任意位置的 500 字"，
    同一个片段里会混进好几个知识点，语义被稀释、相似度全线偏低。
    """

    MD = (
        "# 缓存三大问题\n\n"
        "## 1. 缓存穿透\n\n"
        "穿透是查一个数据库里也不存在的 key，请求全部落到数据库上。\n\n"
        "## 2. 缓存击穿\n\n"
        "击穿是某个热点 key 过期，大量请求同时打到数据库上。\n\n"
        "## 3. 缓存雪崩\n\n"
        "雪崩是一大批 key 同时过期，或者缓存整体挂掉。\n"
    )

    def test_each_section_becomes_its_own_chunk(self):
        chunks = split_text(self.MD)
        assert len(chunks) == 3
        assert all(c.startswith("【") for c in chunks), "片段应带上所属小节标题"

    def test_sections_are_not_mixed(self):
        """同一片段里不允许同时出现两个知识点。"""
        chunks = split_text(self.MD)
        for pair in [("穿透", "击穿"), ("穿透", "雪崩"), ("击穿", "雪崩")]:
            mixed = [c for c in chunks if pair[0] in c and pair[1] in c]
            assert not mixed, f"「{pair[0]}」和「{pair[1]}」被切进了同一个片段"

    def test_no_blank_lines_falls_back_to_line_split(self):
        """★ 回归测试（修正 ①a）：PDF 抽出的文本常常一行一段、几乎没有空行。

        旧实现按 "\\n\\n" 切段 → 整篇 = 1 个片段，等于没切。
        """
        body = "\n".join(f"第{i}行内容大约二十个字左右吧" for i in range(60))
        chunks = split_text(f"# 标题\n\n{body}")
        assert len(chunks) > 1, "又退化成整篇一块了"
        assert all(len(c) <= CHUNK_SIZE for c in chunks)


class TestCollectFiles:
    def test_expands_directory_recursively_and_filters_suffix(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.md").write_text("x", encoding="utf-8")
        (tmp_path / "sub" / "b.txt").write_text("y", encoding="utf-8")
        (tmp_path / "sub" / "skip.docx").write_text("z", encoding="utf-8")

        files = ingest_mod.collect_files([tmp_path])
        assert sorted(f.name for f in files) == ["a.md", "b.txt"]

    def test_missing_path_is_ignored(self, tmp_path):
        assert ingest_mod.collect_files([tmp_path / "nope.md"]) == []
