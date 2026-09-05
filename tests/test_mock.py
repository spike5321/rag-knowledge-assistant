"""rag/mock.py 模拟数据层的单元测试。"""
import unittest

from rag.mock import DEFAULT_ANSWER, MOCK_KNOWLEDGE_BASE, mock_answer, mock_ingest, mock_retrieve


class TestMockIngest(unittest.TestCase):
    def test_ingest_returns_result_per_file(self):
        results = mock_ingest(list(MOCK_KNOWLEDGE_BASE.keys()))
        self.assertEqual(len(results), 2)
        self.assertEqual({r["source"] for r in results}, set(MOCK_KNOWLEDGE_BASE.keys()))
        for r in results:
            self.assertGreater(r["chunks"], 0)

    def test_ingest_unknown_file_uses_fallback_chunk_count(self):
        results = mock_ingest(["不存在的文档.pdf"])
        self.assertEqual(results[0]["chunks"], 12)


class TestMockRetrieve(unittest.TestCase):
    def test_retrieve_returns_hits_for_known_topic(self):
        hits = mock_retrieve("年假有几天？")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["source"], "员工手册.pdf")
        self.assertTrue(all(0 < h["score"] <= 1 for h in hits))

    def test_retrieve_sorted_by_score_desc(self):
        hits = mock_retrieve("混合检索和 rerank 怎么配合？rerank 评测")
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_retrieve_empty_for_unknown_topic(self):
        self.assertEqual(mock_retrieve("今天天气怎么样"), [])


class TestMockAnswer(unittest.TestCase):
    def test_answer_for_known_topic_contains_citation(self):
        reply, hits = mock_answer("年假有几天？")
        self.assertIn("[1]", reply)
        self.assertTrue(hits)

    def test_answer_for_unknown_topic_uses_fallback(self):
        reply, hits = mock_answer("量子力学是什么")
        self.assertEqual(reply, DEFAULT_ANSWER)
        self.assertEqual(hits, [])

    def test_answer_is_keyword_case_insensitive(self):
        reply, _ = mock_answer("What is RAGAS evaluation?")
        self.assertIn("RAGAS", reply)


if __name__ == "__main__":
    unittest.main()
