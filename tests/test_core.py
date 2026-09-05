"""rag/core.py 的单元测试：embed 分批与 provider 降级逻辑，不触网。"""
import pytest

import rag.core as core


class _FakeItem:
    def __init__(self, dim: int):
        self.embedding = [0.5] * dim


class _FakeEmbeddings:
    """模拟 zhipuai SDK 的 embeddings.create 接口形状。"""

    def __init__(self):
        self.calls: list[list[str]] = []

    def create(self, model: str, input: list[str]):
        assert model == core.EMBEDDING_MODEL
        assert len(input) <= 64  # SDK 单批上限
        self.calls.append(list(input))
        return type("Resp", (), {"data": [_FakeItem(8) for _ in input]})


def test_embed_batches_over_64(monkeypatch):
    fake = _FakeEmbeddings()
    monkeypatch.setattr(core, "client", lambda: type("C", (), {"embeddings": fake}))
    texts = [f"t{i}" for i in range(130)]

    vectors = core.embed(texts)

    assert len(vectors) == 130
    assert [len(c) for c in fake.calls] == [64, 64, 2]


def test_embed_single_small_batch(monkeypatch):
    fake = _FakeEmbeddings()
    monkeypatch.setattr(core, "client", lambda: type("C", (), {"embeddings": fake}))

    vectors = core.embed(["你好", "世界"])

    assert len(vectors) == 2
    assert len(fake.calls) == 1


def test_embed_auto_falls_back_on_1211(monkeypatch):
    """智谱返回 1211（账号无 embedding 权限）时，auto 模式降级本地向量化。"""

    def _raise_1211(model, input):
        raise RuntimeError("Error code: 400 ... 1211 模型不存在")

    monkeypatch.setattr(
        core, "client", lambda: type("C", (), {"embeddings": type("E", (), {"create": staticmethod(_raise_1211)})()})
    )
    monkeypatch.setattr(core, "_provider", None)  # 重置 provider 状态
    monkeypatch.setenv("EMBEDDING_PROVIDER", "auto")
    monkeypatch.setattr(core, "_embed_local", lambda texts: [[0.0]] * len(texts))

    vectors = core.embed(["测试"])

    assert vectors == [[0.0]]
    assert core._provider == "local"


def test_embed_zhipu_mode_raises_on_1211(monkeypatch):
    """显式指定 zhipu 时，1211 不降级而是抛出。"""

    def _raise_1211(model, input):
        raise RuntimeError("Error code: 400 ... 1211 模型不存在")

    monkeypatch.setattr(
        core, "client", lambda: type("C", (), {"embeddings": type("E", (), {"create": staticmethod(_raise_1211)})()})
    )
    monkeypatch.setattr(core, "_provider", None)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "zhipu")

    with pytest.raises(RuntimeError):
        core.embed(["测试"])


def test_embed_other_errors_not_swallowed(monkeypatch):
    """非 1211 错误（如网络、鉴权）不触发降级。"""

    def _raise_auth(model, input):
        raise RuntimeError("Error code: 1002 鉴权失败")

    monkeypatch.setattr(
        core, "client", lambda: type("C", (), {"embeddings": type("E", (), {"create": staticmethod(_raise_auth)})()})
    )
    monkeypatch.setattr(core, "_provider", None)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "auto")

    with pytest.raises(RuntimeError):
        core.embed(["测试"])
