"""rag/core.py 的单元测试：embed 分批与 provider 降级逻辑，不触网。"""
import pytest

import rag.core as core


@pytest.fixture(autouse=True)
def _isolate_provider(monkeypatch):
    """每个用例都把 provider 状态和 Key 重置掉。

    不加这层的话，某个用例把 _provider 定成 "local" 之后，
    后面的用例就会真的去加载本地向量模型 —— 要么慢，要么直接联网下载。
    """
    monkeypatch.setattr(core, "_provider", None)
    monkeypatch.setattr(core, "_announced_local", True)   # 测试输出里不用打提示
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")       # 默认走智谱分支
    monkeypatch.setenv("EMBEDDING_PROVIDER", "auto")
    yield


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


def test_auto_uses_local_when_no_api_key(monkeypatch):
    """auto 模式 + 没配 Key → 直接走本地，不发那个注定失败的请求。

    这是"clone 下来装完依赖就能跑"的前提：不该先被
    "你去注册个账号填 Key" 拦住。
    """
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "auto")
    monkeypatch.setattr(core, "_embed_local", lambda texts: [[0.0]] * len(texts))

    def _must_not_call():
        raise AssertionError("没配 Key 时不应该去调智谱")

    monkeypatch.setattr(core, "client", _must_not_call)

    assert core.embed(["测试"]) == [[0.0]]


def test_explicit_zhipu_does_not_fall_back_without_key(monkeypatch):
    """显式指定 zhipu 时，即使没配 Key 也必须报错，不能偷偷改用本地模型
    —— 否则算出来的向量维度和库里的对不上，检索会静默出错。"""
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "zhipu")

    with pytest.raises(RuntimeError, match="ZHIPU_API_KEY"):
        core.embed(["测试"])
