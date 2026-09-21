"""智谱 GLM API 封装：embedding 与 chat。

所有模型调用集中在这里，方便后续换模型或加缓存。

向量化支持双 provider（EMBEDDING_PROVIDER 环境变量）：
- "auto"（默认）：**没配 ZHIPU_API_KEY 就直接走本地**；配了但账号无权限
  （错误码 1211）时也降级本地
- "zhipu"：只用智谱，失败即报错
- "local"：只用本地 fastembed（BAAI/bge-small-zh-v1.5，离线运行）

auto 的默认值刻意偏向"能跑起来"：这是个可以独立使用的知识库工具，
不该先被"你去注册个账号填 Key"拦住。想强制用智谱就显式设 zhipu。

注意：两个 provider 的向量维度不同（2048 vs 512），切换后必须重建向量库 ——
`python -m rag.ingest --rebuild`。
"""
import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()

EMBEDDING_MODEL = "embedding-3"
CHAT_MODEL = os.getenv("CHAT_MODEL", "glm-4.5-flash")  # glm-4-flash 已被智谱下线
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

_client: ZhipuAI | None = None
_local_model = None
_provider: str | None = None      # 配置值：auto / zhipu / local；降级后会定下来
_announced_local = False          # 本地向量化的提示只打一次，别刷屏


def client() -> ZhipuAI:
    global _client
    if _client is None:
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未找到 ZHIPU_API_KEY，请在项目根目录创建 .env 文件并填入你的智谱 API Key"
            )
        _client = ZhipuAI(api_key=api_key)
    return _client


def _configured_provider() -> str:
    """环境变量里配的 provider（首次调用时读一次并记住）。"""
    global _provider
    if _provider is None:
        _provider = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
    return _provider


def _effective_provider() -> str:
    """实际会走的 provider。

    auto 的判定依据是"有没有配 Key"：没配就直接走本地，而不是先发一个必然
    失败的请求、等智谱回报错误再降级。少一次无效请求，也少一段必须联网
    才能触发的分支 —— 离线场景下才真的可用。
    """
    configured = _configured_provider()
    if configured in ("zhipu", "local"):
        return configured
    return "zhipu" if os.getenv("ZHIPU_API_KEY") else "local"


def _announce_local(reason: str) -> None:
    global _announced_local
    if not _announced_local:
        print(f"[rag] {reason}，使用本地向量化（{LOCAL_EMBEDDING_MODEL}）")
        _announced_local = True


def _embed_zhipu(texts: list[str]) -> list[list[float]]:
    result: list[list[float]] = []
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        resp = client().embeddings.create(model=EMBEDDING_MODEL, input=batch)
        result.extend(item.embedding for item in resp.data)
    return result


def _embed_local(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _get_local_model().embed(texts)]


def _get_local_model():
    global _local_model
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if _local_model is None:
        from fastembed import TextEmbedding

        _local_model = TextEmbedding(LOCAL_EMBEDDING_MODEL)
    return _local_model


def embed(texts: list[str]) -> list[list[float]]:
    """向量化文档片段，单批最多 64 条由内部分批保证。"""
    global _provider

    if _effective_provider() == "local":
        if _configured_provider() == "auto":
            _announce_local("未配置 ZHIPU_API_KEY")
        return _embed_local(texts)

    try:
        return _embed_zhipu(texts)
    except Exception as e:
        # 1211 = 模型不存在：账号未开通 embedding 模型，auto 模式下降级到本地
        if _configured_provider() != "auto" or "1211" not in str(e):
            raise
        _announce_local("智谱 embedding 模型不可用（当前账号无权限）")
        _provider = "local"       # 定下来，后续调用不再重试智谱
        return _embed_local(texts)


def embed_query(question: str) -> list[float]:
    """向量化用户查询。

    走本地 bge 时必须给 query 加检索指令前缀：这类模型是按"短文 ↔ 短问"
    非对称检索训练的，query 侧不加前缀会让相似度整体偏低。
    """
    if _effective_provider() == "local":
        instruction = "为这个句子生成表示以用于检索相关文章："
        vec = next(_get_local_model().query_embed([f"{instruction}{question}"]))
        return vec.tolist()
    return embed([question])[0]


def chat(messages: list[dict], temperature: float = 0.1) -> str:
    resp = client().chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content
