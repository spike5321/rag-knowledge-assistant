"""智谱 GLM API 封装：embedding 与 chat。

所有模型调用集中在这里，方便后续换模型或加缓存。

向量化支持双 provider（EMBEDDING_PROVIDER 环境变量）：
- "auto"（默认）：优先智谱 embedding-3，账号无权限（错误码 1211）时自动降级本地 bge 模型
- "zhipu"：只用智谱，失败即报错
- "local"：只用本地 fastembed（BAAI/bge-small-zh-v1.5，离线运行）

注意：两个 provider 的向量维度不同（2048 vs 512），切换后必须删除 db/ 目录重新入库。
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
_provider: str | None = None


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


def _resolve_provider() -> str:
    global _provider
    if _provider is None:
        _provider = os.getenv("EMBEDDING_PROVIDER", "auto")
    return _provider


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
    provider = _resolve_provider()
    if provider in ("auto", "zhipu"):
        try:
            vectors = _embed_zhipu(texts)
            _provider = "zhipu"
            return vectors
        except Exception as e:
            # 1211 = 模型不存在：账号未开通 embedding 模型，auto 模式下降级到本地
            if provider != "auto" or "1211" not in str(e):
                raise
            print(
                "[rag] 智谱 embedding 模型不可用（当前账号无权限），"
                f"自动切换为本地向量化（{LOCAL_EMBEDDING_MODEL}）"
            )
    _provider = "local"
    return _embed_local(texts)


def embed_query(question: str) -> list[float]:
    """向量化用户查询。bge 系列本地模型对 query 需要加检索指令前缀。"""
    if _resolve_provider() == "local":
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
