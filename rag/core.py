"""智谱 GLM API 封装：embedding 与 chat。

所有模型调用集中在这里，方便后续换模型或加缓存。
"""
import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()

EMBEDDING_MODEL = "embedding-3"
CHAT_MODEL = os.getenv("CHAT_MODEL", "glm-4-flash")

_client: ZhipuAI | None = None


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


def embed(texts: list[str]) -> list[list[float]]:
    """批量向量化，embedding-3 单次最多 64 条，这里分批调用。"""
    result: list[list[float]] = []
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        resp = client().embeddings(model=EMBEDDING_MODEL, input=batch)
        result.extend(item.embedding for item in resp.data)
    return result


def chat(messages: list[dict], temperature: float = 0.1) -> str:
    resp = client().chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content
