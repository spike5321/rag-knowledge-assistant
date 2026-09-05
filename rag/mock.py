"""模拟数据层：Demo 专用，不调用任何真实 API。

与界面解耦——demo.py 只依赖这里的函数，之后接入真实后端时
把 pipeline 换成 rag.pipeline.answer 即可，界面代码不动。
"""
import time

# 模拟的知识库片段：来源文件 → 切分后的片段文本
MOCK_KNOWLEDGE_BASE: dict[str, list[str]] = {
    "员工手册.pdf": [
        "年假制度：入职满一年的员工享有 5 天带薪年假，满三年 10 天，满五年 15 天。年假需提前 3 个工作日在 OA 系统申请，跨年不清零，最多结转至次年 3 月 31 日。",
        "报销流程：差旅报销需在行程结束后 15 个工作日内提交，机票和酒店需提供电子发票。单笔超过 5000 元的报销需部门总监和财务双审批，一般 5 个工作日内到账。",
        "考勤规定：公司实行弹性工作制，核心工作时间为 10:00-16:00，每日工作满 8 小时即可。迟到 30 分钟以内每月前两次不计，超过两次按每次 50 元计入月度考核。",
    ],
    "RAG技术调研.md": [
        "混合检索指向量检索与 BM25 关键词检索的结合。向量检索擅长语义相似，但对专有名词、编号等精确匹配场景较弱；BM25 相反。两者用 RRF（倒数排序融合）合并，是当前生产环境 RAG 的主流做法。",
        "Rerank（重排）在粗检索之后执行：先用向量检索召回 top-50，再用交叉编码器对 query 和每个片段精细打分，取 top-5 进入生成环节。典型模型为 bge-reranker，能显著提升相关片段的排序质量。",
        "评测 RAG 常用 RAGAS 框架，核心指标包括：忠实度（答案是否忠于检索内容）、答案相关性、上下文精确率（检索片段中有用占比）、上下文召回率（是否召回所有必要片段）。",
    ],
}

# 模拟回答库：按关键词匹配，均带有 [n] 引用编号
MOCK_ANSWERS: list[tuple[tuple[str, ...], str]] = [
    (
        ("年假", "休假", "请假"),
        "根据员工手册，年假按司龄分级：入职满一年 **5 天**，满三年 **10 天**，满五年 **15 天** [1]。"
        "需要注意两点：一是需提前 3 个工作日在 OA 系统申请 [1]；二是年假跨年不清零，最多结转至次年 3 月 31 日 [1]。",
    ),
    (
        ("报销", "发票", "差旅"),
        "报销流程如下：差旅报销需在行程结束后 15 个工作日内提交，机票和酒店需提供电子发票 [1]。"
        "单笔超过 5000 元需要部门总监和财务双重审批，一般 5 个工作日内到账 [1]。",
    ),
    (
        ("考勤", "迟到", "上班时间", "弹性"),
        "公司实行弹性工作制，核心工作时间为 10:00-16:00，每日工作满 8 小时即可 [1]。"
        "关于迟到：30 分钟以内每月前两次不计，超过两次按每次 50 元计入月度考核 [1]。",
    ),
    (
        ("rerank", "重排", "bge"),
        "Rerank 是在粗检索之后进行的精排步骤：先向量检索召回 top-50，再用交叉编码器对 query 与片段精细打分，"
        "取 top-5 进入生成环节 [1]。典型模型是 bge-reranker，能显著提升排序质量 [1]。"
        "它通常与混合检索配合使用 [2]。",
    ),
    (
        ("ragas", "评测", "评估", "指标"),
        "RAG 评测常用 RAGAS 框架，四个核心指标 [1]：\n"
        "1. **忠实度**——答案是否忠于检索内容\n"
        "2. **答案相关性**——回答是否切题\n"
        "3. **上下文精确率**——检索片段中有用内容的占比\n"
        "4. **上下文召回率**——是否召回了所有必要片段",
    ),
    (
        ("混合检索", "bm25", "向量检索"),
        "混合检索是向量检索与 BM25 的结合 [1]：向量检索擅长语义相似，但对专有名词、编号等精确匹配较弱；"
        "BM25 恰好相反。两者结果通过 RRF（倒数排序融合）合并 [1]，是当前生产环境 RAG 的主流做法 [1]。",
    ),
]

DEFAULT_ANSWER = (
    "抱歉，本 Demo 的模拟知识库只收录了《员工手册》和《RAG 技术调研》两份文档，"
    "没有找到与该问题相关的内容。你可以试试问：\n\n"
    "- 年假有几天？怎么申请？\n- 差旅报销流程是什么？\n- 什么是混合检索？\n- RAGAS 有哪些评测指标？"
)


def mock_ingest(filenames: list[str]) -> list[dict]:
    """模拟文档入库，返回入库结果（片段数与来源为模拟数据）。"""
    time.sleep(0.8)  # 模拟解析与向量化耗时
    return [
        {"source": name, "chunks": len(MOCK_KNOWLEDGE_BASE.get(name, [])) or 12}
        for name in filenames
    ]


def mock_retrieve(question: str, k: int = 4) -> list[dict]:
    """按关键词挑出与问题最相关的模拟片段，构造引用列表。"""
    q = question.lower()
    keywords = _keywords(question)
    hits: list[dict] = []
    for source, chunks in MOCK_KNOWLEDGE_BASE.items():
        for i, text in enumerate(chunks):
            # 关键词需同时出现在问题和片段文本中才算相关，命中越多分越高
            matched = sum(1 for w in keywords if w in q and w in text.lower())
            if matched:
                hits.append(
                    {
                        "source": source,
                        "chunk": i,
                        "text": text,
                        "score": round(min(0.95, 0.6 + 0.08 * matched), 2),
                    }
                )
    hits.sort(key=lambda h: -h["score"])
    return hits[:k]


def _keywords(question: str) -> list[str]:
    return [w for w in ("年假", "报销", "考勤", "迟到", "rerank", "重排", "ragas", "评测", "混合检索", "bm25") if w in question.lower()]


def mock_answer(question: str) -> tuple[str, list[dict]]:
    """返回 (模拟回答, 引用片段)。先查回答库，查不到就走兜底话术。"""
    time.sleep(1.0)  # 模拟大模型生成耗时
    q = question.lower()
    for keywords, answer in MOCK_ANSWERS:
        if any(k in q for k in keywords):
            hits = mock_retrieve(question)
            return answer, hits
    return DEFAULT_ANSWER, []
