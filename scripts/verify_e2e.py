"""端到端验收脚本（一次性）：文档内提问 + 文档外提问。"""
from rag.pipeline import answer

for q in [
    "试用期是多长时间？社保和公积金的缴存比例是多少？",
    "音箱的保修政策是怎样的？恢复出厂设置怎么操作？",
    "公司提供的补充商业医疗保险什么时候生效？",
    "公司食堂几点开饭？",
]:
    print("=" * 60)
    print("问题:", q)
    reply, hits = answer(q)
    print("回答:", reply)
    print("引用:", [f"{h['source']}({h['score']:.2f})" for h in hits])
