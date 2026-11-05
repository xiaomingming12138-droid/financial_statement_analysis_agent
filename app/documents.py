"""Page-aware local retrieval. No document text is treated as an instruction."""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass

MAX_CHARS = 300_000


@dataclass(frozen=True)
class Source:
    id: str
    document_id: str
    title: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Document:
    id: str
    title: str
    pages: list[str]
    is_demo: bool = False

    def metadata(self) -> dict:
        return {"id": self.id, "title": self.title, "pages": len(self.pages), "is_demo": self.is_demo}

    def chunks(self) -> list[Source]:
        result = []
        for page_no, page in enumerate(self.pages, start=1):
            # Overlap retains table headings and context without inventing PDF pages.
            for offset in range(0, len(page), 550):
                text = page[offset:offset + 750].strip()
                if text:
                    result.append(Source(f"{self.id}:p{page_no}:c{offset // 550 + 1}", self.id, self.title, page_no, text))
        return result


def tokens(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        words.extend(segment[i:i + 2] for i in range(max(1, len(segment) - 1)))
    return words


def search(documents: list[Document], query: str, limit: int = 5) -> list[Source]:
    """BM25 over Chinese bigrams / Latin words, with no network dependencies."""
    chunks = [source for document in documents for source in document.chunks()]
    if not chunks:
        return []
    term_counts = [Counter(tokens(source.text)) for source in chunks]
    lengths = [sum(counts.values()) for counts in term_counts]
    average_length = sum(lengths) / len(lengths) or 1
    query_tokens = set(tokens(query))
    scores = []
    for index, counts in enumerate(term_counts):
        score = 0.0
        for token in query_tokens:
            frequency = counts[token]
            if not frequency:
                continue
            document_frequency = sum(token in count for count in term_counts)
            idf = math.log(1 + (len(chunks) - document_frequency + 0.5) / (document_frequency + 0.5))
            score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * lengths[index] / average_length))
        if score > 0:
            scores.append((score, index))
    scores.sort(key=lambda entry: (-entry[0], entry[1]))
    return [chunks[index] for _, index in scores[:limit]]


DEMO = Document(
    id="demo-qiheng-2024",
    title="启衡科技 · 2024 年报（虚构示例）",
    is_demo=True,
    pages=[
        """启衡科技 2024 年度报告 — 虚构示例，非真实公司数据
第 1 页：经营业绩。报告期：2024 年；比较期间：2023 年。单位：万元，人民币。
指标 | 2024 年 | 2023 年
营业收入 | 125000 | 100000
营业成本 | 80000 | 68000
归母净利润 | 18000 | 15000
经营活动现金流量净额 | 22000 | 16000
公司主营企业软件与技术服务。收入增长主要来自企业软件订阅业务拓展。
以上金额均使用万元口径；这份文档仅用于验证研究流程。""",
        """启衡科技 2024 年度报告 — 虚构示例
第 2 页：财务状况。报告期末：2024 年；比较期末：2023 年。单位：万元，人民币。
指标 | 2024 年 | 2023 年
资产总额 | 150000 | 130000
负债总额 | 45000 | 40000
应收账款 | 32000 | 20000
货币资金 | 28000 | 24000
应收账款增幅高于收入增幅，部分客户结算周期延长。
期末财务状况采用时点口径，不代表期间平均余额。""",
        """启衡科技 2024 年度报告 — 虚构示例
第 3 页：风险与披露边界。
客户集中风险：前五大客户收入占营业收入的 48%，比较期间为 41%。
回款风险：应收账款增长较快，客户付款延迟可能影响后续现金流。
竞争风险：行业价格竞争可能压缩服务业务利润空间。
公司未在此示例中披露分部毛利、未来盈利预测、股票价格或估值倍数。
管理层说明属于定性陈述，不能据此推断未来业绩或作出投资建议。""",
    ],
)


class DocumentStore:
    """Single-user, in-memory MVP storage. Uploaded files are not written to disk."""

    def __init__(self) -> None:
        self.documents = {DEMO.id: DEMO}

    def add(self, title: str, pages: list[str]) -> Document:
        pages = [page.strip() for page in pages]
        total_chars = sum(len(page) for page in pages)
        if not total_chars:
            raise ValueError("没有提取到可检索文本；扫描版 PDF 请先进行 OCR。")
        if total_chars > MAX_CHARS:
            raise ValueError("文档文本超过 30 万字符，请上传相关章节。")
        if len(self.documents) >= 11:
            raise ValueError("当前实例最多保留 10 份上传文档；重启可清空。")
        document = Document(f"doc-{uuid.uuid4().hex[:12]}", title[:160], pages)
        self.documents[document.id] = document
        return document

    def select(self, ids: list[str]) -> list[Document]:
        if not ids:
            raise ValueError("请至少选择一份研究文档。")
        if any(doc_id not in self.documents for doc_id in ids):
            raise ValueError("文档已不存在，请刷新页面重新上传。")
        return [self.documents[doc_id] for doc_id in dict.fromkeys(ids)]
