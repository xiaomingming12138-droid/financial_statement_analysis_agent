"""Reproducible offline regression cases, not a claim of real-model accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import ResearchContext, run_demo
from app.documents import DEMO, Document


def evaluate() -> dict:
    cases = []
    for name, question, expected_values, refusal in [
        ("经营质量与计算", "分析收入、利润与现金流", ["25.00", "20.00", "14.40", "122.22", "60.00"], False),
        ("回款风险与引用", "应收账款增长有哪些风险", ["60.00"], False),
        ("股价预测证据不足", "预测明年股价", [], True),
        ("未披露估值", "公司的估值是多少", [], True),
    ]:
        result = run_demo(question, [DEMO])
        values = [item["value"] for item in result["calculations"]]
        checks = {
            "expected_values": all(value in values for value in expected_values),
            "valid_citations": result["checks"]["citation_ids_valid"],
            "no_tool_errors": all(event["status"] == "success" for event in result["trace"]),
            "refusal_when_needed": not refusal or ("证据不足" in result["answer"] and not values),
        }
        cases.append({"name": name, "passed": all(checks.values()), "checks": checks})
    result = run_demo("quantumxyz", [Document("other", "other.txt", ["营业收入为 300 万元。 "])])
    cases.append({"name": "无匹配不编造", "passed": "证据不足" in result["answer"] and not result["sources"]})
    context = ResearchContext([DEMO])
    context.tool("search_documents", {"query": "收入"})
    result = context.finish("无效结论 [not-a-real-source]", "live", 1)
    cases.append({"name": "未知引用被标记", "passed": result["status"] == "needs_review"})
    return {"suite": "offline-regression", "scope": "固定演示流程与引用检查；不衡量真实模型质量", "passed": sum(case["passed"] for case in cases), "total": len(cases), "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate()
    formatted = json.dumps(report, ensure_ascii=False, indent=2)
    print(formatted)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
