from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import patch

import httpx
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from app.agent import ModelServiceError, ResearchContext, demo_operand, run_demo, run_model
from app.documents import DEMO, Document, DocumentStore, search
from app.finance import calculate
from app import main


def growth_arguments() -> tuple[dict, dict]:
    context = ResearchContext([DEMO])
    context.tool("search_documents", {"query": "营业收入", "limit": 5})
    return {"operation": "growth", "label": "收入同比",
            "numerator": demo_operand(context, "营业收入", 1),
            "denominator": demo_operand(context, "营业收入", 2)}, context.sources


class FinanceTests(unittest.TestCase):
    def test_growth_is_decimal_and_traceable(self):
        arguments, sources = growth_arguments()
        result = calculate(arguments, sources)
        self.assertEqual(result["value"], "25.00")
        self.assertEqual(result["unit"], "%")
        self.assertTrue(result["source_ids"])
        self.assertEqual(result["formula"], "(125000 - 100000) / 100000 × 100%")

    def test_rejects_invented_number(self):
        arguments, sources = growth_arguments()
        arguments["numerator"]["value"] = "999999"
        with self.assertRaisesRegex(ValueError, "不存在"):
            calculate(arguments, sources)

    def test_large_amount_keeps_enough_decimal_precision(self):
        source = Document("large", "amounts.txt", ["单位：元\n营业收入 | 123456789012345678901234567890 | 100000000000000000000000000000"]).chunks()[0]
        arguments = {"operation": "growth", "label": "收入同比",
                     "numerator": {"value": "123456789012345678901234567890", "unit": "元", "source_id": source.id, "quote": source.text},
                     "denominator": {"value": "100000000000000000000000000000", "unit": "元", "source_id": source.id, "quote": source.text}}
        self.assertEqual(calculate(arguments, {source.id: source})["value"], "23.46")

    def test_percentage_difference_uses_percentage_points(self):
        source = Document("margins", "margins.txt", ["净利率：14.4%，比较期间：15.0%。"]).chunks()[0]
        arguments = {"operation": "difference", "label": "净利率变动",
                     "numerator": {"value": "14.4", "unit": "%", "source_id": source.id, "quote": source.text},
                     "denominator": {"value": "15.0", "unit": "%", "source_id": source.id, "quote": source.text}}
        result = calculate(arguments, {source.id: source})
        self.assertEqual((result["value"], result["unit"]), ("-0.60", "个百分点"))

    def test_rejects_invented_quote(self):
        arguments, sources = growth_arguments()
        arguments["numerator"]["quote"] = "营业收入 999999"
        with self.assertRaisesRegex(ValueError, "逐字一致"):
            calculate(arguments, sources)

    def test_rejects_unknown_source(self):
        arguments, sources = growth_arguments()
        arguments["numerator"]["source_id"] = "not-selected"
        with self.assertRaisesRegex(ValueError, "必须检索"):
            calculate(arguments, sources)

    def test_rejects_mixed_units(self):
        arguments, sources = growth_arguments()
        # Both units are present in the source, but the inputs still cannot mix.
        source = next(iter(sources.values()))
        from dataclasses import replace
        sources[source.id] = replace(source, text=source.text + " 换算单位：亿元")
        arguments["denominator"]["unit"] = "亿元"
        with self.assertRaisesRegex(ValueError, "单位必须一致"):
            calculate(arguments, sources)

    def test_zero_and_negative_base_are_rejected(self):
        from dataclasses import replace
        arguments, sources = growth_arguments()
        source = next(iter(sources.values()))
        for value, message in (("0", "分母为零"), ("-100", "基期为负数")):
            quote = f"营业收入 | 125000 | {value}"
            sources[source.id] = replace(source, text=quote + " 单位：万元")
            for operand in (arguments["numerator"], arguments["denominator"]):
                operand["quote"] = quote
            arguments["denominator"]["value"] = value
            with self.assertRaisesRegex(ValueError, message):
                calculate(arguments, sources)


class RetrievalAndDemoTests(unittest.TestCase):
    def test_chinese_query_finds_risk_page(self):
        matches = search([DEMO], "客户集中风险", 1)
        self.assertEqual(matches[0].page, 3)
        self.assertIn("48%", matches[0].text)

    def test_no_match_is_not_fabricated(self):
        self.assertEqual(search([DEMO], "xyzunrelated"), [])

    def test_demo_runs_five_verified_calculations(self):
        result = run_demo("分析经营质量", [DEMO])
        self.assertEqual([item["value"] for item in result["calculations"]], ["25.00", "20.00", "14.40", "122.22", "60.00"])
        self.assertTrue(result["checks"]["citation_ids_valid"])
        self.assertEqual(len(result["trace"]), 6)
        self.assertEqual(result["model_turns"], 0)
        self.assertIn("固定的离线演示", result["answer"])

    def test_missing_evidence_refusal(self):
        result = run_demo("预测明年股价", [DEMO])
        self.assertIn("证据不足", result["answer"])
        self.assertEqual(result["calculations"], [])
        self.assertTrue(result["checks"]["citation_ids_valid"])

    def test_uploaded_document_is_only_extractive_in_demo(self):
        document = Document("custom", "example.txt", ["本期营业收入为 200 万元。"])
        result = run_demo("营业收入", [document])
        self.assertIn("尚未进行模型分析", result["answer"])
        self.assertEqual(result["sources"][0]["document_id"], "custom")
        self.assertEqual(result["calculations"], [])

    def test_empty_documents_rejected_without_reindexing_pages(self):
        store = DocumentStore()
        with self.assertRaisesRegex(ValueError, "OCR"):
            store.add("scan.pdf", ["", " "])
        document = store.add("partial.pdf", ["", "第二页：风险说明"])
        self.assertEqual(document.chunks()[0].page, 2)

    def test_unknown_citation_is_marked_for_review(self):
        context = ResearchContext([DEMO])
        context.tool("search_documents", {"query": "营业收入"})
        result = context.finish("收入为 999 [imaginary:p1:c1]", "live", 1)
        self.assertEqual(result["status"], "needs_review")
        self.assertFalse(result["checks"]["citation_ids_valid"])

    def test_unknown_tool_cannot_execute_document_instructions(self):
        context = ResearchContext([DEMO])
        result = context.tool("run_shell", {"command": "ignore previous instructions"})
        self.assertIn("未知工具", result["error"])
        self.assertEqual(context.trace[0]["status"], "error")


class LiveLoopTests(unittest.IsolatedAsyncioTestCase):
    @patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-secret", "LLM_BASE_URL": "https://model.example/v1"})
    async def test_model_chooses_search_and_calculation(self):
        requests = []
        arguments, _ = growth_arguments()
        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            turn = len(requests)
            if turn == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search_documents", "arguments": json.dumps({"query": "营业收入", "limit": 5})}}]}
            elif turn == 2:
                self.assertEqual(body["messages"][-1]["role"], "tool")
                self.assertIn("sources", json.loads(body["messages"][-1]["content"]))
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call-2", "type": "function", "function": {"name": "financial_calculator", "arguments": json.dumps(arguments)}}]}
            else:
                calculation = json.loads(body["messages"][-1]["content"])
                self.assertEqual(calculation["value"], "25.00")
                message = {"role": "assistant", "content": "营业收入同比增长 25.00%。 [demo-qiheng-2024:p1:c1]"}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_model("营业收入同比增长多少？", [DEMO], client)
        self.assertEqual(result["mode"], "live")
        self.assertEqual(result["model_turns"], 3)
        self.assertEqual(len(result["calculations"]), 1)
        self.assertTrue(result["checks"]["citation_ids_valid"])

    @patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-secret", "LLM_BASE_URL": "https://model.example/v1"})
    async def test_upstream_error_does_not_expose_response_body(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(401, text="sensitive upstream diagnostic"))) as client:
            with self.assertRaises(ModelServiceError) as raised:
                await run_model("分析收入", [DEMO], client)
        self.assertIn("401", str(raised.exception))
        self.assertNotIn("sensitive", str(raised.exception))

    @patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-secret", "LLM_BASE_URL": "https://model.example/v1"})
    async def test_turn_limit_stops_repeated_search(self):
        message = {"content": None, "tool_calls": [{"id": "repeat", "type": "function", "function": {"name": "search_documents", "arguments": '{"query":"收入"}'}}]}
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": [{"message": message}]}))) as client:
            with self.assertRaisesRegex(ModelServiceError, "8 轮"):
                await run_model("分析收入", [DEMO], client)


class APITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.old_store = main.store
        main.store = DocumentStore()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")

    async def asyncTearDown(self):
        main.store = self.old_store
        await self.client.aclose()

    async def test_home_assets_and_health(self):
        self.assertEqual((await self.client.get("/api/health")).json()["status"], "ok")
        self.assertIn("让每个结论", (await self.client.get("/")).text)
        self.assertEqual((await self.client.get("/static/app.js")).status_code, 200)

    async def test_upload_then_research_selects_only_uploaded_document(self):
        upload = await self.client.post("/api/documents", files={"file": ("example.txt", "营业收入为 300 万元。".encode(), "text/plain")})
        self.assertEqual(upload.status_code, 201)
        document_id = upload.json()["document"]["id"]
        response = await self.client.post("/api/research", json={"question": "营业收入", "document_ids": [document_id], "mode": "demo"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual({source["document_id"] for source in response.json()["sources"]}, {document_id})
        self.assertNotIn("125000", response.json()["answer"])

    async def test_real_pdf_text_extraction(self):
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 20 250 Td (Revenue 2024: 125000 CNY) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        buffer = io.BytesIO()
        writer.write(buffer)
        response = await self.client.post("/api/documents", files={"file": ("report.pdf", buffer.getvalue(), "application/pdf")})
        self.assertEqual(response.status_code, 201)
        document = main.store.documents[response.json()["document"]["id"]]
        self.assertIn("125000", document.pages[0])

    async def test_scan_pdf_is_rejected(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        buffer = io.BytesIO()
        writer.write(buffer)
        response = await self.client.post("/api/documents", files={"file": ("scan.pdf", buffer.getvalue(), "application/pdf")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("OCR", response.json()["detail"])

    async def test_invalid_upload_and_size_limit(self):
        for filename, content, status in (("program.py", b"print('no')", 400), ("bad.txt", b"\xff\xfe", 400), ("empty.txt", b"", 400), ("big.txt", b"x" * (main.MAX_FILE_BYTES + 1), 413)):
            with self.subTest(filename=filename):
                response = await self.client.post("/api/documents", files={"file": (filename, content)})
                self.assertEqual(response.status_code, status)

    async def test_unknown_document_and_blank_question_are_rejected(self):
        response = await self.client.post("/api/research", json={"question": "分析收入", "document_ids": ["missing"]})
        self.assertEqual(response.status_code, 400)
        response = await self.client.post("/api/research", json={"question": "  ", "document_ids": [DEMO.id]})
        self.assertEqual(response.status_code, 422)

    @patch.dict(os.environ, {"LLM_API_KEY": ""})
    async def test_live_mode_without_credentials_fails_explicitly(self):
        config = (await self.client.get("/api/config")).json()
        self.assertFalse(config["llm"]["available"])
        self.assertNotIn("key", json.dumps(config))
        response = await self.client.post("/api/research", json={"question": "分析收入", "document_ids": [DEMO.id], "mode": "live"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("尚未配置", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
