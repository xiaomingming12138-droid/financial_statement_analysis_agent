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

    def test_repeated_calculation_reuses_result_without_duplicate_cards(self):
        arguments, _ = growth_arguments()
        context = ResearchContext([DEMO])
        context.tool("search_documents", {"query": "营业收入", "limit": 5})
        first = context.tool("financial_calculator", arguments)
        second = context.tool("financial_calculator", arguments)
        self.assertEqual(first["value"], second["value"])
        self.assertEqual(len(context.calculations), 1)
        self.assertTrue(context.trace[-1]["cached"])


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
                message = {"role": "assistant", "content": None, "reasoning_content": "provider-private-reasoning", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search_documents", "arguments": json.dumps({"query": "营业收入", "limit": 5})}}]}
            elif turn == 2:
                tool_result = next(message for message in reversed(body["messages"]) if message["role"] == "tool")
                self.assertIn("sources", json.loads(tool_result["content"]))
                assistant = next(message for message in body["messages"] if message["role"] == "assistant")
                self.assertEqual(assistant["reasoning_content"], "provider-private-reasoning")
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call-2", "type": "function", "function": {"name": "financial_calculator", "arguments": json.dumps(arguments)}}]}
            else:
                tool_result = next(message for message in reversed(body["messages"]) if message["role"] == "tool")
                calculation = json.loads(tool_result["content"])
                self.assertEqual(calculation["value"], "25.00")
                message = {"role": "assistant", "content": "营业收入同比增长 25.00%。 [demo-qiheng-2024:p1:c1]"}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_model("营业收入同比增长多少？", [DEMO], client)
        self.assertEqual(result["mode"], "live")
        self.assertEqual(result["model_turns"], 3)
        self.assertEqual(len(result["calculations"]), 1)
        self.assertTrue(result["checks"]["citation_ids_valid"])
        self.assertNotIn("provider-private-reasoning", json.dumps(result))

    @patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-secret", "LLM_BASE_URL": "https://model.example/v1"})
    async def test_upstream_error_does_not_expose_response_body(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(401, text="sensitive upstream diagnostic"))) as client:
            with self.assertRaises(ModelServiceError) as raised:
                await run_model("分析收入", [DEMO], client)
        self.assertIn("401", str(raised.exception))
        self.assertNotIn("sensitive", str(raised.exception))

    @patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-secret", "LLM_BASE_URL": "https://model.example/v1"})
    async def test_repeated_search_stops_and_preserves_evidence(self):
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            message = {"content": None, "tool_calls": [{"id": f"repeat-{len(requests)}", "type": "function", "function": {"name": "search_documents", "arguments": '{"query":"营业收入"}'}}]}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_model("分析收入", [DEMO], client)
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[-1]["tool_choice"], "none")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["answer_origin"], "evidence_summary")
        self.assertEqual(len(result["trace"]), 2)
        self.assertFalse(result["trace"][0]["cached"])
        self.assertTrue(result["trace"][1]["cached"])
        self.assertTrue(result["sources"])
        self.assertIn("并非模型生成", result["answer"])
        self.assertTrue(result["checks"]["citation_ids_valid"])

    @patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-secret", "LLM_BASE_URL": "https://model.example/v1"})
    async def test_tool_budget_returns_summary_and_replies_to_skipped_calls(self):
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                calls = [{"id": f"search-{i}", "type": "function", "function": {"name": "search_documents", "arguments": json.dumps({"query": f"营业收入 {i}"})}} for i in range(14)]
                message = {"content": None, "tool_calls": calls}
            else:
                self.assertEqual(body["tool_choice"], "none")
                replies = [message for message in body["messages"] if message["role"] == "tool"]
                self.assertEqual(len(replies), 14)
                self.assertEqual(len({message["tool_call_id"] for message in replies}), 14)
                self.assertIn("未执行", json.loads(replies[-1]["content"])["error"])
                message = {"content": "已找到营业收入数据；其余分析未完成。 [demo-qiheng-2024:p1:c1]"}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_model("分析经营质量", [DEMO], client)
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(result["trace"]), 12)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["answer_origin"], "model")
        self.assertTrue(result["checks"]["citation_ids_valid"])

    @patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-secret", "LLM_BASE_URL": "https://model.example/v1"})
    async def test_last_turn_is_reserved_for_an_answer(self):
        requests = []
        arguments, _ = growth_arguments()
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            turn = len(requests)
            if turn == 1:
                name, params = "search_documents", {"query": "营业收入"}
            elif turn <= 7:
                name, params = "financial_calculator", {**arguments, "label": f"收入同比复核 {turn}"}
            else:
                self.assertEqual(body["tool_choice"], "none")
                return httpx.Response(200, json={"choices": [{"message": {"content": "收入同比 25.00%。 [demo-qiheng-2024:p1:c1]"}}]})
            message = {"content": None, "tool_calls": [{"id": f"call-{turn}", "type": "function", "function": {"name": name, "arguments": json.dumps(params)}}]}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_model("分析经营质量", [DEMO], client)
        self.assertEqual(len(requests), 8)
        self.assertEqual(result["model_turns"], 8)
        self.assertEqual(len(result["trace"]), 7)
        self.assertEqual(result["status"], "partial")

    @patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-secret", "LLM_BASE_URL": "https://model.example/v1"})
    async def test_no_progress_stops_even_when_queries_differ(self):
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if body["tool_choice"] == "none":
                message = {"content": "已找到营业收入原文，其余证据不足。 [demo-qiheng-2024:p1:c1]"}
            else:
                message = {"content": None, "tool_calls": [{"id": f"q-{len(requests)}", "type": "function", "function": {"name": "search_documents", "arguments": json.dumps({"query": f"营业收入 zz{len(requests)}", "limit": 1})}}]}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_model("分析营业收入", [DEMO], client)
        self.assertEqual(len(requests), 4)
        self.assertEqual(len(result["trace"]), 3)
        self.assertEqual(result["status"], "partial")

    @patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-secret", "LLM_BASE_URL": "https://model.example/v1"})
    async def test_summary_service_failure_preserves_work_without_body_leak(self):
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if body["tool_choice"] == "none":
                return httpx.Response(400, text="private-upstream-body")
            message = {"content": None, "tool_calls": [{"id": f"repeat-{len(requests)}", "type": "function", "function": {"name": "search_documents", "arguments": '{"query":"营业收入"}'}}]}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_model("分析收入", [DEMO], client)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["sources"])
        self.assertIn("400", " ".join(result["warnings"]))
        self.assertNotIn("private-upstream-body", json.dumps(result))

    @patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-secret", "LLM_BASE_URL": "https://model.example/v1"})
    async def test_fallback_keeps_successful_calculations_and_citations(self):
        arguments, _ = growth_arguments()
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            turn = len(requests)
            name, params = ("search_documents", {"query": "营业收入"}) if turn == 1 else ("financial_calculator", arguments)
            message = {"content": None, "tool_calls": [{"id": f"call-{turn}", "type": "function", "function": {"name": name, "arguments": json.dumps(params)}}]}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_model("分析收入增长", [DEMO], client)
        self.assertEqual(len(requests), 4)
        self.assertEqual(requests[-1]["tool_choice"], "none")
        self.assertEqual(result["answer_origin"], "evidence_summary")
        self.assertEqual(len(result["calculations"]), 1)
        self.assertEqual(result["calculations"][0]["value"], "25.00")
        self.assertIn("25.00%", result["answer"])
        self.assertTrue(result["checks"]["citation_ids_valid"])

    @patch.dict(os.environ, {"LLM_API_KEY": "test-not-a-real-secret", "LLM_BASE_URL": "https://model.example/v1"})
    async def test_failed_calculation_can_be_corrected_with_field_feedback(self):
        arguments, _ = growth_arguments()
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            turn = len(requests)
            if turn == 1:
                name, params = "search_documents", {"query": "营业收入"}
            elif turn == 2:
                name, params = "financial_calculator", {**arguments, "numerator": {**arguments["numerator"], "value": ["private-input-data"]}}
            elif turn == 3:
                feedback = next(message for message in reversed(body["messages"]) if message["role"] == "tool")["content"]
                self.assertIn("numerator.value", feedback)
                self.assertNotIn("private-input-data", feedback)
                name, params = "financial_calculator", arguments
            else:
                message = {"content": "营业收入同比 25.00%。 [demo-qiheng-2024:p1:c1]"}
                return httpx.Response(200, json={"choices": [{"message": message}]})
            message = {"content": None, "tool_calls": [{"id": f"c-{turn}", "type": "function", "function": {"name": name, "arguments": json.dumps(params)}}]}
            return httpx.Response(200, json={"choices": [{"message": message}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_model("分析收入", [DEMO], client)
        self.assertEqual(len(requests), 4)
        self.assertEqual(len(result["calculations"]), 1)
        self.assertEqual(result["calculations"][0]["value"], "25.00")
        self.assertEqual(result["status"], "completed")


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
