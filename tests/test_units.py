"""Regression for a financial table separated from its page's unit header."""

import unittest

from app.documents import Document, Source, search, unit_declarations
from app.finance import calculate


TABLE = "项 目 2025 年度 2024 年度\n营业收入 285,997,870.92 256,541,927.31"
FILLER = "财务报表编制说明。\n" * 90


def arguments(source):
    return {
        "operation": "growth", "label": "合并营业收入2025较2024同比",
        "numerator": {"value": "285997870.92", "unit": "元", "source_id": source.id, "quote": TABLE},
        "denominator": {"value": "256541927.31", "unit": "元", "source_id": source.id, "quote": TABLE},
    }


def table_source(header="单位：元 币种：人民币", previous_page=""):
    document = Document("doc-fb14129f5bed", "uploaded.pdf", [""] * 33 + [previous_page, header + "\n" + FILLER + TABLE])
    source = next(source for source in document.chunks() if source.page == 35 and TABLE in source.text)
    return document, source


class UnitEvidenceTests(unittest.TestCase):
    def test_uploaded_table_uses_same_page_header_without_altering_quote(self):
        document, source = table_source()
        self.assertNotIn("元", source.text)
        self.assertEqual(source.unit_context, "单位：元")
        self.assertEqual(source.page, 35)
        found = search([document], "营业收入", 1)[0]
        result = calculate(arguments(found), {found.id: found})
        self.assertEqual(result["value"], "11.48")
        self.assertEqual(result["unit"], "%")
        self.assertEqual(result["unit_evidence"]["numerator"]["scope"], "same_page")
        self.assertEqual(result["unit_evidence"]["numerator"]["quote"], "单位：元")
        self.assertEqual(found.to_dict()["unit_context"], "单位：元")
        self.assertEqual(result["inputs"]["numerator"]["quote"], TABLE)

    def test_missing_unit_still_rejected(self):
        _, source = table_source("")
        with self.assertRaisesRegex(ValueError, "没有明确单位说明"):
            calculate(arguments(source), {source.id: source})

    def test_header_on_another_page_is_not_inherited(self):
        _, source = table_source("", previous_page="单位：元 币种：人民币")
        self.assertEqual(source.unit_context, "")
        with self.assertRaisesRegex(ValueError, "没有明确单位说明"):
            calculate(arguments(source), {source.id: source})

    def test_conflicting_page_declarations_are_not_inherited(self):
        _, source = table_source("单位：元\n其他表格单位：万元")
        self.assertEqual(source.unit_context, "")
        with self.assertRaisesRegex(ValueError, "没有明确单位说明"):
            calculate(arguments(source), {source.id: source})

    def test_yuan_cannot_match_ten_thousand_yuan(self):
        for header in ("单位：万元", "单位：亿元"):
            with self.subTest(header=header):
                _, source = table_source(header)
                with self.assertRaisesRegex(ValueError, "单位说明与输入不一致"):
                    calculate(arguments(source), {source.id: source})

    def test_pdf_spacing_and_newline_in_header(self):
        for header in ("单 位 ： 人 民 币 元", "单位：\n元", "单位（元）"):
            with self.subTest(header=header):
                _, source = table_source(header)
                result = calculate(arguments(source), {source.id: source})
                self.assertEqual(result["value"], "11.48")
                self.assertEqual(unit_declarations(source.unit_context)[0][0], "元")

    def test_unrelated_word_containing_yuan_is_not_unit_evidence(self):
        source = Source("example:p1:c1", "example", "report.txt", 1, TABLE + "\n公元2025年的说明")
        with self.assertRaisesRegex(ValueError, "没有明确单位说明"):
            calculate(arguments(source), {source.id: source})

    def test_inline_amounts_do_not_match_smaller_unit(self):
        source = Source("example:p1:c1", "example", "report.txt", 1, "收入：10万元，比较期间：20万元。")
        request = arguments(source)
        for operand, value in (("numerator", "10"), ("denominator", "20")):
            request[operand].update(value=value, quote=source.text)
        with self.assertRaisesRegex(ValueError, "数值的单位与输入不一致"):
            calculate(request, {source.id: source})


if __name__ == "__main__":
    unittest.main()
