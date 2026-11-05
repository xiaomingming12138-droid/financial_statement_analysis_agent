"""Deterministic Decimal arithmetic with source and unit verification."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.documents import Source, unit_declarations


class Operand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = Field(max_length=40, description="原文中的数值，不含千位分隔符")
    unit: Literal["万元", "亿元", "元", "%"]
    source_id: str = Field(max_length=100)
    quote: str = Field(min_length=1, max_length=750, description="包含该数值的原文逐字引文")


class Calculation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["growth", "ratio", "difference"]
    label: str = Field(min_length=1, max_length=80)
    numerator: Operand
    denominator: Operand


def validated_value(operand: Operand, sources: dict[str, Source]) -> Decimal:
    source = sources.get(operand.source_id)
    if source is None:
        raise ValueError("计算前必须检索并引用当前文档中的证据。")
    if operand.quote not in source.text:
        raise ValueError("计算引文必须与检索原文逐字一致。")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", operand.value):
        raise ValueError("数值必须是有限十进制数。")
    try:
        value = Decimal(operand.value)
        numbers = [Decimal(number.replace(",", "")) for number in re.findall(r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?(?![\d.])", operand.quote)]
    except InvalidOperation as error:
        raise ValueError("无法解析计算数值。") from error
    if value not in numbers:
        raise ValueError("引文中不存在该数值，拒绝计算。")
    return value


def validate_unit(operand: Operand, value: Decimal, source: Source) -> dict:
    # An inline unit must be attached to the actual input value, not an unrelated
    # amount elsewhere in the chunk. This also supports percentage-point changes.
    inline = []
    for match in re.finditer(r"(-?\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|元|[%％])(?!\s*[/／])", operand.quote):
        if Decimal(match.group(1).replace(",", "")) == value:
            inline.append((match.group(2).replace("％", "%"), match.group(0)))
    if inline:
        units = {unit for unit, _ in inline}
        if units == {operand.unit}:
            return {"source_id": source.id, "quote": inline[0][1], "scope": "value"}
        raise ValueError("引文中该数值的单位与输入不一致，或存在多个单位；请核对单位及引文范围。")

    declarations = unit_declarations(source.text)
    scope = "chunk"
    if not declarations:
        declarations = unit_declarations(source.unit_context)
        scope = "same_page"
    units = {unit for unit, _ in declarations}
    if units == {operand.unit}:
        return {"source_id": source.id, "quote": declarations[0][1], "scope": scope}
    if units:
        raise ValueError("原文单位说明与输入不一致或有歧义；请按单位说明选择元、万元、亿元或百分比。")
    raise ValueError("当前片段及同页上下文没有明确单位说明；请检索单位/币种表头，不能默认使用元或原样重复计算。")


def calculate(arguments: dict, sources: dict[str, Source]) -> dict:
    request = Calculation.model_validate(arguments)
    if request.numerator.unit != request.denominator.unit:
        raise ValueError("两个输入的单位必须一致；请先统一口径。")
    numerator = validated_value(request.numerator, sources)
    denominator = validated_value(request.denominator, sources)
    unit_evidence = {
        "numerator": validate_unit(request.numerator, numerator, sources[request.numerator.source_id]),
        "denominator": validate_unit(request.denominator, denominator, sources[request.denominator.source_id]),
    }
    if request.operation != "difference" and denominator == 0:
        raise ValueError("分母为零，不能计算比率或增长率。")
    if request.operation == "growth" and denominator < 0:
        raise ValueError("基期为负数，不使用普通同比公式；请解释亏损变化。")
    # Inputs allow up to 40 characters. Keep enough precision even for a ratio
    # between the largest allowed integer and the smallest allowed fraction.
    with localcontext() as decimal_context:
        decimal_context.prec = 100
        if request.operation == "growth":
            result = (numerator - denominator) / denominator * 100
            formula = f"({numerator} - {denominator}) / {denominator} × 100%"
        elif request.operation == "ratio":
            result = numerator / denominator * 100
            formula = f"{numerator} / {denominator} × 100%"
        else:
            result = numerator - denominator
            formula = f"{numerator} - {denominator}"
        rounded = str(result.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    unit = request.numerator.unit if request.operation == "difference" else "%"
    if request.operation == "difference" and unit == "%":
        unit = "个百分点"
    return {
        "label": request.label,
        "value": rounded,
        "unit": unit,
        "formula": formula,
        "source_ids": list(dict.fromkeys([request.numerator.source_id, request.denominator.source_id])),
        "inputs": {"numerator": request.numerator.model_dump(), "denominator": request.denominator.model_dump()},
        "unit_evidence": unit_evidence,
    }
