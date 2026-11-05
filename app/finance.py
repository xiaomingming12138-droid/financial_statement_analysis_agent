"""Deterministic Decimal arithmetic with source and unit verification."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.documents import Source


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
    if operand.unit not in source.text:
        raise ValueError("证据没有明确支持所选单位。")
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


def calculate(arguments: dict, sources: dict[str, Source]) -> dict:
    request = Calculation.model_validate(arguments)
    numerator = validated_value(request.numerator, sources)
    denominator = validated_value(request.denominator, sources)
    if request.numerator.unit != request.denominator.unit:
        raise ValueError("两个输入的单位必须一致；请先统一口径。")
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
    }
