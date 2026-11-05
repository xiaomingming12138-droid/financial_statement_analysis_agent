from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

from app.agent import ModelServiceError, model_settings, run_demo, run_model
from app.documents import DocumentStore

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="研迹 · 财报研究 Agent", version="0.1.0")
store = DocumentStore()
MAX_FILE_BYTES = 5 * 1024 * 1024
research_slots = asyncio.Semaphore(2)


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=2, max_length=1000)
    document_ids: list[str] = Field(min_length=1, max_length=10)
    mode: Literal["demo", "live"] = "demo"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/api/config")
def configuration() -> dict:
    return {"llm": model_settings(), "limits": {"upload_mb": 5, "text_chars": 300000, "tool_calls": 12, "model_turns": 8}}


@app.get("/api/documents")
def documents() -> dict:
    return {"documents": [document.metadata() for document in store.documents.values()]}


def extract_pages(payload: bytes, filename: str) -> list[str]:
    extension = Path(filename).suffix.lower()
    if extension in (".txt", ".md"):
        try:
            return [payload.decode("utf-8-sig")]
        except UnicodeDecodeError as error:
            raise ValueError("文本文件需使用 UTF-8 编码。") from error
    if extension != ".pdf":
        raise ValueError("支持 PDF、TXT 和 Markdown 文件。")
    try:
        reader = PdfReader(io.BytesIO(payload))
        if reader.is_encrypted:
            raise ValueError("暂不支持加密 PDF，请先解密。")
        if len(reader.pages) > 120:
            raise ValueError("PDF 最多支持 120 页，请上传相关章节。")
        pages = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            total += len(text)
            if total > 300000:
                raise ValueError("文档文本超过 30 万字符，请上传相关章节。")
            pages.append(text)
        return pages
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("无法解析此 PDF，请检查文件是否完整，或上传 UTF-8 文本。") from error


@app.post("/api/documents", status_code=201)
async def upload(file: UploadFile) -> dict:
    try:
        payload = await file.read(MAX_FILE_BYTES + 1)
        if len(payload) > MAX_FILE_BYTES:
            raise HTTPException(413, "文件超过 5 MB，请上传相关章节。")
        filename = Path((file.filename or "document").replace("\\", "/")).name
        try:
            pages = await asyncio.to_thread(extract_pages, payload, filename)
            document = store.add(filename, pages)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return {"document": document.metadata()}
    finally:
        await file.close()


@app.post("/api/research")
async def research(request: ResearchRequest) -> dict:
    if len(request.question.strip()) < 2:
        raise HTTPException(422, "请输入至少两个字的研究问题。")
    try:
        selected = store.select(request.document_ids)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    async with research_slots:
        try:
            if request.mode == "demo":
                return await asyncio.to_thread(run_demo, request.question, selected)
            return await asyncio.wait_for(run_model(request.question, selected), timeout=180)
        except ModelServiceError as error:
            raise HTTPException(502, str(error)) from error
        except TimeoutError as error:
            raise HTTPException(504, "研究超过 180 秒，请缩小问题后重试。") from error


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
