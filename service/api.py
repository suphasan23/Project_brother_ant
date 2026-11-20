"""HTTP API ตามเอกสาร API Contract: n8n ↔ Search Service

รันจากโฟลเดอร์ราก repo:
    uvicorn service.api:app --host 0.0.0.0 --port 8000
แล้วเปิด http://localhost:8000/docs เพื่อลองเรียกผ่าน Swagger
"""

import secrets
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from service.config import load_settings
from service.indexer import DocumentIndex, DocumentMeta, NoTextError
from service.pdf_text import PdfError, is_pdf

settings = load_settings()
if settings.api_key == "":
    print("✘ ยังไม่ได้ตั้ง API_KEY ในไฟล์ .env (ดูตัวอย่างใน .env.example)")
    sys.exit(1)

index: DocumentIndex | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index
    index = DocumentIndex(settings)  # โหลดโมเดลครั้งเดียวตอนเริ่ม
    print(f"พร้อมใช้งาน: {index.count_documents()} เอกสารใน index")
    yield


app = FastAPI(title="mini-search Search Service", version="1.0.0", lifespan=lifespan)


# ───────────────────────── error ทุกแบบตอบรูปแบบเดียวกัน ─────────────────────────

class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(ApiError)
async def handle_api_error(request: Request, exc: ApiError):
    return error_response(exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    problems = []
    for err in exc.errors():
        # loc เช่น ("body", "query") ส่วนแรกบอกว่าอยู่ตรงไหนของ request ตัดทิ้ง เหลือแค่ชื่อ field
        location = list(err["loc"])
        if location and location[0] in ("body", "query", "path", "header"):
            location = location[1:]
        field = ".".join(str(part) for part in location)
        problems.append(f"{field}: {err['msg']}")
    return error_response(422, "invalid_request", "; ".join(problems))


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request: Request, exc: StarletteHTTPException):
    return error_response(exc.status_code, "http_error", str(exc.detail))


# ───────────────────────── ตรวจ API key ─────────────────────────

def require_api_key(x_api_key: str | None = Header(default=None)):
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise ApiError(401, "unauthorized", "API key ไม่ถูกต้อง (ส่งใน header X-API-Key)")


public = APIRouter(prefix="/api/v1")
protected = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


# ───────────────────────── endpoints ─────────────────────────

@public.get("/health")
def health():
    return {"status": "ok", "model": index.model_name, "documents": index.count_documents()}


def check_iso_time(field: str, value: str) -> str:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ApiError(422, "invalid_request", f"{field}: ต้องเป็นเวลา ISO 8601 เช่น 2026-11-18T09:15:22.000Z")
    return value


@protected.post("/documents")
def upsert_document(
    file: UploadFile = File(...),
    file_id: str = Form(..., min_length=1),
    name: str = Form(..., min_length=1),
    mime_type: str = Form(...),
    folder_path: str = Form(...),
    modified_time: str = Form(...),
    web_view_link: str = Form(...),
    created_time: str = Form(""),
    owners: str = Form(""),
    description: str = Form(""),
):
    started = time.perf_counter()

    if mime_type != "application/pdf":
        raise ApiError(415, "unsupported_file_type", "รองรับเฉพาะ pdf")
    check_iso_time("modified_time", modified_time)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ApiError(413, "file_too_large", f"ไฟล์ใหญ่เกิน {settings.max_upload_mb} MB")
    if not is_pdf(data):
        raise ApiError(415, "unsupported_file_type", "เนื้อไฟล์ไม่ใช่ PDF")

    meta = DocumentMeta(
        file_id=file_id, name=name, mime_type=mime_type, folder_path=folder_path,
        modified_time=modified_time, web_view_link=web_view_link,
        created_time=created_time, owners=owners, description=description,
    )
    try:
        result = index.upsert(meta, data)
    except PdfError as e:
        raise ApiError(422, "invalid_pdf", str(e))
    except NoTextError as e:
        raise ApiError(422, "no_text_extracted", str(e))

    took_ms = int((time.perf_counter() - started) * 1000)
    return {"file_id": file_id, **result, "took_ms": took_ms}


@protected.get("/documents")
def list_documents():
    documents = index.list_documents()
    return {"documents": documents, "total": len(documents)}


@protected.delete("/documents/{file_id}")
def delete_document(file_id: str):
    deleted = index.delete(file_id)
    if deleted is None:
        raise ApiError(404, "not_found", f"ไม่มีไฟล์ {file_id} ใน index")
    return {"file_id": file_id, "deleted_chunks": deleted}


class SearchFilters(BaseModel):
    folder_prefix: str | None = None
    modified_after: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    max_documents: int = Field(50, ge=1, le=200)
    min_score: float | None = Field(None, ge=0, le=1)
    filters: SearchFilters | None = None


@protected.post("/search")
def search(request: SearchRequest):
    filters = request.filters or SearchFilters()
    if filters.modified_after:
        check_iso_time("filters.modified_after", filters.modified_after)
    min_score = request.min_score if request.min_score is not None else settings.search_min_score
    return index.search(
        query=request.query.strip(),
        max_documents=request.max_documents,
        min_score=min_score,
        folder_prefix=filters.folder_prefix,
        modified_after=filters.modified_after,
    )


app.include_router(public)
app.include_router(protected)
