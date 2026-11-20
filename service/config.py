"""ค่าตั้งค่าทั้งหมดของ service อ่านจาก .env (ดูตัวอย่างใน .env.example)"""

import os
from dataclasses import dataclass
from pathlib import Path

from service import ROOT_DIR


@dataclass(frozen=True)
class Settings:
    api_key: str
    embedding_model: str      # path ในเครื่อง หรือชื่อโมเดลบน Hugging Face
    device: str               # auto, cpu, cuda, cuda:0
    batch_size: int
    max_tokens: int
    index_dir: Path
    collection: str
    max_upload_mb: int
    search_candidates: int    # ดึงท่อนมากี่ท่อนก่อนรวมเป็นรายการเอกสาร
    search_min_score: float   # ตัดท่อนที่ความคล้ายต่ำกว่านี้ทิ้ง
    matches_per_document: int


def _resolve_model(value: str) -> str:
    """ถ้าเป็น path ในโปรเจกต์ (เช่น pretrained_models/bge-m3) ให้แปลงเป็น path เต็ม"""
    local = ROOT_DIR / value
    if local.exists():
        return str(local)
    return value


def load_settings() -> Settings:
    return Settings(
        api_key=os.getenv("API_KEY", "").strip(),
        embedding_model=_resolve_model(os.getenv("EMBEDDING_MODEL", "pretrained_models/bge-m3")),
        device=os.getenv("DEVICE", "auto"),
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", "8")),
        max_tokens=int(os.getenv("EMBED_MAX_TOKENS", "512")),
        index_dir=ROOT_DIR / os.getenv("INDEX_DIR", "data/service_index"),
        collection=os.getenv("COLLECTION", "drive-docs"),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
        search_candidates=int(os.getenv("SEARCH_CANDIDATES", "100")),
        search_min_score=float(os.getenv("SEARCH_MIN_SCORE", "0.45")),
        matches_per_document=int(os.getenv("MATCHES_PER_DOCUMENT", "3")),
    )
