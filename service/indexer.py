"""เก็บและค้นหาเอกสาร

ข้อมูลอยู่ 2 ที่ (ในโฟลเดอร์ INDEX_DIR):
    - ChromaDB : ท่อนข้อความ + เวกเตอร์ ใช้ค้นหาด้วยความหมาย
    - SQLite   : ข้อมูลระดับไฟล์ 1 แถวต่อ 1 ไฟล์ (ชื่อ, path, modified_time, link ...)
                 ใช้ตอบ GET /documents, เช็คว่าไฟล์เปลี่ยนไหม และกรองตาม folder / วันที่
"""

import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from build_index import decide_prefixes
from chunk import MIN_CHARS, make_chunks, split_into_pieces
from common import decide_device, describe_device
from service.config import Settings
from service.pdf_text import extract_pages

CHROMA_ADD_BATCH = 100
SNIPPET_CHARS = 300


class NoTextError(Exception):
    """อ่านข้อความจาก PDF ไม่ได้เลย (เช่น เป็นภาพสแกนล้วน)"""


@dataclass
class DocumentMeta:
    file_id: str
    name: str
    mime_type: str
    folder_path: str
    modified_time: str
    web_view_link: str
    created_time: str = ""
    owners: str = ""
    description: str = ""


class DocumentIndex:
    def __init__(self, settings: Settings):
        self.settings = settings

        # ล็อก 2 ตัว: กันการเขียนฐานข้อมูลชนกัน และกัน GPU ทำงานซ้อนจน VRAM ไม่พอ
        self._write_lock = threading.Lock()
        self._gpu_lock = threading.Lock()

        # ① โมเดล (โหลดครั้งเดียว ใช้กับทุก request)
        self.device = decide_device(settings.device)
        self.query_prefix, self.passage_prefix = decide_prefixes(settings.embedding_model, None, None)
        print(f"กำลังโหลดโมเดล {settings.embedding_model} บน {describe_device(self.device)} ...")
        self.model = SentenceTransformer(settings.embedding_model, device=self.device)
        self.model.max_seq_length = settings.max_tokens
        self.model_name = Path(settings.embedding_model).name

        # ② ChromaDB
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.index_dir / "chroma"))
        self.collection = self.client.get_or_create_collection(
            name=settings.collection,
            metadata={"hnsw:space": "cosine", "embedding_model": self.model_name},
        )
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if stored_model != self.model_name:
            raise RuntimeError(
                f"collection '{settings.collection}' ถูกสร้างด้วยโมเดล '{stored_model}' "
                f"แต่ตอนนี้ตั้งค่าเป็น '{self.model_name}' "
                "ให้เปลี่ยน COLLECTION ใน .env เป็นชื่อใหม่ หรือลบโฟลเดอร์ index แล้ว index ใหม่"
            )

        # ③ SQLite (ข้อมูลระดับไฟล์)
        self.db = sqlite3.connect(settings.index_dir / "documents.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                file_id TEXT PRIMARY KEY,
                name TEXT, mime_type TEXT, folder_path TEXT, modified_time TEXT,
                web_view_link TEXT, created_time TEXT, owners TEXT, description TEXT,
                pages INTEGER, chunks INTEGER, indexed_at TEXT
            )"""
        )
        self.db.commit()

    # ───────────────────────── อ่านข้อมูลระดับไฟล์ ─────────────────────────

    def _get_row(self, file_id: str):
        return self.db.execute("SELECT * FROM documents WHERE file_id = ?", (file_id,)).fetchone()

    def count_documents(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def list_documents(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT file_id, name, folder_path, modified_time, chunks FROM documents ORDER BY folder_path, name"
        ).fetchall()
        return [dict(row) for row in rows]

    # ───────────────────────── เพิ่ม / แทนที่ / ลบ ─────────────────────────

    def _embed(self, texts: list[str]):
        with self._gpu_lock:
            return self.model.encode(
                texts,
                batch_size=self.settings.batch_size,
                normalize_embeddings=True,  # ต้องตรงกับตอนค้นหา
                show_progress_bar=False,
            )

    def upsert(self, meta: DocumentMeta, data: bytes) -> dict:
        """index ไฟล์ 1 ไฟล์ ถ้ามีอยู่แล้วจะแทนที่ ถ้า modified_time เท่าเดิมจะข้าม"""
        existing = self._get_row(meta.file_id)
        if existing is not None and existing["modified_time"] == meta.modified_time:
            return {"status": "unchanged", "chunks": existing["chunks"], "pages": existing["pages"]}

        pages = extract_pages(data)

        # หั่นทีละหน้า เพื่อให้แต่ละท่อนรู้เลขหน้าของตัวเอง
        ids, documents, metadatas, embed_texts = [], [], [], []
        for page_number, page_text in pages:
            chunk_number = 0
            for text in make_chunks(split_into_pieces(page_text)):
                if len(text) < MIN_CHARS:
                    continue
                ids.append(f"{meta.file_id}:{page_number}:{chunk_number}")
                documents.append(text)
                metadatas.append({"file_id": meta.file_id, "page": page_number})
                # เติม path + ชื่อไฟล์ไว้หน้าข้อความ ให้เวกเตอร์รู้ว่าเป็นเอกสารของลูกค้า/งานไหน
                embed_texts.append(
                    f"{self.passage_prefix}{meta.folder_path} > {meta.name} (หน้า {page_number}): {text}"
                )
                chunk_number += 1

        if len(ids) == 0:
            raise NoTextError("อ่านข้อความจาก PDF ไม่ได้ (อาจเป็นภาพสแกน ต้องใช้ OCR)")

        vectors = self._embed(embed_texts).tolist()

        with self._write_lock:
            self.collection.delete(where={"file_id": meta.file_id})
            for start in range(0, len(ids), CHROMA_ADD_BATCH):
                end = start + CHROMA_ADD_BATCH
                self.collection.add(
                    ids=ids[start:end],
                    embeddings=vectors[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                )
            row = asdict(meta) | {
                "pages": len(pages),
                "chunks": len(ids),
                "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            columns = ", ".join(row.keys())
            placeholders = ", ".join("?" for _ in row)
            self.db.execute(
                f"INSERT OR REPLACE INTO documents ({columns}) VALUES ({placeholders})",
                list(row.values()),
            )
            self.db.commit()

        status = "updated" if existing is not None else "indexed"
        return {"status": status, "chunks": len(ids), "pages": len(pages)}

    def delete(self, file_id: str):
        """ลบไฟล์ออกจาก index คืนจำนวนท่อนที่ลบ หรือ None ถ้าไม่มีไฟล์นี้"""
        with self._write_lock:
            existing = self._get_row(file_id)
            if existing is None:
                return None
            self.collection.delete(where={"file_id": file_id})
            self.db.execute("DELETE FROM documents WHERE file_id = ?", (file_id,))
            self.db.commit()
            return existing["chunks"]

    # ───────────────────────── ค้นหา ─────────────────────────

    def _filter_file_ids(self, folder_prefix, modified_after):
        """หา file_id ที่ผ่านเงื่อนไข folder / วันที่ (คืน None = ไม่ได้กรอง)"""
        if not folder_prefix and not modified_after:
            return None
        rows = self.db.execute("SELECT file_id, folder_path, modified_time FROM documents").fetchall()
        allowed = []
        for row in rows:
            if folder_prefix and not (row["folder_path"] or "").startswith(folder_prefix):
                continue
            # เวลาเป็น ISO 8601 ทั้งคู่ จึงเทียบแบบข้อความได้เลย
            if modified_after and (row["modified_time"] or "") < modified_after:
                continue
            allowed.append(row["file_id"])
        return allowed

    def search(self, query: str, max_documents: int, min_score: float,
               folder_prefix=None, modified_after=None) -> dict:
        started = time.perf_counter()
        results = []

        total_chunks = self.collection.count()
        allowed = self._filter_file_ids(folder_prefix, modified_after)

        if total_chunks > 0 and allowed != []:
            where = None if allowed is None else {"file_id": {"$in": allowed}}
            query_vector = self._embed([self.query_prefix + query]).tolist()
            raw = self.collection.query(
                query_embeddings=query_vector,
                n_results=min(self.settings.search_candidates, total_chunks),
                where=where,
            )
            results = self._group_by_file(raw, min_score, max_documents)

        took_ms = int((time.perf_counter() - started) * 1000)
        return {"query": query, "total": len(results), "took_ms": took_ms, "results": results}

    def _group_by_file(self, raw, min_score: float, max_documents: int) -> list[dict]:
        """รวมท่อนที่ค้นเจอเป็นรายการเอกสาร คะแนนของไฟล์ = คะแนนของท่อนที่ดีที่สุด"""
        by_file = {}
        for i in range(len(raw["ids"][0])):
            score = 1 - raw["distances"][0][i]  # ChromaDB ให้ distance = 1 - cosine
            if score < min_score:
                continue
            meta = raw["metadatas"][0][i]
            snippet = raw["documents"][0][i]
            if len(snippet) > SNIPPET_CHARS:
                snippet = snippet[:SNIPPET_CHARS] + "..."
            by_file.setdefault(meta["file_id"], []).append(
                {"page": meta["page"], "snippet": snippet, "score": round(score, 4)}
            )

        results = []
        for file_id, matches in by_file.items():
            row = self._get_row(file_id)
            if row is None:
                continue  # ถูกลบระหว่างค้นหา
            matches.sort(key=lambda m: m["score"], reverse=True)
            results.append({
                "file_id": file_id,
                "name": row["name"],
                "folder_path": row["folder_path"],
                "web_view_link": row["web_view_link"],
                "modified_time": row["modified_time"],
                "score": matches[0]["score"],
                "matches": matches[: self.settings.matches_per_document],
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:max_documents]
