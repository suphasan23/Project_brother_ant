"""แปลงทุกท่อนเป็นเวกเตอร์ (embedding) แล้วเก็บลง ChromaDB เพื่อใช้ค้นหาด้วยความหมาย

แต่ละโมเดลจะถูกเก็บแยกเป็นคนละ collection (ตั้งชื่ออัตโนมัติจากชื่อโมเดล)
จึงสร้างดัชนีหลายโมเดลไว้เทียบกันได้ โดยไม่ทับกัน

ตัวอย่างการรัน:
    python scripts/build_index.py --model pretrained_models/bge-m3 --device cuda --batch-size 8
    python scripts/build_index.py --model pretrained_models/multilingual-e5-small --device cuda
    python scripts/build_index.py --model pretrained_models/bge-m3 --collection bge-m3-chunk300
    python scripts/build_index.py --help
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from common import decide_device, describe_device

DEFAULT_CHUNKS = "data/chunks/chunks.jsonl"
INDEX_DIR = Path("data/index")
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_TEST_QUERY = "ธนาคารแห่งประเทศไทยก่อตั้งเมื่อไร"

# ตัวอักษรที่ ChromaDB ยอมให้ใช้ในชื่อ collection
ALLOWED_NAME_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"


def parse_args() -> argparse.Namespace:
    """อ่านค่าที่ผู้ใช้พิมพ์ต่อท้ายคำสั่งตอนรัน"""
    parser = argparse.ArgumentParser(
        description="สร้างดัชนีเวกเตอร์จาก chunks.jsonl แล้วเก็บลง ChromaDB"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "ชื่อโมเดลบน Hugging Face (เช่น BAAI/bge-m3) "
            "หรือ path ของโฟลเดอร์โมเดลในเครื่อง (เช่น pretrained_models/bge-m3) "
            f"ค่าเริ่มต้น: {DEFAULT_MODEL}"
        ),
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "ชื่อ collection ที่จะเก็บใน ChromaDB "
            "(ถ้าไม่ระบุ จะตั้งจากชื่อโมเดล เช่น pretrained_models/bge-m3 → bge-m3)"
        ),
    )
    parser.add_argument(
        "--chunks",
        default=DEFAULT_CHUNKS,
        help=f"ไฟล์ท่อนข้อความที่จะนำมาทำดัชนี (ค่าเริ่มต้น: {DEFAULT_CHUNKS})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "อุปกรณ์ที่ใช้ประมวลผล: auto (ใช้ GPU ถ้ามี ไม่มีก็ใช้ CPU), "
            "cpu, cuda (GPU ตัวแรก) หรือ cuda:0, cuda:1 ... เพื่อเลือก GPU ตามลำดับ "
            "(ค่าเริ่มต้น: auto)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="แปลงทีละกี่ท่อน ถ้าแรมไม่พอให้ลดลง (ค่าเริ่มต้น: 16)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="ความยาวข้อความสูงสุดที่ให้โมเดลอ่าน หน่วยเป็น token (ค่าเริ่มต้น: 512)",
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help="ข้อความที่ต้องเติมหน้าคำถาม (ถ้าไม่ระบุ จะเลือกให้อัตโนมัติตามโมเดล)",
    )
    parser.add_argument(
        "--passage-prefix",
        default=None,
        help="ข้อความที่ต้องเติมหน้าเนื้อหา (ถ้าไม่ระบุ จะเลือกให้อัตโนมัติตามโมเดล)",
    )
    parser.add_argument(
        "--test-query",
        default=DEFAULT_TEST_QUERY,
        help="คำถามที่ใช้ทดสอบค้นหาหลังสร้างดัชนีเสร็จ",
    )
    return parser.parse_args()


def make_collection_name(model: str) -> str:
    """ตั้งชื่อ collection จากชื่อโมเดล

    ตัวอย่าง:
        "pretrained_models/bge-m3"          → "bge-m3"
        "BAAI/bge-m3"                        → "bge-m3"
        "intfloat/multilingual-e5-small"     → "multilingual-e5-small"
    """
    # เอาเฉพาะส่วนท้ายสุดของ path (ชื่อโฟลเดอร์ หรือชื่อโมเดลหลัง /)
    base_name = Path(model).name

    # เปลี่ยนตัวอักษรที่ ChromaDB ไม่ยอมรับให้เป็น "-"
    cleaned = ""
    for char in base_name:
        if char in ALLOWED_NAME_CHARS:
            cleaned += char
        else:
            cleaned += "-"
    return cleaned


def check_collection_name(name: str):
    """ตรวจชื่อ collection ตามกฎของ ChromaDB ถ้าไม่ผ่านจะหยุดโปรแกรมพร้อมบอกเหตุผล

    กฎ: ยาว 3-512 ตัวอักษร, ใช้ได้เฉพาะ a-z A-Z 0-9 . _ -,
        ต้องขึ้นต้นและลงท้ายด้วยตัวอักษรหรือตัวเลข
    """
    problems = []

    if len(name) < 3 or len(name) > 512:
        problems.append("ต้องยาว 3-512 ตัวอักษร")

    for char in name:
        if char not in ALLOWED_NAME_CHARS:
            problems.append(f"มีตัวอักษรที่ใช้ไม่ได้: '{char}'")
            break

    if len(name) > 0 and not name[0].isalnum():
        problems.append("ต้องขึ้นต้นด้วยตัวอักษรหรือตัวเลข")
    if len(name) > 0 and not name[-1].isalnum():
        problems.append("ต้องลงท้ายด้วยตัวอักษรหรือตัวเลข")

    if len(problems) > 0:
        print(f"✘ ชื่อ collection '{name}' ใช้ไม่ได้:")
        for problem in problems:
            print(f"  - {problem}")
        print("  ระบุชื่อเองด้วย --collection เช่น --collection my-model")
        sys.exit(1)


def decide_prefixes(model: str, query_prefix, passage_prefix) -> tuple[str, str]:
    """บางโมเดลต้องเติมคำนำหน้าข้อความถึงจะทำงานได้ดี

    - โมเดลตระกูล e5 ถูกฝึกมาให้เห็น "query: " หน้าคำถาม และ "passage: " หน้าเนื้อหา
    - bge-m3 ไม่ต้องเติมอะไร
    ถ้าผู้ใช้ระบุเองผ่าน --query-prefix / --passage-prefix จะใช้ค่าที่ระบุแทน
    """
    is_e5 = "e5" in model.lower()

    if query_prefix is None:
        if is_e5:
            query_prefix = "query: "
        else:
            query_prefix = ""

    if passage_prefix is None:
        if is_e5:
            passage_prefix = "passage: "
        else:
            passage_prefix = ""

    return query_prefix, passage_prefix


def read_chunks(path: Path) -> list[dict]:
    """อ่านไฟล์ JSONL ทีละบรรทัด แต่ละบรรทัดคือ 1 ท่อน"""
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            chunks.append(json.loads(line))
    return chunks


def text_for_embedding(chunk: dict) -> str:
    """เติมชื่อบทความและหัวข้อไว้หน้าข้อความ ให้เวกเตอร์รู้ว่าท่อนนี้พูดถึงอะไร

    ตัวอย่าง: "ธนาคารแห่งประเทศไทย > ประวัติ: ก่อตั้งขึ้นเมื่อ พ.ศ. 2485..."
    """
    if chunk["section"] == "บทนำ":
        return f"{chunk['title']}: {chunk['text']}"
    return f"{chunk['title']} > {chunk['section']}: {chunk['text']}"


def main():
    args = parse_args()
    device = decide_device(args.device)
    query_prefix, passage_prefix = decide_prefixes(
        args.model, args.query_prefix, args.passage_prefix
    )

    # ตั้งชื่อ collection: ใช้ที่ผู้ใช้ระบุ ถ้าไม่ระบุก็ตั้งจากชื่อโมเดล
    if args.collection is not None:
        collection_name = args.collection
    else:
        collection_name = make_collection_name(args.model)
    check_collection_name(collection_name)
    print(f"collection: {collection_name}")

    # ① อ่านท่อนทั้งหมด
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"✘ ไม่พบไฟล์ {chunks_path} ให้รัน python scripts\\chunk.py ก่อน")
        sys.exit(1)

    chunks = read_chunks(chunks_path)
    print(f"อ่านได้ {len(chunks)} ท่อน จาก {chunks_path}")

    # ② เตรียมข้อมูล 4 อย่างที่ ChromaDB ต้องการ (ลำดับต้องตรงกันทุก list)
    ids = []          # รหัสไม่ซ้ำของแต่ละท่อน
    documents = []    # ข้อความต้นฉบับ ไว้แสดงผลตอนค้นเจอ
    metadatas = []    # ข้อมูลที่มา
    embed_texts = []  # ข้อความที่เติมบริบทแล้ว ใช้ทำเวกเตอร์

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        metadatas.append({
            "article_id": chunk["article_id"],
            "title": chunk["title"],
            "section": chunk["section"],
            "url": chunk["url"],
        })
        embed_texts.append(passage_prefix + text_for_embedding(chunk))

    # ③ โหลดโมเดล
    if Path(args.model).exists():
        print(f"กำลังโหลดโมเดลจากโฟลเดอร์ในเครื่อง: {args.model}")
    else:
        print(f"ไม่พบโฟลเดอร์ '{args.model}' ในเครื่อง จะโหลดจาก Hugging Face แทน")
        print("(ครั้งแรกอาจใช้เวลานาน ครั้งต่อไปจะใช้ไฟล์ที่เก็บไว้)")

    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_tokens

    print(f"ใช้อุปกรณ์: {describe_device(device)}")
    if query_prefix != "" or passage_prefix != "":
        print(f"คำนำหน้า: query='{query_prefix}' passage='{passage_prefix}'")

    # ④ แปลงข้อความเป็นเวกเตอร์
    print("กำลังแปลงข้อความเป็นเวกเตอร์ ...")
    vectors = model.encode(
        embed_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,  # ปรับความยาวเวกเตอร์เป็น 1 ใช้เทียบแบบ cosine ได้
        show_progress_bar=True,
    )
    print(f"ได้เวกเตอร์ {vectors.shape[0]} ตัว ตัวละ {vectors.shape[1]} มิติ")

    # ChromaDB ต้องการ list ธรรมดา ไม่ใช่ numpy array
    embeddings = vectors.tolist()

    # ⑤ สร้าง collection ใหม่ (ลบเฉพาะ collection ชื่อนี้ ของโมเดลอื่นไม่โดนลบ)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))

    try:
        client.delete_collection(collection_name)
        print(f"ลบ collection '{collection_name}' เดิมแล้ว (กำลังสร้างใหม่)")
    except Exception:
        pass  # ยังไม่เคยสร้าง ไม่ต้องลบ

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",  # วัดความใกล้ด้วย cosine
            # จดไว้ ตอนค้นหา (search.py) ต้องใช้โมเดลและคำนำหน้าเดียวกัน
            "embedding_model": args.model,
            "query_prefix": query_prefix,
            # จดไว้ดูภายหลังว่าดัชนีนี้สร้างจากอะไร เมื่อไร
            "embedding_dim": int(vectors.shape[1]),
            "chunks_file": str(chunks_path),
            "num_chunks": len(chunks),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    # ⑥ ใส่ข้อมูลทีละชุด (ใส่ทีเดียวเยอะๆ อาจเกินขีดจำกัดของ ChromaDB)
    add_batch_size = 100
    for start in range(0, len(ids), add_batch_size):
        end = start + add_batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"เก็บลง ChromaDB แล้ว {collection.count()} รายการ ใน collection '{collection_name}' ที่ {INDEX_DIR}")

    # แสดงรายชื่อ collection ทั้งหมดที่มีในฐานข้อมูลตอนนี้
    print("\ncollection ทั้งหมดใน ChromaDB:")
    for existing in client.list_collections():
        existing_settings = existing.metadata or {}
        existing_model = existing_settings.get("embedding_model", "?")
        print(f"  - {existing.name}: {existing.count()} ท่อน (โมเดล: {existing_model})")

    # ⑦ ทดสอบค้นหา 1 คำถาม
    question = args.test_query
    print(f"\nทดสอบค้นหา: {question}")

    question_vector = model.encode(
        [query_prefix + question], normalize_embeddings=True
    ).tolist()
    results = collection.query(query_embeddings=question_vector, n_results=3)

    # results มีโครงสร้างแบบ list ซ้อน list เพราะค้นได้หลายคำถามพร้อมกัน
    # เราค้นคำถามเดียว จึงใช้ [0]
    found_documents = results["documents"][0]
    found_metadatas = results["metadatas"][0]
    found_distances = results["distances"][0]

    for i in range(len(found_documents)):
        similarity = 1 - found_distances[i]  # ChromaDB ให้ระยะห่าง (distance) = 1 - cosine
        meta = found_metadatas[i]
        print(f"\n[{i + 1}] {meta['title']} > {meta['section']}  (ความคล้าย {similarity:.3f})")
        print(f"    {found_documents[i][:150]}...")


if __name__ == "__main__":
    main()
