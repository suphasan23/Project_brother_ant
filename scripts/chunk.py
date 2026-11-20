"""หั่นบทความยาวๆ เป็นท่อนสั้นๆ (chunk) เพื่อเตรียมนำไปทำ embedding"""

import json
from pathlib import Path

INPUT_PATH = Path("data/raw/wikipedia.jsonl")
OUTPUT_PATH = Path("data/chunks/chunks.jsonl")

MAX_CHARS = 500      # ความยาวสูงสุดของแต่ละท่อน (ตัวอักษร)
OVERLAP_CHARS = 100  # ให้ท่อนถัดไปมีข้อความท้ายของท่อนก่อนหน้าซ้ำประมาณนี้
MIN_CHARS = 30       # ท่อนที่สั้นกว่านี้ถือว่าไม่มีเนื้อหา ทิ้งไป

# หัวข้อที่ไม่มีเนื้อหาเป็นประโยชน์ต่อการค้นหา (มีแต่ลิงก์หรือรายการอ้างอิง)
SKIP_SECTIONS = {
    "อ้างอิง",
    "ดูเพิ่ม",
    "แหล่งข้อมูลอื่น",
    "เชิงอรรถ",
    "บรรณานุกรม",
    "หนังสืออ่านเพิ่ม",
}


def read_articles(path: Path) -> list[dict]:
    """อ่านไฟล์ JSONL ทีละบรรทัด แต่ละบรรทัดคือ 1 บทความ"""
    articles = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            articles.append(json.loads(line))
    return articles


def split_into_sections(text: str) -> list[dict]:
    """แยกบทความตามหัวข้อ เช่น '== ประวัติ ==' และตัดหัวข้อที่อยู่ใน SKIP_SECTIONS ทิ้ง

    คืนค่าเป็น list ของ {"section": ชื่อหัวข้อ, "paragraphs": [ย่อหน้า, ...]}
    """
    sections = []
    current = {"section": "บทนำ", "paragraphs": []}
    skipping = False

    for line in text.split("\n"):
        line = line.strip()
        if line == "":
            continue

        # บรรทัดหัวข้อขึ้นต้นและลงท้ายด้วย = เช่น "== ประวัติ ==" หรือ "=== การศึกษา ==="
        is_heading = line.startswith("==") and line.endswith("==")
        if is_heading:
            # เก็บหัวข้อเดิมก่อนเริ่มหัวข้อใหม่
            if len(current["paragraphs"]) > 0:
                sections.append(current)

            level = len(line) - len(line.lstrip("="))  # นับจำนวน = ข้างหน้า
            name = line.strip("=").strip()

            # หัวข้อใหญ่ (==) เป็นตัวกำหนดว่าจะข้ามหรือไม่
            # หัวข้อย่อย (===, ====) ที่อยู่ข้างใต้ จะข้ามตามหัวข้อใหญ่ของมัน
            if level == 2:
                skipping = name in SKIP_SECTIONS

            current = {"section": name, "paragraphs": []}
            continue

        if not skipping:
            current["paragraphs"].append(line)

    # อย่าลืมหัวข้อสุดท้าย
    if len(current["paragraphs"]) > 0:
        sections.append(current)

    return sections


def split_into_pieces(paragraph: str) -> list[str]:
    """แบ่งย่อหน้าเป็นชิ้นเล็กๆ ตามช่องว่าง

    ภาษาไทยเว้นวรรคระหว่างประโยคหรือวลี จึงตัดตรงช่องว่างได้โดยไม่ทำให้คำขาด
    ถ้ามีชิ้นไหนยาวเกินไป (ไม่มีช่องว่างเลย) จะตัดตามจำนวนตัวอักษรแทน
    """
    max_piece = MAX_CHARS - OVERLAP_CHARS
    pieces = []

    for word in paragraph.split(" "):
        if word == "":
            continue
        while len(word) > max_piece:
            pieces.append(word[:max_piece])
            word = word[max_piece:]
        pieces.append(word)

    return pieces


def make_chunks(pieces: list[str]) -> list[str]:
    """รวมชิ้นเล็กๆ ให้เป็นท่อนยาวไม่เกิน MAX_CHARS โดยมีส่วนซ้อนทับ (overlap)

    ตัวอย่าง (สมมติ MAX_CHARS สั้นๆ):
        ท่อนที่ 1: A B C D
        ท่อนที่ 2:       D E F G   <- D ซ้ำกับท้ายท่อนแรก
    การซ้อนทับช่วยไม่ให้ข้อความที่อยู่ตรงรอยต่อหายไปจากบริบท
    """
    chunks = []
    current = []      # ชิ้นที่อยู่ในท่อนปัจจุบัน
    current_len = 0   # ความยาวรวมของท่อนปัจจุบัน

    for piece in pieces:
        # ถ้าใส่ชิ้นนี้แล้วจะยาวเกิน ให้ปิดท่อนนี้ แล้วเริ่มท่อนใหม่
        if current_len + len(piece) > MAX_CHARS and len(current) > 0:
            chunks.append(" ".join(current))

            # เอาชิ้นท้ายๆ ของท่อนเดิม (ไม่เกิน OVERLAP_CHARS) มาเป็นจุดเริ่มของท่อนใหม่
            overlap = []
            overlap_len = 0
            for previous in reversed(current):
                if overlap_len + len(previous) > OVERLAP_CHARS:
                    break
                overlap.insert(0, previous)
                overlap_len += len(previous) + 1  # +1 คือช่องว่างที่ใช้เชื่อม

            current = overlap
            current_len = overlap_len

        current.append(piece)
        current_len += len(piece) + 1

    if len(current) > 0:
        chunks.append(" ".join(current))

    return chunks


def chunk_article(article: dict) -> list[dict]:
    """หั่น 1 บทความเป็นหลายท่อน พร้อมแนบข้อมูลที่มา"""
    results = []
    sections = split_into_sections(article["text"])

    for section in sections:
        # รวมทุกย่อหน้าในหัวข้อเดียวกันเป็นชิ้นเล็กๆ
        # (ไม่รวมข้ามหัวข้อ เพื่อให้แต่ละท่อนพูดเรื่องเดียวกัน)
        pieces = []
        for paragraph in section["paragraphs"]:
            pieces.extend(split_into_pieces(paragraph))

        for text in make_chunks(pieces):
            if len(text) < MIN_CHARS:
                continue

            chunk_number = len(results)
            results.append({
                "chunk_id": f"{article['id']}-{chunk_number}",
                "article_id": article["id"],
                "title": article["title"],
                "section": section["section"],
                "url": article["url"],
                "text": text,
            })

    return results


def main():
    articles = read_articles(INPUT_PATH)
    print(f"อ่านได้ {len(articles)} บทความ จาก {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_chars = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for article in articles:
            chunks = chunk_article(article)

            for chunk in chunks:
                line = json.dumps(chunk, ensure_ascii=False)
                f.write(line + "\n")
                total_chars += len(chunk["text"])

            total_chunks += len(chunks)
            print(f"  {article['title']}: {len(chunks)} ท่อน")

    average = 0
    if total_chunks > 0:
        average = total_chars // total_chunks

    print(f"รวม {total_chunks} ท่อน (เฉลี่ย {average} ตัวอักษร/ท่อน)")
    print(f"บันทึกที่: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
