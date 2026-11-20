"""ดึงบทความจาก Wikipedia ตามหัวข้อที่กำหนด แล้วบันทึกเป็นไฟล์ JSONL

ตัวอย่างการรัน:
    python scripts/fetch_wikipedia.py
    python scripts/fetch_wikipedia.py --topics ปัญญาประดิษฐ์ ประเทศไทย --limit 20
    python scripts/fetch_wikipedia.py --topics-file topics.txt --limit 30
    python scripts/fetch_wikipedia.py --topics เศรษฐกิจ --limit 20 --append
    python scripts/fetch_wikipedia.py --lang en --topics "Artificial intelligence" --limit 10
    python scripts/fetch_wikipedia.py --help
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

DEFAULT_TOPICS = ["ปัญญาประดิษฐ์", "การเรียนรู้ของเครื่อง", "ประเทศไทย"]
DEFAULT_OUTPUT = "data/raw/wikipedia.jsonl"
MAX_LIMIT = 500   # Wikipedia ให้ผลค้นหาได้สูงสุด 500 รายการต่อ 1 คำค้น
MAX_RETRIES = 3   # ลองใหม่กี่ครั้งเมื่อโดน 429 (ยิงถี่เกินไป)

HEADERS = {
    # Wikimedia บังคับให้ระบุ User-Agent ถ้าไม่ใส่อาจโดนบล็อก
    "User-Agent": "mini-search/0.1 (learning project)"
}

# ใช้ Session เพื่อใช้การเชื่อมต่อเดิมซ้ำ ทำให้ยิงหลายคำขอได้เร็วขึ้น
session = requests.Session()
session.headers.update(HEADERS)


def parse_args() -> argparse.Namespace:
    """อ่านค่าที่ผู้ใช้พิมพ์ต่อท้ายคำสั่งตอนรัน"""
    parser = argparse.ArgumentParser(description="ดึงบทความจาก Wikipedia แล้วบันทึกเป็น JSONL")
    parser.add_argument(
        "--topics",
        nargs="+",  # รับได้หลายค่า คั่นด้วยช่องว่าง
        default=None,
        help="หัวข้อที่จะค้น ใส่ได้หลายหัวข้อ คั่นด้วยช่องว่าง "
             "(หัวข้อที่มีช่องว่างในตัวให้ครอบด้วย \"...\")",
    )
    parser.add_argument(
        "--topics-file",
        default=None,
        help="ไฟล์ข้อความที่มีหัวข้อบรรทัดละ 1 หัวข้อ (บรรทัดที่ขึ้นต้นด้วย # จะถูกข้าม)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=f"จำนวนบทความสูงสุดต่อ 1 หัวข้อ (1-{MAX_LIMIT}, ค่าเริ่มต้น: 5)",
    )
    parser.add_argument(
        "--lang",
        default="th",
        help="ภาษาของ Wikipedia เช่น th, en, ja (ค่าเริ่มต้น: th)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"ไฟล์ที่จะบันทึก (ค่าเริ่มต้น: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--append",
        action="store_true",  # ใส่ --append = True, ไม่ใส่ = False
        help="เขียนต่อท้ายไฟล์เดิมแทนการเขียนทับ (ข้ามบทความที่มีอยู่แล้ว)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="เวลาพักหลังทุกคำขอ หน่วยวินาที (ค่าเริ่มต้น: 1.0)",
    )
    return parser.parse_args()


def read_topics_file(path: Path) -> list[str]:
    """อ่านหัวข้อจากไฟล์ บรรทัดละ 1 หัวข้อ ข้ามบรรทัดว่างและบรรทัดที่ขึ้นต้นด้วย #"""
    topics = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            topics.append(line)
    return topics


def collect_topics(args: argparse.Namespace) -> list[str]:
    """รวมหัวข้อจาก --topics และ --topics-file แล้วตัดตัวซ้ำออก"""
    all_topics = []

    if args.topics is not None:
        all_topics.extend(args.topics)

    if args.topics_file is not None:
        topics_path = Path(args.topics_file)
        if not topics_path.exists():
            print(f"✘ ไม่พบไฟล์หัวข้อ: {topics_path}")
            sys.exit(1)
        all_topics.extend(read_topics_file(topics_path))

    # ไม่ได้ระบุหัวข้อมาเลย → ใช้ค่าเริ่มต้น
    if len(all_topics) == 0:
        all_topics = DEFAULT_TOPICS

    # ตัดหัวข้อซ้ำ โดยยังรักษาลำดับเดิมไว้
    unique_topics = []
    for topic in all_topics:
        if topic not in unique_topics:
            unique_topics.append(topic)
    return unique_topics


def read_existing(path: Path) -> tuple[set, set]:
    """อ่านไฟล์เดิม (ใช้ตอน --append) คืนค่า id และชื่อบทความที่มีอยู่แล้ว"""
    ids = set()
    titles = set()
    if not path.exists():
        return ids, titles

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            article = json.loads(line)
            ids.add(article["id"])
            titles.add(article["title"])
    return ids, titles


def call_api(api_url: str, params: dict, delay: float) -> dict:
    """ยิงคำขอไปที่ API ถ้าโดน 429 จะรอแล้วลองใหม่"""
    for attempt in range(1, MAX_RETRIES + 1):
        response = session.get(api_url, params=params, timeout=30)
        time.sleep(delay)

        if response.status_code == 429:
            # server อาจบอกมาว่าให้รอกี่วินาที (Retry-After) ถ้าไม่บอกก็รอนานขึ้นทุกรอบ
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = 5 * attempt
            print(f"  ... โดนจำกัดความเร็ว รอ {wait} วินาที (ครั้งที่ {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()

    # ลองครบทุกรอบแล้วยังโดน 429
    response.raise_for_status()
    return response.json()


def search_titles(api_url: str, query: str, limit: int, delay: float) -> list[str]:
    """ค้นหาชื่อบทความจากคำค้น"""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    data = call_api(api_url, params, delay)
    results = data["query"]["search"]

    titles = []
    for item in results:
        titles.append(item["title"])
    return titles


def get_article(api_url: str, title: str, delay: float) -> dict | None:
    """ดึงเนื้อหาบทความเป็นข้อความล้วน (ไม่มี HTML)"""
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,  # ได้ plain text
        "inprop": "url",   # ได้ URL ของบทความ
        "titles": title,
        "redirects": 1,
        "format": "json",
    }
    data = call_api(api_url, params, delay)
    pages = data["query"]["pages"]

    # pages ใช้ pageid เป็น key ซึ่งเราไม่รู้ล่วงหน้า
    # เลยแปลงค่าทั้งหมดเป็น list แล้วหยิบตัวแรก (เราขอแค่บทความเดียว)
    page_list = list(pages.values())
    page = page_list[0]

    if "missing" in page:
        return None

    return {
        "id": page["pageid"],
        "title": page["title"],
        "url": page["fullurl"],
        "text": page.get("extract", ""),
    }


def main():
    args = parse_args()

    if args.limit < 1 or args.limit > MAX_LIMIT:
        print(f"✘ --limit ต้องอยู่ระหว่าง 1 ถึง {MAX_LIMIT} (ได้รับ {args.limit})")
        sys.exit(1)

    api_url = f"https://{args.lang}.wikipedia.org/w/api.php"
    topics = collect_topics(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # เก็บทั้ง id และชื่อ กันบทความซ้ำ
    # (ต้องเช็ก id ด้วย เพราะชื่อต่างกันอาจ redirect ไปบทความเดียวกัน)
    seen_ids = set()
    seen_titles = set()
    file_mode = "w"  # เขียนทับ

    if args.append:
        seen_ids, seen_titles = read_existing(out_path)
        file_mode = "a"  # เขียนต่อท้าย
        print(f"โหมดต่อท้าย: มีอยู่แล้ว {len(seen_ids)} บทความใน {out_path}")

    print(f"Wikipedia ภาษา: {args.lang}")
    print(f"หัวข้อ {len(topics)} หัวข้อ, สูงสุดหัวข้อละ {args.limit} บทความ")
    print(f"(ประมาณการสูงสุด {len(topics) * args.limit} บทความ, อย่างน้อย ~{len(topics) * args.limit * args.delay:.0f} วินาที)")

    saved_count = 0
    skipped_count = 0

    with out_path.open(file_mode, encoding="utf-8") as f:
        for topic_number, topic in enumerate(topics, start=1):
            print(f"\n[{topic_number}/{len(topics)}] ค้นหา: {topic}")

            try:
                titles = search_titles(api_url, topic, args.limit, args.delay)
            except requests.RequestException as e:
                print(f"  ✘ ค้นหา '{topic}' ไม่สำเร็จ: {e}")
                continue

            if len(titles) == 0:
                print(f"  - ไม่พบบทความสำหรับ '{topic}'")
                continue

            for title in titles:
                if title in seen_titles:
                    skipped_count += 1
                    continue
                seen_titles.add(title)

                try:
                    article = get_article(api_url, title, args.delay)
                except requests.RequestException as e:
                    print(f"  ✘ ดึง '{title}' ไม่สำเร็จ: {e}")
                    continue

                if article is None or article["text"] == "":
                    print(f"  - ข้าม '{title}' (ไม่มีเนื้อหา)")
                    continue

                if article["id"] in seen_ids:
                    skipped_count += 1
                    continue
                seen_ids.add(article["id"])

                line = json.dumps(article, ensure_ascii=False)
                f.write(line + "\n")
                f.flush()  # เขียนลงดิสก์ทันที ถ้าโปรแกรมหยุดกลางทาง ข้อมูลที่ได้แล้วไม่หาย
                saved_count += 1
                print(f"  ✔ {article['title']} ({len(article['text'])} chars)")

    print(f"\nบันทึกใหม่ {saved_count} บทความ (ข้ามที่ซ้ำ {skipped_count}) ที่: {out_path}")
    if args.append:
        print(f"รวมทั้งไฟล์: {len(seen_ids)} บทความ")


if __name__ == "__main__":
    main()
