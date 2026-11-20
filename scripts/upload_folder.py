"""ส่ง PDF ทุกไฟล์ในโฟลเดอร์ในเครื่องเข้า Search Service (ใช้ทดสอบแทน n8n)

folder_path ของแต่ละไฟล์ = path นับจากโฟลเดอร์ที่ระบุ เช่น
    data/mock/ลูกค้า/ลูกค้า A/งาน A/mom.pdf  →  /ลูกค้า/ลูกค้า A/งาน A
file_id สร้างจาก path ของไฟล์ (ไฟล์เดิมได้ id เดิมทุกครั้ง)

ตัวอย่างการรัน (ต้องเปิด service ไว้ก่อน):
    python scripts/upload_folder.py data/mock
    python scripts/upload_folder.py data/mock --url http://192.168.1.50:8000/api/v1
    python scripts/upload_folder.py --help
"""

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ส่ง PDF ในโฟลเดอร์เข้า Search Service")
    parser.add_argument("folder", help="โฟลเดอร์ที่มี PDF (ค้นทุกโฟลเดอร์ย่อย)")
    parser.add_argument("--url", default="http://localhost:8000/api/v1", help="base URL ของ service")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="ค่าเริ่มต้นอ่านจาก .env")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.folder)
    if not root.is_dir():
        print(f"✘ ไม่พบโฟลเดอร์ {root}")
        sys.exit(1)

    pdfs = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".pdf")
    print(f"พบ {len(pdfs)} ไฟล์ PDF ใน {root}")
    headers = {"X-API-Key": args.api_key}

    for pdf in pdfs:
        relative = pdf.relative_to(root)
        folder_path = "/" + relative.parent.as_posix() if relative.parent != Path(".") else "/"
        file_id = "local-" + hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:16]
        modified = datetime.fromtimestamp(pdf.stat().st_mtime, tz=timezone.utc)

        form = {
            "file_id": file_id,
            "name": pdf.name,
            "mime_type": "application/pdf",
            "folder_path": folder_path,
            "modified_time": modified.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "web_view_link": pdf.resolve().as_uri(),
        }
        with pdf.open("rb") as f:
            response = requests.post(
                f"{args.url}/documents", headers=headers, data=form,
                files={"file": (pdf.name, f, "application/pdf")}, timeout=600,
            )

        if response.ok:
            body = response.json()
            print(f"  ✔ {relative}  {body['status']}  {body['chunks']} ท่อน  {body['took_ms']} ms")
        else:
            print(f"  ✘ {relative}  HTTP {response.status_code}  {response.text}")
            if response.status_code == 401:
                print("    API key ไม่ตรงกับ .env ของ service")
                sys.exit(1)


if __name__ == "__main__":
    main()
