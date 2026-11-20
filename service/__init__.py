"""Search Service: API สำหรับ index และค้นหาเอกสาร PDF จาก Google Drive (ผ่าน n8n)

รันจากโฟลเดอร์ราก repo:
    uvicorn service.api:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

# อ่าน .env ก่อน import อย่างอื่น เพราะค่าอย่าง HF_HUB_OFFLINE ต้องตั้งก่อนโหลด huggingface
load_dotenv(ROOT_DIR / ".env")

# ใช้ฟังก์ชันเดิมใน scripts/ (chunk.py, build_index.py, common.py) ร่วมกัน ไม่ต้องเขียนซ้ำ
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
