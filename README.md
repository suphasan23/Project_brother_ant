# mini-search

ระบบค้นหาเอกสารใน Google Drive ด้วยความหมาย พิมพ์คำถามเช่น "เอกสารที่คุยกับลูกค้า A เรื่องงาน A" แล้วได้**รายการเอกสารที่เกี่ยวข้องทั้งหมด** พร้อมลิงก์ Drive และข้อความช่วงที่ตรงกับคำถาม

โมเดลและฐานข้อมูลทั้งหมดรันบนเครื่องเรา ส่วนที่ต่อออกข้างนอกมีแค่ Google Drive เนื้อหาเอกสารจึงไม่ถูกส่งไปที่บริการ AI ภายนอก

> **สถานะ:** Search Service (FastAPI, โฟลเดอร์ `service/`) รับ PDF และค้นหาได้แล้ว ทดสอบกับ mock data ในเครื่อง ยังไม่ได้ต่อกับ n8n จริง ดู [แผนงาน](#แผนงาน)

## ภาพรวมระบบ

```
[เครื่องเพื่อน: n8n]                           [เครื่องผม: Search Service]
Google Drive (OAuth)
  → หาไฟล์ PDF ใหม่ / ถูกแก้ / ถูกลบ
  → ดาวน์โหลด PDF
  → POST   /documents ───────────────────►  แยกข้อความ → chunk → embed (bge-m3)
  → DELETE /documents/{file_id} ─────────►  → เก็บ / ลบใน ChromaDB
                                                     ▲
Chat Trigger / หน้าเว็บ                               │
  → POST /search ────────────────────────►  ค้นหา → รวมผลตามไฟล์
  ◄──────── รายการเอกสาร + link + snippet ───────────┘
```

| ส่วน | ใครดูแล | หน้าที่ |
| --- | --- | --- |
| n8n workflow | เพื่อน | เชื่อม Google Drive, sync ไฟล์, หน้าแชท |
| Search Service (repo นี้) | ผม | index เอกสาร, ค้นหา, (ภายหลัง) ตอบคำถามแบบ RAG |

Search Service ไม่ต้องมี credential ของ Google Drive และไม่ต้องใช้ API key ของ AI เจ้าไหนเลย

## โครงสร้าง repo

```
service/                 Search Service (API ที่ n8n เรียก)
  api.py                 endpoint ทั้งหมด, ตรวจ API key, รูปแบบ error
  indexer.py             index / ลบ / ค้นหา (ChromaDB + SQLite) รวมผลเป็นรายการเอกสาร
  pdf_text.py            อ่านข้อความจาก PDF รายหน้า
  config.py              อ่านค่าตั้งค่าจาก .env
scripts/
  upload_folder.py       ส่ง PDF ในโฟลเดอร์ในเครื่องเข้า service (ทดสอบแทน n8n)
  common.py              เลือก CPU/GPU ใช้ร่วมกันทุกสคริปต์
  fetch_wikipedia.py     ดึงบทความ Wikipedia มาเป็นข้อมูลทดสอบ → data/raw/wikipedia.jsonl
  chunk.py               หั่นบทความเป็นท่อน (500 ตัวอักษร, overlap 100) → data/chunks/chunks.jsonl
  build_index.py         embed ทุกท่อนแล้วเก็บลง ChromaDB → data/index/
  search.py              ค้นหาจาก command line
  evaluate_retrieval.py  วัดผลการค้นหา (Hit@k, MRR) ด้วยชุดคำถามที่มีเฉลย
requirements.txt
.env.example             ตัวอย่างค่าตั้งค่า (คัดลอกเป็น .env)
```

ไฟล์ข้อมูล (`data/`), โมเดล (`pretrained_models/`), ชุดคำถาม (`eval/`) และ `.env` ไม่ถูก commit ขึ้น git

## ติดตั้ง

ต้องมี Python 3.10 ขึ้นไป

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate

# มี GPU (NVIDIA): ติดตั้ง PyTorch แบบ CUDA ก่อน ไม่งั้นจะได้แบบ CPU (ช้ากว่าแต่ใช้ได้)
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

# ดาวน์โหลดโมเดลมาไว้ในโปรเจกต์ (~2.3 GB)
hf download BAAI/bge-m3 --local-dir pretrained_models/bge-m3
```

## รัน Search Service

```powershell
Copy-Item .env.example .env      # แล้วแก้ API_KEY ใน .env
uvicorn service.api:app --host 0.0.0.0 --port 8000
```

เปิด `http://localhost:8000/docs` เพื่อลองเรียกผ่าน Swagger ในแต่ละ endpoint ให้ใส่ค่า `API_KEY` ในช่อง `x-api-key`

ทดสอบด้วย PDF ในเครื่องแทน n8n: วาง PDF ไว้ในโครง `data/mock/ลูกค้า/<ชื่อลูกค้า>/<ชื่องาน>/` แล้วรัน (เปิด service ค้างไว้อีกหน้าต่าง)

```powershell
python scripts/upload_folder.py data/mock
```

`folder_path` จะได้จากโครงโฟลเดอร์ เช่น `/ลูกค้า/ลูกค้า A/งาน A` ส่งไฟล์เดิมซ้ำจะได้ `unchanged`

ค่าตั้งค่าทั้งหมดอยู่ใน `.env.example` ที่ใช้บ่อย:

| ค่า | ค่าเริ่มต้น | ความหมาย |
| --- | --- | --- |
| `API_KEY` | (ต้องตั้ง) | รหัสที่ n8n ส่งมาใน header `X-API-Key` |
| `EMBEDDING_MODEL` | `pretrained_models/bge-m3` | path ในโปรเจกต์ หรือชื่อบน Hugging Face |
| `DEVICE` | `auto` | `cpu`, `cuda` |
| `EMBED_BATCH_SIZE` | `8` | ลดลงถ้าเจอ CUDA out of memory |
| `SEARCH_MIN_SCORE` | `0.45` | ตัดผลที่คะแนนต่ำกว่านี้ (ปรับหลังวัดผลกับเอกสารจริง) |
| `HF_HUB_OFFLINE` | `1` | ห้ามต่อ Hugging Face ใช้โมเดลในเครื่องเท่านั้น |

index ถูกเก็บที่ `data/service_index/` ถ้าเปลี่ยนโมเดล service จะไม่ยอมเปิด ให้ลบโฟลเดอร์นี้หรือเปลี่ยน `COLLECTION` แล้ว index ใหม่

## pipeline ต้นแบบ (Wikipedia)

ใช้ทดลองเทียบโมเดลและวัดผล แยกจาก Search Service

รันตามลำดับ:

```bash
# 1. ดึงข้อมูลทดสอบ
python scripts/fetch_wikipedia.py --topics ปัญญาประดิษฐ์ ประเทศไทย --limit 20

# 2. หั่นเป็นท่อน
python scripts/chunk.py

# 3. สร้างดัชนี (ครั้งแรกจะโหลดโมเดลจาก Hugging Face)
python scripts/build_index.py --model BAAI/bge-m3 --device auto --batch-size 8

# 4. ค้นหา
python scripts/search.py --collection bge-m3 --query "แบงก์ชาติมีหน้าที่อะไร"
python scripts/search.py --collection bge-m3            # โหมดถามวน

# 5. วัดผล (ต้องมี eval/questions.jsonl)
python scripts/evaluate_retrieval.py --collection bge-m3
```

ทุกสคริปต์มี `--help` ดูตัวเลือกทั้งหมดได้

แต่ละโมเดลถูกเก็บเป็นคนละ collection จึงเทียบหลายโมเดลได้ เช่น

```bash
python scripts/build_index.py --model intfloat/multilingual-e5-small
python scripts/evaluate_retrieval.py --collection bge-m3 multilingual-e5-small
```

### ชุดคำถามสำหรับวัดผล

`eval/questions.jsonl` หนึ่งบรรทัดต่อหนึ่งคำถาม:

```json
{"id": "q1", "type": "fact", "question": "ธนาคารแห่งประเทศไทยก่อตั้งเมื่อไร", "article_id": "12345", "answer_contains": ["2485"]}
```

ถือว่าค้นเจอเมื่อท่อนที่ได้มาจาก `article_id` ที่ถูกต้อง และมีคำใดคำหนึ่งใน `answer_contains`

## API (Search Service)

> รายละเอียดเต็มอยู่ในเอกสาร API Contract: n8n ↔ Search Service ส่วนนี้เป็นสรุป

- **Base URL:** `http://<IP เครื่องที่รัน service>:8000/api/v1`
- **การยืนยันตัวตน:** header `X-API-Key: <key>` ทุก request ยกเว้น `/health`
- **Swagger:** `http://<host>:8000/docs`

| Method | Path | ทำอะไร |
| --- | --- | --- |
| POST | `/documents` | เพิ่มหรืออัปเดต PDF 1 ไฟล์ (`multipart/form-data`) ถ้า `modified_time` เท่าเดิมจะข้าม |
| GET | `/documents` | รายการไฟล์ที่ index แล้ว ให้ n8n เทียบกับ Drive |
| DELETE | `/documents/{file_id}` | ลบไฟล์ออกจาก index |
| POST | `/search` | ค้นหา คืนรายการเอกสารเรียงตามความเกี่ยวข้อง |
| GET | `/health` | เช็คว่าพร้อมใช้งาน |

Field ที่บังคับใน `POST /documents`: `file`, `file_id`, `name`, `mime_type`, `folder_path`, `modified_time`, `web_view_link`

ตัวอย่าง:

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "X-API-Key: <API_KEY>" \
  -F "file=@mom.pdf" -F "file_id=1AbCxyz" -F "name=MoM ลูกค้า A.pdf" \
  -F "mime_type=application/pdf" -F "folder_path=/ลูกค้า/ลูกค้า A/งาน A" \
  -F "modified_time=2026-11-18T09:15:22.000Z" \
  -F "web_view_link=https://drive.google.com/file/d/1AbCxyz/view"

curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: <API_KEY>" -H "Content-Type: application/json" \
  -d '{"query": "เอกสารที่คุยกับลูกค้า A เรื่องงาน A"}'
```

### รันให้เครื่องอื่นเรียกได้

n8n อยู่คนละเครื่องกับ service จึงต้องตั้งค่าเพิ่ม:

1. รัน service ด้วย `--host 0.0.0.0` (ไม่ใช่ `127.0.0.1`)
2. เปิด port 8000 ใน Windows Firewall
3. ถ้าอยู่คนละเครือข่าย ใช้ [Tailscale](https://tailscale.com/) ทั้งสองเครื่อง แทนการเปิด port ออก internet
4. ให้เพื่อนเปิด `http://<IP>:8000/api/v1/health` จากเครื่องเขา ต้องได้ `{"status":"ok"}`

## ข้อตกลงระยะแรก

| เรื่อง | ตอนนี้ | ภายหลัง |
| --- | --- | --- |
| ข้อมูล | mock โครงโฟลเดอร์ `/ลูกค้า/<ชื่อลูกค้า>/<ชื่องาน>/` | ปรับวิธีแยกชื่อลูกค้าและชื่องานเมื่ออีกทีมส่ง mock data มา |
| ที่รัน | service เครื่องผม, n8n เครื่องเพื่อน | ย้ายไป server เดียวกันด้วย Docker Compose |
| ชนิดไฟล์ | PDF เท่านั้น | Google Docs (export เป็น PDF ใน n8n), OCR สำหรับ PDF ที่เป็นภาพสแกน |
| สิทธิ์ | ทุกคนค้นเจอทุกไฟล์ | เพิ่ม `allowed_users` ตอน ingest และ `user_email` ตอนค้น (ไม่บังคับ) |
| Sync | ทุก 30 นาที | ปรับตามจำนวนไฟล์จริง |
| RAG chatbot | เก็บ node เดิมใน n8n ไว้ ยังไม่ใช้ | เพิ่ม `POST /ask` ใช้ LLM local (Ollama) จาก index เดียวกัน |

## แผนงาน

- [x] pipeline ต้นแบบ: chunk → embed → ChromaDB → search → evaluate
- [x] แยกข้อความจาก PDF รายหน้า (PyMuPDF) แก้สระอำที่ PDF เก็บผิด
- [x] เก็บ metadata: ต่อท่อน `file_id`, `page` / ต่อไฟล์ใน SQLite `name`, `folder_path`, `modified_time`, `web_view_link`
- [x] index ทีละไฟล์แบบ upsert ข้ามไฟล์ที่ `modified_time` เท่าเดิม
- [x] ค้นหาคืนเป็นรายการเอกสาร (รวมผลตาม `file_id`) กรองตาม `folder_prefix`, `modified_after` ได้
- [x] FastAPI ครบ 5 endpoint ตาม contract โหลดโมเดลครั้งเดียวตอนเริ่ม
- [ ] ต่อกับ n8n ของเพื่อนข้ามเครื่อง
- [ ] เปลี่ยน mock เป็นข้อมูลจากอีกทีม แล้วปรับการแยกชื่อลูกค้า / งาน
- [ ] hybrid search: BM25 ตัดคำด้วย PyThaiNLP ร่วมกับ vector เพื่อให้ค้นชื่อลูกค้าหรือรหัสงานเจอ
- [ ] ชุดคำถาม eval จากเอกสารจริง วัดผลระดับไฟล์
- [ ] `POST /ask` (RAG) ด้วย LLM local
- [ ] สิทธิ์การเข้าถึงตาม Drive permissions
