"""อ่านข้อความจาก PDF ทีละหน้า"""

import re

import pymupdf


class PdfError(Exception):
    """เปิด PDF ไม่ได้ (ไฟล์เสีย หรือติดรหัสผ่าน)"""


def is_pdf(data: bytes) -> bool:
    """ไฟล์ PDF ทุกไฟล์ขึ้นต้นด้วย %PDF (เช็คจากเนื้อไฟล์ ไม่เชื่อแค่ชื่อหรือ mime_type)"""
    return data[:1024].lstrip().startswith(b"%PDF")


def normalize_thai(text: str) -> str:
    """แก้รูปแบบตัวอักษรไทยที่ PDF มักเก็บผิด แล้วรวมช่องว่างให้เหลือตัวเดียว

    PDF ไทยหลายไฟล์เก็บ "ำ" เป็นนิคหิต + สระอา (ํ + า) ถ้าไม่แปลง
    คำว่า "ทำ" ในเอกสารจะไม่ตรงกับ "ทำ" ที่ผู้ใช้พิมพ์ค้น
    """
    text = text.replace("ํา", "ำ")
    return re.sub(r"\s+", " ", text).strip()


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    """คืนค่า [(เลขหน้า, ข้อความ), ...] นับหน้าจาก 1 ข้ามหน้าที่ไม่มีข้อความ

    บรรทัดใน PDF ถูกตัดตามความกว้างหน้ากระดาษ จึงรวมทุกบรรทัดเป็นข้อความเดียวต่อหน้า
    PDF ที่เป็นภาพสแกนล้วนจะไม่มีข้อความออกมา (ต้องใช้ OCR ซึ่งยังไม่รองรับ)
    """
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise PdfError(f"เปิดไฟล์ PDF ไม่ได้: {e}") from e

    with document:
        if document.needs_pass:
            raise PdfError("PDF ติดรหัสผ่าน")

        pages = []
        for page_number, page in enumerate(document, start=1):
            text = normalize_thai(page.get_text("text"))
            if text != "":
                pages.append((page_number, text))
        return pages
