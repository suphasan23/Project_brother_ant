"""ฟังก์ชันที่ใช้ร่วมกันหลายสคริปต์ (แก้ที่นี่ที่เดียว มีผลทุกไฟล์)"""

import sys

import torch


def decide_device(requested: str) -> str:
    """ตรวจและเลือกอุปกรณ์ตามที่ผู้ใช้ขอ ถ้าขอ GPU แต่ใช้ไม่ได้ จะหยุดโปรแกรมพร้อมบอกเหตุผล

    ค่าที่รับได้: auto, cpu, cuda, cuda:0, cuda:1, ...
    """
    requested = requested.lower()
    gpu_available = torch.cuda.is_available()

    if requested == "auto":
        if gpu_available:
            return "cuda:0"
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested.startswith("cuda"):
        if not gpu_available:
            print("✘ ขอใช้ GPU แต่ PyTorch มองไม่เห็น GPU")
            print(f"  PyTorch เวอร์ชัน: {torch.__version__}")
            print("  ถ้าเวอร์ชันลงท้ายด้วย +cpu ต้องติดตั้ง PyTorch เวอร์ชัน CUDA ก่อน")
            print("  หรือรันด้วย --device cpu แทน")
            sys.exit(1)

        # "cuda" เฉยๆ หมายถึง GPU ตัวแรก
        if requested == "cuda":
            return "cuda:0"

        # "cuda:1" → เช็กว่ามี GPU ตัวที่ 1 จริงไหม (นับจาก 0)
        index_text = requested.split(":")[1]
        if not index_text.isdigit():
            print(f"✘ รูปแบบ --device ไม่ถูกต้อง: '{requested}' (ตัวอย่างที่ถูก: cuda:0)")
            sys.exit(1)

        index = int(index_text)
        gpu_count = torch.cuda.device_count()
        if index >= gpu_count:
            print(f"✘ ไม่มี GPU ตัวที่ {index} (เครื่องนี้มี GPU {gpu_count} ตัว: cuda:0 ถึง cuda:{gpu_count - 1})")
            sys.exit(1)
        return requested

    print(f"✘ ไม่รู้จัก --device '{requested}' (ใช้ได้: auto, cpu, cuda, cuda:0, cuda:1, ...)")
    sys.exit(1)


def describe_device(device: str) -> str:
    """คืนข้อความอธิบายอุปกรณ์ เช่น 'cuda:0 (NVIDIA GeForce RTX 3050 Ti Laptop GPU)'"""
    if device == "cpu":
        return "cpu"
    return f"{device} ({torch.cuda.get_device_name(device)})"
