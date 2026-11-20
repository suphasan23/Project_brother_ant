"""วัดผลการค้นหาด้วยชุดคำถามที่มีเฉลย (eval/questions.jsonl)

ตัวชี้วัด:
    Hit@k  = กี่เปอร์เซ็นต์ของคำถาม ที่ท่อนคำตอบอยู่ใน k อันดับแรก
    MRR    = ค่าเฉลี่ยของ 1/อันดับ (เจออันดับ 1 = 1.0, อันดับ 2 = 0.5, ไม่เจอ = 0)

ตัวอย่างการรัน:
    python scripts/evaluate_retrieval.py --collection bge-m3
    python scripts/evaluate_retrieval.py --collection bge-m3 multilingual-e5-small   # เทียบหลายตัว
    python scripts/evaluate_retrieval.py --collection bge-m3 --show-all               # ดูผลทุกข้อ
    python scripts/evaluate_retrieval.py --help
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from common import decide_device, describe_device
from search import choose_collection, open_client, search

DEFAULT_QUESTIONS = "eval/questions.jsonl"
RESULTS_DIR = Path("data/eval_results")
HIT_KS = [1, 3, 5, 10]   # จะคำนวณ Hit@k ที่ค่า k เหล่านี้
FAIL_RANK = 5            # ถือว่า "พลาด" ถ้าท่อนคำตอบไม่อยู่ใน 5 อันดับแรก


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="วัดผลการค้นหาด้วยชุดคำถามที่มีเฉลย")
    parser.add_argument(
        "--collection",
        nargs="+",
        required=True,
        help="ชื่อ collection ที่จะวัดผล ใส่ได้หลายตัวเพื่อเทียบกัน (ดูรายชื่อ: python scripts\\search.py --list)",
    )
    parser.add_argument(
        "--questions",
        default=DEFAULT_QUESTIONS,
        help=f"ไฟล์ชุดคำถาม (ค่าเริ่มต้น: {DEFAULT_QUESTIONS})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="ค้นลึกสุดกี่อันดับ (ค่าเริ่มต้น: 10)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, ... (ค่าเริ่มต้น: auto)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="แสดงผลรายข้อของทุกคำถาม (ปกติแสดงเฉพาะข้อที่พลาด)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help=f"ไม่ต้องบันทึกผลลง {RESULTS_DIR}",
    )
    return parser.parse_args()


# ───────────────────────── อ่านชุดคำถาม ─────────────────────────

def read_questions(path: Path) -> list[dict]:
    """อ่านชุดคำถาม และตรวจว่าแต่ละข้อมีฟิลด์ครบ"""
    if not path.exists():
        print(f"✘ ไม่พบไฟล์ชุดคำถาม: {path}")
        sys.exit(1)

    required_fields = ["id", "type", "question", "article_id", "answer_contains"]
    questions = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line == "":
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"✘ {path} บรรทัด {line_number}: JSON ไม่ถูกต้อง ({e})")
                sys.exit(1)

            for field in required_fields:
                if field not in item:
                    print(f"✘ {path} บรรทัด {line_number}: ไม่มีฟิลด์ '{field}'")
                    sys.exit(1)

            questions.append(item)

    return questions


# ───────────────────────── ตัดสินว่าท่อนไหน "ตอบได้" ─────────────────────────

def is_relevant(result: dict, question: dict) -> bool:
    """ท่อนนี้ตอบคำถามได้ไหม: ต้องมาจากบทความที่ถูก และมีคำใดคำหนึ่งใน answer_contains"""
    if result["article_id"] != question["article_id"]:
        return False

    for answer in question["answer_contains"]:
        if answer in result["text"]:
            return True
    return False


def find_rank(results: list[dict], question: dict):
    """หาอันดับของท่อนแรกที่ตอบได้ (นับจาก 1) ถ้าไม่เจอคืน None"""
    for index in range(len(results)):
        if is_relevant(results[index], question):
            return index + 1
    return None


# ───────────────────────── วัดผล 1 collection ─────────────────────────

def evaluate_collection(client, collection_name: str, questions: list[dict], top_k: int, device: str) -> dict:
    """รันทุกคำถามกับ collection เดียว แล้วจดผลรายข้อ"""
    collection = choose_collection(client, collection_name)
    settings = collection.metadata or {}
    model_name = settings.get("embedding_model")
    query_prefix = settings.get("query_prefix", "")

    print(f"\n{'=' * 70}")
    print(f"collection: {collection.name} ({collection.count()} ท่อน)")
    print(f"โมเดล: {model_name} บน {describe_device(device)}")
    print(f"{'=' * 70}")

    model = SentenceTransformer(model_name, device=device)

    rows = []
    for question in questions:
        results = search(collection, model, query_prefix, question["question"], top_k, None)

        rank = None
        if question["article_id"] is not None:
            rank = find_rank(results, question)

        top_score = 0.0
        if len(results) > 0:
            top_score = results[0]["score"]

        rows.append({
            "id": question["id"],
            "type": question["type"],
            "question": question["question"],
            "answerable": question["article_id"] is not None,
            "rank": rank,
            "top_score": top_score,
            "top_results": results[:3],  # เก็บ 3 อันดับแรกไว้ดูตอนวิเคราะห์ข้อที่พลาด
        })

    # คืนหน่วยความจำ GPU ก่อนโหลดโมเดลถัดไป
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "collection": collection.name,
        "embedding_model": model_name,
        "num_chunks": collection.count(),
        "rows": rows,
    }


# ───────────────────────── คำนวณตัวชี้วัด ─────────────────────────

def compute_metrics(rows: list[dict], top_k: int) -> dict:
    """คำนวณ Hit@k และ MRR จากคำถามที่มีคำตอบ"""
    answerable_rows = []
    for row in rows:
        if row["answerable"]:
            answerable_rows.append(row)

    total = len(answerable_rows)
    metrics = {"count": total}
    if total == 0:
        return metrics

    for k in HIT_KS:
        if k > top_k:
            continue
        hits = 0
        for row in answerable_rows:
            if row["rank"] is not None and row["rank"] <= k:
                hits += 1
        metrics[f"hit@{k}"] = hits / total

    reciprocal_sum = 0.0
    for row in answerable_rows:
        if row["rank"] is not None:
            reciprocal_sum += 1 / row["rank"]
    metrics["mrr"] = reciprocal_sum / total

    return metrics


def compute_metrics_by_type(rows: list[dict], top_k: int) -> dict:
    """คำนวณตัวชี้วัดแยกตามประเภทคำถาม (ไม่รวมข้อที่ไม่มีคำตอบ)"""
    rows_by_type = {}
    for row in rows:
        if not row["answerable"]:
            continue
        if row["type"] not in rows_by_type:
            rows_by_type[row["type"]] = []
        rows_by_type[row["type"]].append(row)

    metrics_by_type = {}
    for question_type in rows_by_type:
        metrics_by_type[question_type] = compute_metrics(rows_by_type[question_type], top_k)
    return metrics_by_type


def sweep_thresholds(rows: list[dict]) -> list[dict]:
    """ลองค่า min-score หลายค่า ดูว่าแต่ละค่าแยกคำถามที่มี/ไม่มีคำตอบได้ดีแค่ไหน

    ใช้คะแนนอันดับ 1 ของแต่ละคำถามเทียบกับเกณฑ์:
      - คำถามที่มีคำตอบ: ควร "ผ่าน" เกณฑ์ (คะแนน >= เกณฑ์)
      - คำถามที่ไม่มีคำตอบ: ควร "ถูกตัด" (คะแนน < เกณฑ์)
    """
    answerable_scores = []
    unanswerable_scores = []
    for row in rows:
        if row["answerable"]:
            answerable_scores.append(row["top_score"])
        else:
            unanswerable_scores.append(row["top_score"])

    table = []
    for step in range(20, 91, 5):  # 0.20, 0.25, ..., 0.90
        threshold = step / 100

        kept = 0
        for score in answerable_scores:
            if score >= threshold:
                kept += 1

        rejected = 0
        for score in unanswerable_scores:
            if score < threshold:
                rejected += 1

        kept_rate = kept / len(answerable_scores) if len(answerable_scores) > 0 else 0
        rejected_rate = rejected / len(unanswerable_scores) if len(unanswerable_scores) > 0 else 0

        table.append({
            "threshold": threshold,
            "kept": kept,
            "kept_rate": kept_rate,
            "rejected": rejected,
            "rejected_rate": rejected_rate,
            "balanced": (kept_rate + rejected_rate) / 2,  # ค่าเฉลี่ยของทั้งสองฝั่ง
        })
    return table


# ───────────────────────── แสดงผล ─────────────────────────

def format_rank(rank) -> str:
    if rank is None:
        return "ไม่เจอ"
    return f"อันดับ {rank}"


def print_report(result: dict, questions: list[dict], top_k: int, show_all: bool):
    rows = result["rows"]
    metrics = compute_metrics(rows, top_k)
    metrics_by_type = compute_metrics_by_type(rows, top_k)
    thresholds = sweep_thresholds(rows)

    # ── ภาพรวม ──
    print(f"\n── ผลการค้นหา ({metrics['count']} ข้อที่มีคำตอบ) ──")
    for k in HIT_KS:
        key = f"hit@{k}"
        if key in metrics:
            hits = round(metrics[key] * metrics["count"])
            print(f"  Hit@{k:<3} {metrics[key] * 100:6.1f}%  ({hits}/{metrics['count']})")
    print(f"  MRR     {metrics['mrr']:.3f}")

    # ── แยกตามประเภท ──
    print("\n── แยกตามประเภทคำถาม ──")
    print(f"  {'type':<12} {'ข้อ':>4} {'Hit@1':>7} {'Hit@5':>7} {'MRR':>7}")
    for question_type in metrics_by_type:
        m = metrics_by_type[question_type]
        hit1 = m.get("hit@1", 0) * 100
        hit5 = m.get("hit@5", 0) * 100
        print(f"  {question_type:<12} {m['count']:>4} {hit1:>6.0f}% {hit5:>6.0f}% {m['mrr']:>7.3f}")

    # ── คะแนนอันดับ 1: มีคำตอบ vs ไม่มีคำตอบ ──
    answerable_scores = []
    unanswerable_scores = []
    for row in rows:
        if row["answerable"]:
            answerable_scores.append(row["top_score"])
        else:
            unanswerable_scores.append(row["top_score"])

    print("\n── คะแนนอันดับ 1 ──")
    if len(answerable_scores) > 0:
        print(f"  มีคำตอบ     ต่ำสุด {min(answerable_scores):.3f}  สูงสุด {max(answerable_scores):.3f}")
    if len(unanswerable_scores) > 0:
        print(f"  ไม่มีคำตอบ  ต่ำสุด {min(unanswerable_scores):.3f}  สูงสุด {max(unanswerable_scores):.3f}")

    # ── ตาราง min-score ──
    if len(unanswerable_scores) > 0:
        best = thresholds[0]
        for row in thresholds:
            if row["balanced"] > best["balanced"]:
                best = row

        print("\n── หาค่า min-score ──")
        print(f"  {'min-score':>9}  {'มีคำตอบผ่าน':>12}  {'ไม่มีคำตอบถูกตัด':>16}")
        for row in thresholds:
            marker = "  ← สมดุลที่สุด" if row is best else ""
            print(f"  {row['threshold']:>9.2f}  {row['kept_rate'] * 100:>11.0f}%  {row['rejected_rate'] * 100:>15.0f}%{marker}")

    # ── รายข้อ ──
    if show_all:
        print("\n── ผลรายข้อ ──")
        for row in rows:
            if row["answerable"]:
                status = format_rank(row["rank"])
            else:
                status = "(ไม่มีคำตอบ)"
            print(f"  {row['id']} [{row['type']:<12}] {status:<10} top={row['top_score']:.3f}  {row['question']}")

    # ── ข้อที่พลาด ──
    questions_by_id = {}
    for question in questions:
        questions_by_id[question["id"]] = question

    failed_rows = []
    for row in rows:
        if row["answerable"] and (row["rank"] is None or row["rank"] > FAIL_RANK):
            failed_rows.append(row)

    print(f"\n── ข้อที่พลาด (ท่อนคำตอบไม่อยู่ใน top-{FAIL_RANK}): {len(failed_rows)} ข้อ ──")
    for row in failed_rows:
        question = questions_by_id[row["id"]]
        print(f"\n  {row['id']} [{row['type']}] \"{row['question']}\"  → {format_rank(row['rank'])}")
        print(f"    ต้องการ: บทความ {question['article_id']} ที่มี {question['answer_contains']}")
        print("    ค้นได้:")
        for i in range(len(row["top_results"])):
            found = row["top_results"][i]
            snippet = found["text"][:80].replace("\n", " ")
            print(f"      [{i + 1}] {found['score']:.3f} {found['title']} > {found['section']}: {snippet}...")

    return metrics, metrics_by_type, thresholds


def print_comparison(summaries: list[dict]):
    """ตารางเทียบหลาย collection"""
    print(f"\n{'=' * 70}")
    print("สรุปเทียบ")
    print(f"{'=' * 70}")
    print(f"  {'collection':<28} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>7}")
    for summary in summaries:
        m = summary["metrics"]
        line = f"  {summary['collection']:<28}"
        for k in HIT_KS:
            key = f"hit@{k}"
            if key in m:
                line += f" {m[key] * 100:>6.1f}%"
            else:
                line += f" {'-':>7}"
        line += f" {m['mrr']:>7.3f}"
        print(line)


# ───────────────────────── main ─────────────────────────

def main():
    args = parse_args()
    questions = read_questions(Path(args.questions))

    answerable_count = 0
    for question in questions:
        if question["article_id"] is not None:
            answerable_count += 1

    print(f"ชุดคำถาม: {args.questions} ({len(questions)} ข้อ: มีคำตอบ {answerable_count}, ไม่มีคำตอบ {len(questions) - answerable_count})")

    client = open_client()
    device = decide_device(args.device)

    summaries = []
    for collection_name in args.collection:
        result = evaluate_collection(client, collection_name, questions, args.top_k, device)
        metrics, metrics_by_type, thresholds = print_report(result, questions, args.top_k, args.show_all)

        summary = {
            "collection": result["collection"],
            "embedding_model": result["embedding_model"],
            "num_chunks": result["num_chunks"],
            "questions_file": args.questions,
            "top_k": args.top_k,
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "metrics_by_type": metrics_by_type,
            "thresholds": thresholds,
            "rows": result["rows"],
        }
        summaries.append(summary)

        if not args.no_save:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = RESULTS_DIR / f"{result['collection']}_{timestamp}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\nบันทึกผลที่: {out_path}")

    if len(summaries) > 1:
        print_comparison(summaries)


if __name__ == "__main__":
    main()
