import os
import re
import glob
import json
from typing import List, Dict, Tuple

import numpy as np
from pypdf import PdfReader

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "index"
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

CHUNK_MIN_LEN = 250
CHUNK_MAX_LEN = 1400


def pdf_to_text(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = re.sub(r"[ \t]+", " ", t).strip()
        if t:
            out.append((i + 1, t))
    return out


def split_into_chunks(page_no: int, text: str) -> List[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    chunks = []
    buf = ""

    for b in blocks:
        if not buf:
            buf = b
            continue
        if len(buf) < CHUNK_MIN_LEN:
            buf += "\n\n" + b
            continue
        if len(buf) > CHUNK_MAX_LEN:
            chunks.append(buf[:CHUNK_MAX_LEN])
            buf = buf[CHUNK_MAX_LEN:]
        else:
            chunks.append(buf)
            buf = b

    if buf and len(buf) >= 120:
        chunks.append(buf)

    return [f"[Sayfa {page_no}] {c}" for c in chunks if len(c) >= 120]


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("PDF bulunamadı.")
        return

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(MODEL_NAME)

    records = []
    texts = []

    for pdf in pdf_files:
        pages = pdf_to_text(pdf)
        for page_no, page_text in pages:
            chunks = split_into_chunks(page_no, page_text)
            for c in chunks:
                records.append({
                    "pdf": os.path.basename(pdf),
                    "page": page_no,
                    "text": c
                })
                texts.append(c)

        print(f"OK: {pdf}")

    vectors = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(EMB_PATH, np.asarray(vectors, dtype=np.float32))

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "model": MODEL_NAME,
        "pdf_count": len(pdf_files),
        "chunk_count": len(records)
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("İndeks oluşturuldu.")


if __name__ == "__main__":
    main()
