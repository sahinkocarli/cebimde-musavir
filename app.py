import os
import json
from typing import List, Dict, Tuple

import numpy as np
import streamlit as st

APP_TITLE = "Cebimde Müşavir – Profesyonel Mevzuat Analizi"
INDEX_DIR = "index"
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
META_PATH = os.path.join(INDEX_DIR, "meta.json")
DEFAULT_TOPK = 8


@st.cache_data
def load_meta() -> Dict:
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_chunks() -> List[Dict]:
    if not os.path.exists(CHUNKS_PATH):
        return []
    out = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@st.cache_resource
def load_embeddings() -> np.ndarray:
    if not os.path.exists(EMB_PATH):
        return np.zeros((0, 1), dtype=np.float32)
    vecs = np.load(EMB_PATH)
    return np.asarray(vecs, dtype=np.float32)


@st.cache_resource
def get_embedder():
    from sentence_transformers import SentenceTransformer
    model = load_meta().get("model", "sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(model)


def cosine_topk(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if doc_vecs.size == 0:
        return []
    sims = doc_vecs @ query_vec
    k = max(1, min(k, sims.shape[0]))
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    top = sorted([(int(i), float(sims[i])) for i in idx], key=lambda x: x[1], reverse=True)
    return top


def search(query: str, topk: int) -> List[Tuple[int, float]]:
    embedder = get_embedder()
    q = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    q = np.asarray(q, dtype=np.float32)
    vecs = load_embeddings()
    return cosine_topk(q, vecs, topk)


st.set_page_config(page_title="Cebimde Müşavir", layout="wide")
st.title(APP_TITLE)

meta = load_meta()
chunks = load_chunks()
vecs = load_embeddings()

if not meta or not chunks or vecs.size == 0:
    st.error(
        "Index bulunamadı. Repo kökünde `index/` klasörü olmalı.\n\n"
        "Çözüm: `python build_index.py` çalıştır → `index/` klasörünü commit/push et."
    )
    st.stop()

st.caption(
    f"İndeks hazır: {meta.get('pdf_count', '?')} PDF | "
    f"{meta.get('chunk_count', '?')} parça"
)

with st.sidebar:
    topk = st.slider("Kaç sonuç?", 3, 20, DEFAULT_TOPK, 1)
    show_all = st.checkbox("Sonuçları açık göster", value=False)

query = st.text_area(
    "Sorun / olay / metin",
    height=140,
    placeholder="Örn: Kira geliri istisnası şartları, beyan sınırı ve tahsilat esasları nedir?"
)
run = st.button("Ara", width="stretch")

if run:
    if not query.strip():
        st.error("Soru/metin boş.")
    else:
        with st.spinner("Aranıyor..."):
            hits = search(query.strip(), topk)

        st.subheader("En alakalı bölümler")
        if not hits:
            st.info("Eşleşme bulunamadı.")
        else:
            for rank, (i, score) in enumerate(hits, start=1):
                rec = chunks[i]
                title = f"{rank}) Benzerlik: {score:.3f} | {rec['pdf']} | Sayfa {rec.get('page', '?')}"
                text = rec["text"]

                if show_all:
                    st.markdown(f"### {title}")
                    st.write(text)
                else:
                    with st.expander(title, expanded=(rank == 1)):
                        st.write(text)
else:
    st.info("Soru yaz → **Ara**'ya bas. (İndeks sayesinde hızlı.)")
