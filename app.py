import os
import json
from typing import List, Dict, Tuple

import streamlit as st
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

APP_TITLE = "Cebimde Müşavir – Profesyonel Mevzuat Analizi (Hızlı)"
INDEX_DIR = "index"
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
TFIDF_MAT_PATH = os.path.join(INDEX_DIR, "tfidf_matrix.npz")
VECTORIZER_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.joblib")
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
def load_tfidf():
    if not (os.path.exists(TFIDF_MAT_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None
    X = sparse.load_npz(TFIDF_MAT_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return X, vectorizer


def search(query: str, topk: int) -> List[Tuple[int, float]]:
    X, vectorizer = load_tfidf()
    if X is None or vectorizer is None:
        return []
    q = vectorizer.transform([query])
    sims = linear_kernel(q, X).ravel()
    topk = max(1, min(topk, sims.shape[0]))
    idx = sims.argsort()[-topk:][::-1]
    return [(int(i), float(sims[i])) for i in idx]


st.set_page_config(page_title="Cebimde Müşavir", layout="wide")
st.title(APP_TITLE)

meta = load_meta()
chunks = load_chunks()
X, vectorizer = load_tfidf()

if not meta or not chunks or X is None or vectorizer is None:
    st.error(
        "Index bulunamadı. Repo kökünde index/ klasörü olmalı.\n\n"
        "Çözüm: python build_index.py çalıştır → index/ klasörünü commit/push et."
    )
    st.stop()

st.caption(
    f"İndeks hazır: {meta.get('pdf_count', '?')} PDF | "
    f"{meta.get('chunk_count', '?')} parça | "
    f"Motor: {meta.get('engine', 'tfidf')}"
)

with st.sidebar:
    topk = st.slider("Kaç sonuç?", 3, 20, DEFAULT_TOPK, 1)
    show_all = st.checkbox("Sonuçları açık göster", value=False)

query = st.text_area(
    "Sorun / olay / metin",
    height=140,
    placeholder="Örn: Kira geliri istisnası istisna tutarı, beyan sınırı ve tahsilat esasları nedir?"
)

if st.button("Ara", width="stretch"):
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
                title = f"{rank}) Skor: {score:.4f} | {rec['pdf']} | Sayfa {rec.get('page', '?')}"
                if show_all:
                    st.markdown(f"### {title}")
                    st.write(rec["text"])
                else:
                    with st.expander(title, expanded=(rank == 1)):
                        st.write(rec["text"])
