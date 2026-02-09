import time
import os
import json
from typing import List, Dict, Tuple

import streamlit as st
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

# --------------------
# STARTUP TIMING
# --------------------
T0 = time.perf_counter()

APP_TITLE = "Cebimde Müşavir – Profesyonel Mevzuat Analizi (Hızlı)"
INDEX_DIR = "index"
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
TFIDF_MAT_PATH = os.path.join(INDEX_DIR, "tfidf_matrix.npz")
VECTORIZER_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.joblib")
META_PATH = os.path.join(INDEX_DIR, "meta.json")
DEFAULT_TOPK = 8


@st.cache_data
def load_meta() -> Dict:
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_chunks() -> List[Dict]:
    out = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


@st.cache_resource
def load_tfidf():
    X = sparse.load_npz(TFIDF_MAT_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return X, vectorizer


def search(query: str, topk: int):
    X, vectorizer = load_tfidf()
    q = vectorizer.transform([query])
    sims = linear_kernel(q, X).ravel()
    idx = sims.argsort()[-topk:][::-1]
    return [(int(i), float(sims[i])) for i in idx]


# --------------------
# STREAMLIT UI
# --------------------
st.set_page_config(page_title="Cebimde Müşavir", layout="wide")
st.title(APP_TITLE)

T1 = time.perf_counter()

meta = load_meta()
chunks = load_chunks()
X, vectorizer = load_tfidf()

T2 = time.perf_counter()

st.caption(
    f"⏱ Başlangıç süreleri | "
    f"UI: {(T1 - T0):.2f}s | "
    f"Index yükleme: {(T2 - T1):.2f}s | "
    f"Toplam: {(T2 - T0):.2f}s"
)

with st.sidebar:
    topk = st.slider("Kaç sonuç?", 3, 20, DEFAULT_TOPK, 1)
    show_all = st.checkbox("Sonuçları açık göster", value=False)

query = st.text_area(
    "Sorun / olay / metin",
    height=140,
    placeholder="Örn: Kira geliri istisnası, beyan sınırı nedir?"
)

if st.button("Ara", use_container_width=True):
    if not query.strip():
        st.error("Soru boş.")
    else:
        with st.spinner("Aranıyor..."):
            hits = search(query.strip(), topk)

        for rank, (i, score) in enumerate(hits, start=1):
            rec = chunks[i]
            title = f"{rank}) Skor {score:.4f} | {rec['pdf']} | Sayfa {rec['page']}"
            if show_all:
                st.markdown(f"### {title}")
                st.write(rec["text"])
            else:
                with st.expander(title, expanded=(rank == 1)):
                    st.write(rec["text"])
