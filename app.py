# app.py
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st

APP_TITLE = "Cebimde Müşavir – Profesyonel Mevzuat Analizi"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOPK = 5


# =========================
# Cache'li kaynaklar
# =========================

@st.cache_resource
def get_embedder():
    # Import'u içeri alıyoruz ki cold-start dışında her rerun'da import/modele takılmasın.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(ttl=3600)
def load_text_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str, min_len: int = 250, max_len: int = 1400) -> List[str]:
    """
    Metni paragraflara göre böler. Çok kısa parçaları birleştirir.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Paragraf ayırma: boş satır veya çoklu newline
    raw = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    buf = ""

    for p in raw:
        if not buf:
            buf = p
            continue

        # buffer çok kısa ise birleştir
        if len(buf) < min_len:
            buf = buf + "\n\n" + p
            continue

        # buffer çok uzunsa kır
        if len(buf) > max_len:
            chunks.append(buf[:max_len].strip())
            buf = buf[max_len:].strip()
            if p:
                buf = (buf + "\n\n" + p).strip() if buf else p
            continue

        # normal: chunk'ı kapat, yenisine geç
        chunks.append(buf)
        buf = p

    if buf:
        chunks.append(buf)

    # Son temizlik
    chunks = [c.strip() for c in chunks if len(c.strip()) >= 80]
    return chunks


@st.cache_data(ttl=3600)
def embed_chunks(chunks: List[str]) -> np.ndarray:
    """
    Mevzuat chunk'larını embed eder ve L2 normalize eder.
    """
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)

    embedder = get_embedder()
    vecs = embedder.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine için ideal
    )
    return np.asarray(vecs, dtype=np.float32)


def cosine_topk(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """
    query_vec: (d,)
    doc_vecs: (n, d) normalize varsayımıyla dot product = cosine
    """
    if doc_vecs.size == 0:
        return []
    sims = doc_vecs @ query_vec  # (n,)
    k = max(1, min(k, sims.shape[0]))
    idx = np.argpartition(-sims, kth=k-1)[:k]
    top = sorted([(int(i), float(sims[i])) for i in idx], key=lambda x: x[1], reverse=True)
    return top


@dataclass
class AnalysisResult:
    top: List[Tuple[int, float]]
    chunks: List[str]


def analyze(user_text: str, mevzuat_text: str, topk: int) -> AnalysisResult:
    chunks = split_into_chunks(mevzuat_text)
    doc_vecs = embed_chunks(chunks)

    embedder = get_embedder()
    q = embedder.encode([user_text], normalize_embeddings=True, show_progress_bar=False)[0]
    q = np.asarray(q, dtype=np.float32)

    top = cosine_topk(q, doc_vecs, topk)
    return AnalysisResult(top=top, chunks=chunks)


# =========================
# UI
# =========================

st.set_page_config(page_title="Cebimde Müşavir", layout="wide")
st.title(APP_TITLE)
st.caption("Not: Streamlit Cloud'da ilk açılış (cold start) biraz uzun sürebilir. Sonrasında cache devreye girer.")

# Mevzuat kaynağı
with st.sidebar:
    st.header("Ayarlar")
    topk = st.slider("Kaç sonuç gösterilsin?", 3, 15, DEFAULT_TOPK, 1)
    st.write("---")
    source_mode = st.radio("Mevzuat kaynağı", ["Dosyadan (mevzuat.txt)", "Elle yapıştır"], index=0)

    mevzuat_text = ""
    if source_mode == "Dosyadan (mevzuat.txt)":
        mevzuat_text = load_text_file("mevzuat.txt")
        if not mevzuat_text:
            st.warning("Repo'da `mevzuat.txt` bulunamadı. Sidebar'dan 'Elle yapıştır' seçebilirsin.")
    else:
        mevzuat_text = st.text_area("Mevzuat metnini buraya yapıştır", height=220)

# Kullanıcı metni
user_text = st.text_area("Analiz edilecek metin / olay / soru", height=200, placeholder="Örn: Kira geliri elde ettim, beyan ve istisna şartları nedir?")

col1, col2, col3 = st.columns([1, 1, 6])
run = col1.button("Analiz Et", width="stretch")
col2.button("Temizle", width="stretch", on_click=lambda: st.session_state.clear())

# Sonuçları session_state'te tut (sayfa oynayınca kaybolmasın)
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

if run:
    if not user_text.strip():
        st.error("Metin boş. Analiz için bir şey yaz.")
    elif not (mevzuat_text or "").strip():
        st.error("Mevzuat metni yok. `mevzuat.txt` ekle veya 'Elle yapıştır' seç.")
    else:
        # Aynı metni tekrar tekrar hesaplama
        if user_text.strip() != st.session_state.last_query:
            with st.spinner("Mevzuat taranıyor, ilgili bölümler bulunuyor..."):
                res = analyze(user_text.strip(), mevzuat_text, topk)
            st.session_state.last_result = res
            st.session_state.last_query = user_text.strip()

# Gösterim
res = st.session_state.last_result
if res:
    st.subheader("En alakalı mevzuat bölümleri")
    if not res.top:
        st.info("Eşleşme bulunamadı (mevzuat boş veya chunk oluşmadı).")
    else:
        for rank, (i, score) in enumerate(res.top, start=1):
            with st.expander(f"{rank}) Benzerlik: {score:.3f}  |  Bölüm #{i+1}", expanded=(rank == 1)):
                st.write(res.chunks[i])

    st.write("---")
    st.subheader("İpucu")
    st.write(
        "Bu sürüm sadece 'en alakalı mevzuat parçalarını' çıkarır. "
        "İstersen bir sonraki adımda bu parçaları LLM'e verip 'profesyonel yorum / gerekçe / riskler' üreten katmanı ekleriz."
    )
else:
    st.info("Metni girip **Analiz Et**'e bas. (İlk kez açtıysan model yüklenmesi biraz sürebilir.)")
