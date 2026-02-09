# app.py
import os
import re
import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st
from pypdf import PdfReader

APP_TITLE = "Cebimde Müşavir – Profesyonel Mevzuat Analizi"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk ayarları (istersen sonra oynarız)
CHUNK_MIN_LEN = 250
CHUNK_MAX_LEN = 1400
DEFAULT_TOPK = 7


# =========================
# Cache'li kaynaklar
# =========================

@st.cache_resource
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def _file_signature(path: str) -> Tuple[str, int, int]:
    """
    Cache key gibi düşün:
    dosya adı + boyut + mtime değişirse cache kırılır.
    """
    stt = os.stat(path)
    return (os.path.basename(path), int(stt.st_size), int(stt.st_mtime))


@st.cache_data(ttl=24 * 3600)
def pdf_to_text_cached(file_sig: Tuple[str, int, int], path: str) -> str:
    """
    PDF -> text (cache)
    file_sig sadece cache invalidation için.
    """
    reader = PdfReader(path)
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = re.sub(r"[ \t]+", " ", t).strip()
        if t:
            parts.append(f"[Sayfa {i+1}]\n{t}")
    return "\n\n".join(parts).strip()


def split_into_chunks(text: str, min_len: int = CHUNK_MIN_LEN, max_len: int = CHUNK_MAX_LEN) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Sayfa etiketlerini koruyarak bölmek için önce bloklara ayır
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]

    chunks: List[str] = []
    buf = ""

    for b in raw_blocks:
        if not buf:
            buf = b
            continue

        # Çok kısa ise birleştir
        if len(buf) < min_len:
            buf = buf + "\n\n" + b
            continue

        # Çok uzunsa kır
        if len(buf) > max_len:
            chunks.append(buf[:max_len].strip())
            buf = buf[max_len:].strip()
            if b:
                buf = (buf + "\n\n" + b).strip() if buf else b
            continue

        chunks.append(buf.strip())
        buf = b

    if buf:
        chunks.append(buf.strip())

    # Aşırı kısa parçaları ele
    chunks = [c for c in chunks if len(c) >= 120]
    return chunks


@st.cache_data(ttl=24 * 3600)
def embed_chunks_cached(chunks: List[str]) -> np.ndarray:
    """
    Chunks -> embeddings (cache)
    normalize_embeddings=True ile cosine çok hızlı olur.
    """
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)

    embedder = get_embedder()
    vecs = embedder.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return np.asarray(vecs, dtype=np.float32)


def cosine_topk(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if doc_vecs.size == 0:
        return []
    sims = doc_vecs @ query_vec  # normalize ise dot = cosine
    k = max(1, min(k, sims.shape[0]))
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    top = sorted([(int(i), float(sims[i])) for i in idx], key=lambda x: x[1], reverse=True)
    return top


@dataclass
class Corpus:
    chunks: List[str]
    vecs: np.ndarray
    chunk_meta: List[Dict[str, str]]  # {pdf, ...}


def build_corpus(selected_paths: List[str]) -> Corpus:
    """
    Seçilen PDF'lerden chunk + embedding üretir.
    Ağır işler cache'li fonksiyonlar sayesinde tekrar tekrar yapılmaz.
    """
    all_chunks: List[str] = []
    all_meta: List[Dict[str, str]] = []

    for path in selected_paths:
        sig = _file_signature(path)
        text = pdf_to_text_cached(sig, path)
        chunks = split_into_chunks(text)
        pdf_name = os.path.basename(path)

        for c in chunks:
            all_chunks.append(c)
            all_meta.append({"pdf": pdf_name})

    vecs = embed_chunks_cached(all_chunks)
    return Corpus(chunks=all_chunks, vecs=vecs, chunk_meta=all_meta)


def analyze_query(query: str, corpus: Corpus, topk: int) -> List[Tuple[int, float]]:
    embedder = get_embedder()
    q = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    q = np.asarray(q, dtype=np.float32)
    return cosine_topk(q, corpus.vecs, topk)


# =========================
# UI
# =========================

st.set_page_config(page_title="Cebimde Müşavir", layout="wide")
st.title(APP_TITLE)
st.caption("PDF tabanlı mevzuat araması. İlk açılış (cold start) Streamlit Cloud yüzünden yavaş olabilir; sonra cache devreye girer.")

# Repo kökünden PDF'leri topla
pdf_files = sorted(glob.glob("*.pdf"))
pretty_names = [os.path.basename(p) for p in pdf_files]

with st.sidebar:
    st.header("Kaynak PDF seçimi")
    if not pdf_files:
        st.error("Repo kökünde PDF bulunamadı. PDF'leri repo köküne yükle (şu an sen yüklemişsin, burada görünmeli).")
    else:
        st.write("Hız için: Hepsini seçme. İlgili 1–3 PDF seç.")
        selected = st.multiselect(
            "Taranacak PDF'ler",
            options=pretty_names,
            default=pretty_names[:2] if len(pretty_names) >= 2 else pretty_names
        )

    st.write("---")
    topk = st.slider("Kaç sonuç gösterilsin?", 3, 20, DEFAULT_TOPK, 1)
    show_all = st.checkbox("Sonuçları geniş aç", value=False)

query = st.text_area("Sorun / olay / metin (ne arıyoruz?)", height=140, placeholder="Örn: Kira geliri istisnası şartları, beyan sınırı ve tahsilat esasları nedir?")

run = st.button("Ara", width="stretch")

# Session state: corpus’u tutalım ki sayfa oynayınca tekrar inşa etmesin
if "corpus_key" not in st.session_state:
    st.session_state.corpus_key = None
if "corpus" not in st.session_state:
    st.session_state.corpus = None

def make_corpus_key(sel: List[str]) -> Tuple:
    # corpus key: seçilen pdf isimleri + signature'lar
    paths = [p for p in pdf_files if os.path.basename(p) in sel]
    sigs = tuple(_file_signature(p) for p in paths)
    return (tuple(sorted(sel)), sigs)

if run:
    if not query.strip():
        st.error("Arama metni boş. Bir soru/metin gir.")
    elif not pdf_files:
        st.error("PDF yok. Repo köküne PDF ekle.")
    elif not selected:
        st.error("En az 1 PDF seç.")
    else:
        selected_paths = [p for p in pdf_files if os.path.basename(p) in selected]
        corpus_key = make_corpus_key(selected)

        # Corpus yoksa veya seçilen pdf değiştiyse rebuild
        if st.session_state.corpus is None or st.session_state.corpus_key != corpus_key:
            with st.spinner("PDF'ler okunuyor, parçalara ayrılıyor ve indeksleniyor (ilk sefer uzun sürebilir)..."):
                st.session_state.corpus = build_corpus(selected_paths)
                st.session_state.corpus_key = corpus_key

        corpus: Corpus = st.session_state.corpus

        with st.spinner("En alakalı bölümler bulunuyor..."):
            hits = analyze_query(query.strip(), corpus, topk)

        st.subheader("En alakalı mevzuat bölümleri")
        if not hits:
            st.info("Eşleşme bulunamadı. (Metin çıkarma boş olabilir ya da çok kısa chunk kalmış olabilir.)")
        else:
            for rank, (idx, score) in enumerate(hits, start=1):
                meta = corpus.chunk_meta[idx]
                title = f"{rank}) Benzerlik: {score:.3f}  |  Kaynak: {meta['pdf']}"
                if show_all:
                    st.markdown(f"### {title}")
                    st.write(corpus.chunks[idx])
                else:
                    with st.expander(title, expanded=(rank == 1)):
                        st.write(corpus.chunks[idx])

        st.write("---")
        st.caption("Not: Bu aşama 'bul ve göster' (retrieval). Sonraki adımda bu bölümleri LLM'e verip 'profesyonel yorum + gerekçe + risk/öneri' ürettirebiliriz.")
else:
    st.info("PDF seç → soru yaz → **Ara**'ya bas.")
