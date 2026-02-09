import time
import os
import json
from typing import List, Dict, Tuple

import streamlit as st
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

# --------------------
# STARTUP TIMING
# --------------------
T0 = time.perf_counter()

APP_TITLE = "Cebimde Müşavir – (Hızlı Arama + Hesaplama)"
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
    topk = max(1, min(topk, sims.shape[0]))
    idx = sims.argsort()[-topk:][::-1]
    return [(int(i), float(sims[i])) for i in idx]


# --------------------
# HESAPLAMA: KİRA GELİRİ (BASİT MODÜL)
# Not: Bu bir "yardımcı hesap" modülüdür. Kesin işlem için mevzuat ve güncel tutarlar kontrol edilir.
# --------------------
def kira_basit_hesap(
    yillik_kira_tl: float,
    gider_yontemi: str,
    gercek_gider_tl: float,
    goturu_oran: float = 0.15,
    istisna_tl: float = 0.0
) -> Dict:
    """
    Basit kira matrahı hesabı:
    - İstisna (varsa) düşülür
    - Götürü gider (%15) ya da gerçek gider düşülür
    - Negatif olursa 0 yapılır
    """
    brut = max(0.0, yillik_kira_tl)
    kalan = max(0.0, brut - max(0.0, istisna_tl))

    if gider_yontemi == "Götürü (%15)":
        gider = kalan * goturu_oran
    else:
        gider = max(0.0, gercek_gider_tl)

    matrah = max(0.0, kalan - gider)

    return {
        "Brüt Kira (Yıllık)": brut,
        "İstisna": max(0.0, istisna_tl),
        "İstisna Sonrası": kalan,
        "Gider Yöntemi": gider_yontemi,
        "Gider": gider,
        "Vergiye Esas Matrah": matrah,
    }


# --------------------
# STREAMLIT UI
# --------------------
st.set_page_config(page_title="Cebimde Müşavir", layout="wide")
st.title(APP_TITLE)

T1 = time.perf_counter()

# Index var mı kontrol
index_ok = all(
    os.path.exists(p)
    for p in [CHUNKS_PATH, TFIDF_MAT_PATH, VECTORIZER_PATH, META_PATH]
)

if not index_ok:
    st.error(
        "Index bulunamadı. Repo kökünde `index/` klasörü olmalı.\n\n"
        "Çözüm: `python build_index.py` çalıştır → `index/` klasörünü commit/push et."
    )
    st.stop()

meta = load_meta()
chunks = load_chunks()
X, vectorizer = load_tfidf()

T2 = time.perf_counter()

st.caption(
    f"⏱ Açılış | UI: {(T1 - T0):.2f}s | Index: {(T2 - T1):.2f}s | Toplam: {(T2 - T0):.2f}s"
)

tab1, tab2 = st.tabs(["🔎 Mevzuat Arama (Hızlı)", "🧮 Vergi Hesaplama (Basit)"])

# --------------------
# TAB 1: ARAMA
# --------------------
with tab1:
    with st.sidebar:
        st.markdown("### Arama Ayarları")
        topk = st.slider("Kaç sonuç?", 3, 20, DEFAULT_TOPK, 1)
        show_all = st.checkbox("Sonuçları açık göster", value=False)

    query = st.text_area(
        "Sorun / olay / metin",
        height=140,
        placeholder="Örn: Kira geliri istisnası, beyan sınırı, tahsilat esasları nedir?"
    )

    if st.button("Ara", use_container_width=True):
        if not query.strip():
            st.error("Soru boş.")
        else:
            with st.spinner("Aranıyor..."):
                hits = search(query.strip(), topk)

            st.subheader("En alakalı bölümler")
            if not hits:
                st.info("Eşleşme bulunamadı.")
            else:
                for rank, (i, score) in enumerate(hits, start=1):
                    rec = chunks[i]
                    title = f"{rank}) Skor {score:.4f} | {rec['pdf']} | Sayfa {rec.get('page','?')}"
                    if show_all:
                        st.markdown(f"### {title}")
                        st.write(rec["text"])
                    else:
                        with st.expander(title, expanded=(rank == 1)):
                            st.write(rec["text"])

# --------------------
# TAB 2: HESAPLAMA
# --------------------
with tab2:
    st.markdown("### Kira Geliri – Basit Matrah Hesabı")
    st.caption("Bu modül hızlı ve pratik bir taslaktır. Kesin vergi için güncel istisna/oranlar ve durum detayları kontrol edilmelidir.")

    col1, col2, col3 = st.columns(3)
    with col1:
        yillik_kira = st.number_input("Yıllık brüt kira (TL)", min_value=0.0, value=0.0, step=1000.0)
    with col2:
        istisna = st.number_input("Kira istisnası (TL) (varsa)", min_value=0.0, value=0.0, step=500.0)
    with col3:
        gider_yontemi = st.selectbox("Gider yöntemi", ["Götürü (%15)", "Gerçek gider"])

    gercek_gider = 0.0
    if gider_yontemi == "Gerçek gider":
        gercek_gider = st.number_input("Gerçek gider toplamı (TL)", min_value=0.0, value=0.0, step=500.0)

    if st.button("Hesapla", use_container_width=True):
        sonuc = kira_basit_hesap(
            yillik_kira_tl=float(yillik_kira),
            gider_yontemi=gider_yontemi,
            gercek_gider_tl=float(gercek_gider),
            goturu_oran=0.15,
            istisna_tl=float(istisna),
        )
        df = pd.DataFrame([sonuc])
        st.success("Hesaplandı.")
        st.dataframe(df, use_container_width=True)

        st.markdown("#### Kaynak taraması (ilgili mevzuat bölümleri)")
        with st.spinner("Mevzuatta ilgili yerler aranıyor..."):
            hits = search("kira geliri istisna götürü gider gerçek gider beyan", 6)
        for rank, (i, score) in enumerate(hits, start=1):
            rec = chunks[i]
            title = f"{rank}) Skor {score:.4f} | {rec['pdf']} | Sayfa {rec.get('page','?')}"
            with st.expander(title, expanded=(rank == 1)):
                st.write(rec["text"])
