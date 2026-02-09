import streamlit as st

# =========================
# 1) CACHE'Lİ KAYNAKLAR (BURAYA)
# =========================

@st.cache_resource
def get_client():
    """
    Buraya LLM client / DB connection / model load koy.
    Örn: OpenAI client, Weaviate client, vs.
    """
    # return client
    return None


@st.cache_data(ttl=3600)
def load_mevzuat_text():
    """
    Eğer dosyadan mevzuat okuyorsan buraya al.
    Her rerun'da dosya okuma işkencesi biter.
    """
    # with open("mevzuat.txt", "r", encoding="utf-8") as f:
    #     return f.read()
    return ""


@st.cache_data(ttl=3600)
def analyze_cached(user_text: str):
    """
    Ağır analizi buraya koy.
    Bu fonksiyonun içinde LLM çağrısı / parsing / embedding ne varsa olur.
    """
    client = get_client()
    mevzuat = load_mevzuat_text()

    # === BURAYA SENİN ANALİZ KODUN GELECEK ===
    # result = analyze(user_text, mevzuat, client)
    result = f"Demo sonuç: {user_text[:200]}"
    return result


# =========================
# 2) UI (BURADAN AŞAĞISI EKRAN)
# =========================

st.set_page_config(page_title="Cebimde Müşavir", layout="wide")
st.title("Cebimde Müşavir – Profesyonel Mevzuat Analizi")

# Session state (sonucu tutmak için)
if "last_text" not in st.session_state:
    st.session_state.last_text = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

user_text = st.text_area("Metninizi girin", height=200)

col1, col2 = st.columns([1, 3])
with col1:
    run = st.button("Analiz Et", use_container_width=True)

# =========================
# 3) BUTON TIKLANINCA ÇALIŞTIR (EN KRİTİK YER)
# =========================

if run:
    if not user_text.strip():
        st.warning("Analiz için metin girmen gerekiyor.")
    else:
        # Aynı metin tekrar analiz edilmesin (boşa bekletmesin)
        if user_text != st.session_state.last_text:
            with st.spinner("Analiz yapılıyor..."):
                st.session_state.last_result = analyze_cached(user_text)
            st.session_state.last_text = user_text

# =========================
# 4) SONUCU GÖSTER
# =========================

if st.session_state.last_result:
    st.subheader("Sonuç")
    st.write(st.session_state.last_result)
