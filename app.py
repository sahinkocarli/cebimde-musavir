import streamlit as st
import time, requests
import plotly.express as px

st.set_page_config(page_title="Cebimde Müşavir Pro", page_icon="🏦", layout="wide")

# Secrets önerilir:
# WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
# WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"]
# HF_TOKEN = st.secrets["HF_TOKEN"]

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

st.title("🏦 Cebimde Müşavir: Pro (Demo)")
st.caption("🚀 GİB 2026 Mevzuatı | Anlık Analiz Modu")

tab1, tab2 = st.tabs(["💬 Akıllı Danışman", "📊 Finansal Simülasyon"])

if "soru" not in st.session_state:
    st.session_state.soru = ""

with tab1:
    col_a, col_b, col_c = st.columns([4, 1, 1])
    with col_a:
        soru = st.text_input("Sorunuzu yazın:", key="soru",
                             placeholder="Örn: Genç girişimci ihracat istisnasından yararlanabilir mi?")
    with col_b:
        st.write("")
        ara = st.button("Analiz Et 🔎")
    with col_c:
        st.write("")
        temizle = st.button("Temizle 🧹")

    if temizle:
        st.session_state.soru = ""
        st.rerun()

    if ara and soru:
        soru_lower = soru.lower()

        if any(k in soru_lower for k in ["genç", "ihracat", "istisna", "girişimci", "yazılım"]):
            with st.spinner("Mevzuat Taranıyor..."):
                time.sleep(0.6)  # demo gecikmesi azalt
            st.success("⚡ Analiz Tamamlandı (Demo)")
            st.markdown("### 📝 Müşavir Analizi")
            st.info("... (hazır demo metnin) ...")

        elif "mtv" in soru_lower:
            st.success("⚡ Analiz Tamamlandı (Demo)")
            st.info("2026 MTV ödemeleri Ocak ve Temmuz...")

        else:
            st.warning("Bu soru demo senaryosunda yok. (Gerçek aramayı sonra bağlarız.)")

with tab2:
    st.subheader("📊 Kazanç Simülasyonu")
    col1, col2 = st.columns(2)
    with col1:
        gelir = st.number_input("Yıllık Gelir (TL)", value=1_000_000, step=10_000)
        ihracat = st.checkbox("İhracat İndirimi (%80)", value=True)
        genc = st.checkbox("Genç Girişimci", value=True)
    with col2:
        matrah = gelir
        if ihracat: matrah *= 0.20
        if genc: matrah = max(0, matrah - 230_000)
        vergi = matrah * 0.20
        net = gelir - vergi

        fig = px.pie(names=["Net Kazanç", "Vergi"], values=[net, vergi], hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Net Kazanç", f"{net:,.0f} TL")
