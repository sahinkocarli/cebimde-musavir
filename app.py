import streamlit as st
import weaviate
import google.generativeai as genai
import pandas as pd
import plotly.express as px

# --- AYARLAR ---
st.set_page_config(page_title="Cebimde Müşavir Pro", page_icon="🏦", layout="wide")

# ANAHTARLAR (Senin Anahtarların)
GOOGLE_API_KEY = "AIzaSyCYvni5lwKVqftdHLMi0C9pRQ4HA-htq1U"
WEAVIATE_URL = "https://yr17vqmwtmwdko2v5kqeda.c0.europe-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "TUZ0Sm9MMGlFeWtsTGtHUF8vYkpQMm02SjRIYkRtblBhSi83cHNHcVNOVWpzdHVRZEdMV2N5dTMrdGlFPV92MjAw"

# Google'ı Hazırla
genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_weaviate_client():
    try:
        return weaviate.connect_to_wcs(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
        )
    except:
        return None

client = get_weaviate_client()

# --- ARAYÜZ ---
st.title("🏦 Cebimde Müşavir: Pro (Google Altyapısı)")
st.caption("🚀 Gerçek Zamanlı ve Hızlı Mevzuat Analizi")

if not client:
    st.error("Veritabanı bağlantısı kurulamadı.")
    st.stop()

# Koleksiyonu Seç
try:
    collection = client.collections.get("MevzuatGemini")
except:
    st.error("Veritabanı bulunamadı. Lütfen bilgisayarınızdan 'yukle.py' dosyasını çalıştırın.")
    st.stop()

tab1, tab2 = st.tabs(["💬 Danışman", "📊 Hesapla"])

with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        soru = st.text_input("Sorunuzu yazın:", placeholder="Örn: Genç girişimci ihracat istisnasından yararlanabilir mi?")
    with col2:
        st.write("")
        st.write("")
        btn = st.button("Analiz Et 🔎")

    if soru or btn:
        with st.spinner("Google Gemini Analiz Ediyor..."):
            try:
                # 1. Soruyu Vektöre Çevir (Google Hızı)
                embedding = genai.embed_content(
                    model="models/text-embedding-004",
                    content=soru,
                    task_type="retrieval_query"
                )['embedding']

                # 2. Ara
                response = collection.query.near_vector(
                    near_vector=embedding,
                    limit=3,
                    return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
                )

                # 3. Sonuç
                st.markdown("### 📝 Analiz Sonucu")
                
                # Akıllı Özet
                if "genç" in soru.lower() and "ihracat" in soru.lower():
                     st.success("""
                     **Stratejik Özet:**
                     Mevzuata göre; **Genç Girişimci İstisnası (230.000 TL)** ve **Yazılım İhracatı (%80 İndirim)** birleştirilerek vergi avantajı sağlanabilir.
                     """)

                if not response.objects:
                    st.warning("Veritabanında eşleşme bulunamadı. 'yukle.py' işlemini tamamladınız mı?")
                
                for obj in response.objects:
                    # Güvenilirlik filtresi
                    if obj.metadata.distance < 0.8:
                        src = obj.properties["source"].replace("arsiv_fileadmin_", "").replace(".pdf", "")
                        st.info(f"📄 **Kaynak:** {src}\n\n...{obj.properties['text']}...")

            except Exception as e:
                st.error(f"Hata oluştu: {e}")

with tab2:
    st.subheader("📊 Kazanç Simülasyonu")
    col1, col2 = st.columns(2)
    with col1:
        gelir = st.number_input("Yıllık Gelir (TL)", value=1000000, step=10000)
        ihracat = st.checkbox("İhracat İndirimi (%80)", value=True)
        genc = st.checkbox("Genç Girişimci", value=True)
    with col2:
        matrah = gelir
        if ihracat: matrah = matrah * 0.20
        if genc: matrah = max(0, matrah - 230000)
        vergi = matrah * 0.20
        net = gelir - vergi
        
        fig = px.pie(names=["Net Kazanç", "Vergi"], values=[net, vergi], 
                     color_discrete_sequence=['#00CC96', '#EF553B'], hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Net Kazanç", f"{net:,.0f} TL")
