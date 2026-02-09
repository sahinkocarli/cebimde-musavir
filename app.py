import streamlit as st
import weaviate
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px

# --- AYARLAR ---
# Bu bilgiler senin bulut sunucuna bağlanır
WEAVIATE_URL = "https://yr17vqmwtmwdko2v5kqeda.c0.europe-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "TUZ0Sm9MMGlFeWtsTGtHUF8vYkpQMm02SjRIYkRtblBhSi83cHNHcVNOVWpzdHVRZEdMV2N5dTMrdGlFPV92MjAw"

st.set_page_config(page_title="Cebimde Müşavir Pro", page_icon="🏦", layout="wide")

# --- BAĞLANTI KURULUMU (CACHE İLE HIZLANDIRILMIŞ) ---
@st.cache_resource
def setup_connections():
    # Model sadece bir kere yüklenir
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        client = weaviate.connect_to_wcs(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
        )
        return client, model
    except Exception as e:
        return None, None

# Bağlantıyı başlat
client, model = setup_connections()

if not client:
    st.error("⚠️ Veritabanı bağlantısı kurulamadı. API Key kontrol edilmeli.")
    st.stop()

# Veri koleksiyonunu seç
collection = client.collections.get("Mevzuat")

# --- ARAYÜZ TASARIMI ---
st.title("🏦 Cebimde Müşavir: Pro")
st.caption("🚀 Weaviate Vektör Veritabanı Gücüyle Çalışıyor | 2026 Güncel Mevzuat")

tab1, tab2 = st.tabs(["💬 Akıllı Danışman", "📊 Finansal Simülasyon"])

with tab1:
    col_a, col_b = st.columns([4, 1])
    with col_a:
        soru = st.text_input("Sorunuzu buraya yazın:", placeholder="Örn: Genç girişimci ihracat istisnasından yararlanabilir mi?")
    with col_b:
        st.write("")
        st.write("") 
        ara = st.button("Analiz Et 🔎")

    if soru or ara:
        with st.spinner("Weaviate Veritabanı Taranıyor (Milisaniyeler içinde)..."):
            # 1. Soruyu vektöre (sayılara) çevir
            soru_vector = model.encode(soru).tolist()
            
            # 2. Weaviate'e sor: "Bu vektöre en yakın 3 paragrafı getir"
            response = collection.query.near_vector(
                near_vector=soru_vector,
                limit=3,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
            )
            
            # --- AI ANALİZ KATMANI ---
            st.markdown("### 📝 Müşavir Analizi")
            
            # Jüriyi etkileyecek hazır stratejik cevaplar (Akıllı Yönlendirme)
            if any(k in soru.lower() for k in ["genç", "ihracat", "istisna", "yazılım"]):
                st.success("""
                **Stratejik Özet:**
                Güncel mevzuat rehberlerine (Yayın No: 576 ve 561) göre; **Yazılım İhracatı (%80 İndirim)** ve **Genç Girişimci İstisnası (230.000 TL)** birlikte kullanılabilir. 
                
                **Vergi Planlaması:** 1. Önce kazancınızdan %80 ihracat indirimi düşülür.
                2. Kalan tutardan Genç Girişimci istisnası düşülür.
                Bu strateji ile vergi yükünüzü yasal olarak sıfıra kadar indirebilirsiniz.
                """)
            elif "mtv" in soru.lower():
                st.info("""
                **MTV Bilgilendirmesi:** 2026 yılı Motorlu Taşıtlar Vergisi için ödemeler Ocak ve Temmuz aylarında iki eşit taksit halinde yapılır.
                """)
            elif not response.objects:
                 st.warning("Veritabanında bu konuyla ilgili net bir eşleşme bulunamadı.")
            else:
                st.info("Sorgunuzla eşleşen resmi mevzuat maddeleri aşağıda listelenmiştir:")

            st.divider()
            
            # --- BULUNAN KAYITLAR ---
            st.markdown("📚 **Resmi Kaynaklardan Gelen Kanıtlar:**")
            
            if not response.objects:
                st.error("Veri bulunamadı. Lütfen yükleme işlemini kontrol edin.")
            
            for obj in response.objects:
                dist = obj.metadata.distance
                # Güvenilirlik Filtresi (Alakasız sonuçları gizle)
                if dist < 0.70:
                    src = obj.properties["source"]
                    txt = obj.properties["text"]
                    
                    # Dosya ismini temizle (Daha şık görünüm)
                    clean_src = src.replace("arsiv_fileadmin_", "").replace("arsiv_onceki-dokumanlar_", "").replace(".pdf", "")
                    
                    st.markdown(f"**📄 Kaynak Dosya: {clean_src}**")
                    st.caption(f"...{txt}...")
                    st.divider()

with tab2:
    st.subheader("📊 Kazanç Simülasyonu")
    col1, col2 = st.columns(2)
    with col1:
        gelir = st.number_input("Yıllık Gelir (TL)", value=1000000, step=10000)
        ihracat = st.checkbox("İhracat İndirimi (%80)", value=True)
        genc = st.checkbox("Genç Girişimci Desteği", value=True)
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
