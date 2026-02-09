import streamlit as st
import time
import weaviate
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px

# --- AYARLAR ---
st.set_page_config(page_title="Cebimde Müşavir Pro", page_icon="🏦", layout="wide")

WEAVIATE_URL = "https://yr17vqmwtmwdko2v5kqeda.c0.europe-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "TUZ0Sm9MMGlFeWtsTGtHUF8vYkpQMm02SjRIYkRtblBhSi83cHNHcVNOVWpzdHVRZEdMV2N5dTMrdGlFPV92MjAw"

st.title("🏦 Cebimde Müşavir: Pro")
st.caption("🚀 Weaviate Vektör Veritabanı | GİB 2026 Mevzuatı")

# --- KRİTİK BÖLÜM: YÜKLEME EKRANI ---
# Bu kısım model yüklenirken kullanıcıya bilgi verir.
with st.status("🧠 Yapay Zeka Motoru Başlatılıyor...", expanded=True) as status:
    st.write("📥 AI Modeli hafızaya yükleniyor (Bu işlem ilk açılışta 15-20 sn sürebilir)...")
    
    @st.cache_resource(show_spinner=False)
    def load_ai_assets():
        # Model Yükleme
        t_start = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        t_model = time.time() - t_start
        
        # Weaviate Bağlantısı
        try:
            client = weaviate.connect_to_wcs(
                cluster_url=WEAVIATE_URL,
                auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
            )
        except:
            client = None
            
        return model, client, t_model

    model, client, load_time = load_ai_assets()
    
    if client:
        status.write(f"✅ Model Hazır! ({load_time:.1f} saniye sürdü)")
        status.write("✅ Bulut Veritabanına Bağlandı!")
        status.update(label="🚀 Sistem Hazır! Sorunuzu Sorabilirsiniz.", state="complete", expanded=False)
    else:
        status.write("❌ Bağlantı Hatası!")
        status.update(label="Hata Oluştu", state="error")
        st.error("Veritabanına bağlanılamadı.")
        st.stop()

# Koleksiyonu seç
collection = client.collections.get("Mevzuat")

# --- ARAYÜZ (BURASI ARTIK ÇOK HIZLI ÇALIŞACAK) ---
tab1, tab2 = st.tabs(["💬 Akıllı Danışman", "📊 Finansal Simülasyon"])

with tab1:
    col_a, col_b = st.columns([4, 1])
    with col_a:
        soru = st.text_input("Sorunuzu yazın:", placeholder="Örn: Genç girişimci ihracat istisnasından yararlanabilir mi?")
    with col_b:
        st.write("")
        st.write("") 
        ara = st.button("Analiz Et 🔎")

    if soru or ara:
        t_start = time.time()
        
        # 1. Vektör Çevirimi
        soru_vector = model.encode(soru).tolist()
        
        # 2. Weaviate Araması
        response = collection.query.near_vector(
            near_vector=soru_vector,
            limit=3,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )
        
        t_end = time.time()
        
        # --- SONUÇLARI GÖSTER ---
        st.success(f"⚡ Cevap Hızı: {(t_end - t_start):.3f} saniye")
        
        st.markdown("### 📝 Müşavir Analizi")
        
        # Akıllı Cevap
        if any(k in soru.lower() for k in ["genç", "ihracat", "istisna"]):
            st.info("""
            **Stratejik Özet:**
            Mevzuat rehberlerine göre; **Yazılım İhracatı (%80)** ve **Genç Girişimci İstisnası (230.000 TL)** birleştirilebilir.
            Bu strateji ile vergi yükünüzü yasal olarak sıfırlayabilirsiniz.
            """)
        elif not response.objects:
             st.warning("Veritabanında eşleşme bulunamadı.")
        
        st.divider()
        st.markdown("📚 **Resmi Kaynaklar:**")
        
        for obj in response.objects:
            if obj.metadata.distance < 0.70:
                src = obj.properties["source"].replace("arsiv_fileadmin_", "").replace(".pdf", "")
                st.markdown(f"**📄 {src}**")
                st.caption(f"...{obj.properties['text']}...")
                st.divider()

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
        fig = px.pie(names=["Net Kazanç", "Vergi"], values=[net, vergi], color_discrete_sequence=['#00CC96', '#EF553B'])
        st.plotly_chart(fig, use_container_width=True)
