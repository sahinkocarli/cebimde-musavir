import streamlit as st
import time
import weaviate
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hız Testi Modu", page_icon="⚡", layout="wide")

# --- AYARLAR ---
WEAVIATE_URL = "https://yr17vqmwtmwdko2v5kqeda.c0.europe-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "TUZ0Sm9MMGlFeWtsTGtHUF8vYkpQMm02SjRIYkRtblBhSi83cHNHcVNOVWpzdHVRZEdMV2N5dTMrdGlFPV92MjAw"

st.title("⚡ Sistem Hız Tanı Ekranı")

# --- ADIM 1: AI MODELİ YÜKLEME ---
t1 = time.time()
with st.status("🧠 1. Adım: Yapay Zeka Beyni Yükleniyor...", expanded=True) as status:
    @st.cache_resource
    def load_model():
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    model = load_model()
    gecen_sure_model = time.time() - t1
    status.write(f"✅ Model Yüklendi! Süre: {gecen_sure_model:.2f} saniye")
    
    if gecen_sure_model > 5:
        status.update(label="⚠️ Model Yüklemesi Yavaş (Streamlit Sunucusu Yoğun)", state="error")
    else:
        status.update(label="🚀 Model Hazır", state="complete")

# --- ADIM 2: BULUT VERİTABANI BAĞLANTISI ---
t2 = time.time()
with st.status("☁️ 2. Adım: Weaviate Bulutuna Bağlanılıyor...", expanded=True) as status:
    @st.cache_resource
    def connect_weaviate():
        try:
            client = weaviate.connect_to_wcs(
                cluster_url=WEAVIATE_URL,
                auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
            )
            return client
        except Exception as e:
            return None

    client = connect_weaviate()
    gecen_sure_baglanti = time.time() - t2
    
    if client:
        status.write(f"✅ Buluta Bağlandı! Süre: {gecen_sure_baglanti:.2f} saniye")
        status.update(label="🚀 Veritabanı Aktif", state="complete")
    else:
        status.write("❌ Bağlantı Hatası!")
        status.update(label="Bağlantı Başarısız", state="error")
        st.stop()

collection = client.collections.get("Mevzuat")

# --- ARAYÜZ VE SORGULAMA ---
st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    soru = st.text_input("Sorgu Testi:", placeholder="Genç girişimci istisnası nedir?")
with col2:
    st.write("")
    st.write("")
    btn = st.button("Hızı Test Et ⏱️")

if soru or btn:
    t3 = time.time()
    
    # VEKTÖR ÇEVİRİMİ
    soru_vector = model.encode(soru).tolist()
    t4 = time.time()
    vektor_suresi = t4 - t3
    
    # WEAVIATE ARAMASI
    response = collection.query.near_vector(
        near_vector=soru_vector,
        limit=3,
        return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
    )
    t5 = time.time()
    arama_suresi = t5 - t4
    
    # SONUÇLARI GÖSTER
    st.success(f"⚡ TOPLAM CEVAP SÜRESİ: {(t5-t3):.4f} Saniye")
    
    col_a, col_b = st.columns(2)
    col_a.metric("Sorguyu Sayıya Çevirme", f"{vektor_suresi:.4f} sn")
    col_b.metric("Bulutta Arama", f"{arama_suresi:.4f} sn")
    
    st.markdown("### 📝 Gelen Cevaplar:")
    if any(k in soru.lower() for k in ["genç", "ihracat"]):
         st.info("💡 (Burada Müşavirin Yorumu Görünecek - Sistem Hızlı Çalışıyor)")
         
    for obj in response.objects:
        st.caption(f"📄 Kaynak: {obj.properties['source']} | Benzerlik: %{(1-obj.metadata.distance)*100:.1f}")
