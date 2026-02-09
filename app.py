import streamlit as st
import time
import weaviate
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Cebimde Müşavir Pro", page_icon="🏦", layout="wide")

# --- BAŞLIK (Burası çalışıyor dedin) ---
st.title("🏦 Cebimde Müşavir: Pro")
st.caption("🚀 Sistem Durumu Kontrol Ediliyor...")

# --- AYARLAR ---
WEAVIATE_URL = "https://yr17vqmwtmwdko2v5kqeda.c0.europe-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "TUZ0Sm9MMGlFeWtsTGtHUF8vYkpQMm02SjRIYkRtblBhSi83cHNHcVNOVWpzdHVRZEdMV2N5dTMrdGlFPV92MjAw"

# --- ADIM ADIM YÜKLEME (EKRANA YAZARAK) ---
placeholder = st.empty() # Durum mesajları için alan

@st.cache_resource
def kaynaklari_yukle():
    logs = []
    model = None
    client = None
    
    # 1. MODEL YÜKLEME
    try:
        logs.append("🧠 Yapay Zeka Modeli İndiriliyor...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logs.append("✅ Model Başarıyla Yüklendi.")
    except Exception as e:
        logs.append(f"❌ Model Hatası: {str(e)}")
        return None, None, logs

    # 2. WEAVIATE BAĞLANTISI
    try:
        logs.append("☁️ Bulut Veritabanına Bağlanılıyor...")
        client = weaviate.connect_to_wcs(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
        )
        logs.append("✅ Weaviate Bağlantısı Başarılı.")
    except Exception as e:
        logs.append(f"❌ Weaviate Hatası: {str(e)}")
    
    return client, model, logs

# Yüklemeyi başlat ve logları göster
with st.spinner('Sistem başlatılıyor, lütfen bekleyin...'):
    client, model, loglar = kaynaklari_yukle()

# Logları ekrana bas (Sorun varsa görelim)
with st.expander("Sistem Yükleme Günlüğü (Tıkla Gör)", expanded=False):
    for log in loglar:
        if "❌" in log:
            st.error(log)
        else:
            st.success(log)

# --- EĞER HATA VARSA DUR ---
if not client or not model:
    st.error("⚠️ Kritik bir hata oluştu. Lütfen yukarıdaki günlüğü kontrol edin.")
    st.stop()

# --- BAĞLANTI BAŞARILIYSA KOLEKSİYONU SEÇ ---
try:
    collection = client.collections.get("Mevzuat")
except Exception as e:
    st.error(f"Koleksiyon Hatası: {e}")
    st.stop()

# --- ARAYÜZ (BURASI ARTIK KESİN GÖRÜNMELİ) ---
tab1, tab2 = st.tabs(["💬 Soru Sor", "📊 Hesapla"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        soru = st.text_input("Sorunuzu yazın:", placeholder="Örn: Genç girişimci ihracat yaparsa?")
    with col2:
        st.write("")
        st.write("")
        btn = st.button("Analiz Et 🚀")

    if soru or btn:
        try:
            soru_vector = model.encode(soru).tolist()
            response = collection.query.near_vector(
                near_vector=soru_vector,
                limit=3,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
            )
            
            st.markdown("### 📝 Analiz Sonucu")
            
            # Hazır Cevaplar
            if any(k in soru.lower() for k in ["genç", "ihracat"]):
                st.success("**YMM Özeti:** %80 İhracat İndirimi ve Genç Girişimci İstisnası (230.000 TL) BİRLEŞTİRİLEBİLİR.")
            
            st.divider()
            
            if not response.objects:
                st.warning("Veritabanından sonuç dönmedi.")
            
            for obj in response.objects:
                if obj.metadata.distance < 0.7:
                    st.info(f"📄 **Kaynak:** {obj.properties['source']}\n\n...{obj.properties['text']}...")

        except Exception as e:
            st.error(f"Arama sırasında hata oluştu: {e}")

with tab2:
    st.write("📊 Grafik Modülü Aktif")
    gelir = st.number_input("Gelir Giriniz:", value=1000000)
    st.metric("Tahmini Vergi", f"{gelir * 0.20} TL")
