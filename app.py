import streamlit as st
import time

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hız Testi", page_icon="⚡", layout="wide")

st.title("🏦 Cebimde Müşavir: Pro")
st.write("✅ 1. Başlık yazıldı. Kod çalışmaya başladı.")

# --- AĞIR İŞLEMLERİ TAKİP ETME ---
durum_kutusu = st.empty() # Buraya anlık durum yazacağız

def sistemi_baslat():
    # ADIM 1: KÜTÜPHANELER
    durum_kutusu.info("⏳ 2. Weaviate kütüphanesi çağırılıyor...")
    import weaviate
    st.write("✅ Weaviate kütüphanesi yüklendi.")
    
    durum_kutusu.info("⏳ 3. Yapay Zeka (SentenceTransformers) kütüphanesi çağırılıyor (En Ağır Kısım)...")
    # Bu satır sunucuyu en çok yoran kısımdır
    from sentence_transformers import SentenceTransformer
    st.write("✅ Yapay Zeka kütüphanesi hafızaya alındı.")
    
    # ADIM 2: MODEL İNDİRME
    durum_kutusu.info("⏳ 4. Model (MiniLM) indiriliyor...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("✅ Model başarıyla indirildi ve hazır.")
    
    # ADIM 3: BULUT BAĞLANTISI
    durum_kutusu.info("⏳ 5. Weaviate Bulutuna bağlanılıyor...")
    try:
        client = weaviate.connect_to_wcs(
            cluster_url="https://yr17vqmwtmwdko2v5kqeda.c0.europe-west3.gcp.weaviate.cloud",
            auth_credentials=weaviate.auth.AuthApiKey("TUZ0Sm9MMGlFeWtsTGtHUF8vYkpQMm02SjRIYkRtblBhSi83cHNHcVNOVWpzdHVRZEdMV2N5dTMrdGlFPV92MjAw")
        )
        st.write("✅ Bulut bağlantısı başarılı!")
        return client, model
    except Exception as e:
        st.error(f"❌ Bağlantı Hatası: {str(e)}")
        return None, None

# İşlemi Başlat
if st.button("🚀 Sistemi Başlat (Tıkla)"):
    client, model = sistemi_baslat()
    
    if client and model:
        st.success("🎉 SİSTEM TAMAMEN AÇILDI! ARTIK HIZLI ÇALIŞACAK.")
        durum_kutusu.empty()
        
        # Test Sorusu
        soru = st.text_input("Soru Sor:", "Genç girişimci istisnası nedir?")
        if st.button("Analiz Et"):
            collection = client.collections.get("Mevzuat")
            vector = model.encode(soru).tolist()
            response = collection.query.near_vector(near_vector=vector, limit=1)
            st.write(response.objects[0].properties['text'])

else:
    st.info("👆 Yukarıdaki butona basarak yüklemeyi başlatın.")
