import streamlit as st
import time

# --- 1. SİTEYİ HEMEN AÇ ---
st.set_page_config(page_title="Debug Modu", page_icon="🐞")
st.title("🐞 Hata Ayıklama Modu")
st.success("✅ Site şu an açık! (Bunu görüyorsan sunucu çalışıyor)")

st.info("Aşağıdaki butona bastığında ağır kütüphaneleri hafızaya çağırmayı deneyeceğiz.")

# --- 2. FONKSİYON İÇİNDE YÜKLEME (EN ÖNEMLİ KISIM) ---
def kutuphaneleri_yukle(status_box):
    try:
        # ADIM 1
        t1 = time.time()
        status_box.write("⏳ 1. 'weaviate' kütüphanesi çağırılıyor...")
        import weaviate
        status_box.write(f"✅ Weaviate geldi ({time.time()-t1:.2f} sn)")
        
        # ADIM 2 (EN RİSKLİ YER)
        t2 = time.time()
        status_box.write("⏳ 2. 'sentence-transformers' çağırılıyor (En Ağır İşlem)...")
        from sentence_transformers import SentenceTransformer
        status_box.write(f"✅ Yapay Zeka Motoru yüklendi! ({time.time()-t2:.2f} sn)")
        
        # ADIM 3 (MODEL İNDİRME)
        t3 = time.time()
        status_box.write("⏳ 3. Model (MiniLM) indiriliyor...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        status_box.write(f"✅ Model Hazır! ({time.time()-t3:.2f} sn)")
        
        return True, model, weaviate
        
    except Exception as e:
        st.error(f"❌ KRİTİK HATA: {str(e)}")
        return False, None, None

# --- 3. TETİKLEYİCİ BUTON ---
if st.button("🚀 Motoru Başlat"):
    # Durum kutusu oluştur
    status = st.status("Yükleme İşlemi Başladı...", expanded=True)
    
    basari, model, weaviate_lib = kutuphaneleri_yukle(status)
    
    if basari:
        status.update(label="🎉 BAŞARILI! Sistem Çalışıyor.", state="complete", expanded=False)
        st.balloons()
        
        # Basit bir test yapalım
        st.divider()
        st.write("🤖 **Hızlı Test:**")
        soru = st.text_input("Bir şey yaz:", "Vergi")
        if soru:
            vec = model.encode(soru).tolist()
            st.write(f"Vektör boyutu: {len(vec)} (Çalışıyor!)")
    else:
        status.update(label="❌ Yükleme Başarısız Oldu", state="error")
