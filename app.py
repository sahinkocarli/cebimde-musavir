import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
import pandas as pd
import plotly.express as px
import re
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Cebimde Müşavir AI", page_icon="🏦", layout="wide")

# --- MODEL VE VERİ SİSTEMİ ---
@st.cache_resource
def model_yukle():
    # En hızlı ve verimli model
    return SentenceTransformer('all-MiniLM-L6-v2')

model = model_yukle()

def verileri_hazirla(uploaded_file=None):
    # Ana Bilgi Bankası (PDF yokken bile sistemin bildiği temel gerçekler)
    banka = [
        "Genç Girişimci İstisnası: 29 yaş altı mükellefler için 3 vergilendirme dönemi boyunca yıllık kazanç istisnası sağlar.",
        "Yazılım İhracatı İndirimi: Yurt dışına verilen yazılım hizmetlerinden elde edilen kazancın %80'i vergiden indirilir.",
        "Çifte Avantaj: Yazılım ihracatı indirimi ve genç girişimci istisnası aynı anda kullanılabilir. Önce %80 indirim uygulanır.",
        "Bağış ve Yardımlar: Kurumlar ve gelir vergisi matrahından belli oranlarda indirilebilir."
    ]
    
    # Mevcut klasördeki tüm PDF'leri tara
    pdf_dosyalari = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    for dosya in pdf_dosyalari:
        try:
            with open(dosya, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        # Metni mantıklı parçalara böl
                        temiz_metin = [s.strip() for s in re.split(r'\.|\n', text) if len(s) > 50]
                        banka.extend(temiz_metin)
        except:
            continue

    if not banka:
        banka = ["Sistem henüz veri ile beslenmedi."]
            
    return banka, model.encode(banka)

# --- ARAYÜZ BAŞLIĞI ---
st.title("🏦 Cebimde Müşavir: AI Destekli Mevzuat Uzmanı")
st.markdown("---")

# Verileri yükle
bilgi_bankasi, vektorler = verileri_hazirla()

# Sekmeler
tab1, tab2 = st.tabs(["💬 Akıllı Danışman", "📊 Finansal Dashboard"])

# --- SEKME 1: DANIŞMAN ---
with tab1:
    st.subheader("🤖 Mevzuat ve Strateji Analizi")
    soru = st.text_input("Sorunuzu buraya yazın (Örn: İhracat ve genç girişimci aynı anda olur mu?)")
    
    if soru:
        v = model.encode(soru)
        # Cosine Similarity (Benzerlik ölçümü)
        benzerlik = np.dot(vektorler, v) / (np.linalg.norm(vektorler, axis=1) * np.linalg.norm(v))
        
        # En iyi 3 eşleşmeyi getir
        top_indices = np.argsort(benzerlik)[-3:][::-1]
        
        st.info("🔍 **İlgili Mevzuat Maddeleri:**")
        for i in top_indices:
            if benzerlik[i] > 0.25: # Belirli bir doğruluk eşiği
                st.write(f"📍 {bilgi_bankasi[i]}")
        
        st.success("💡 **Müşavir Notu:** Hem ihracat %80 indirimini hem de genç girişimci istisnasını aynı anda kullanabilirsiniz. Bu strateji ödenecek verginizi %90'a yakın azaltabilir.")

# --- SEKME 2: GRAFİKLER ---
with tab2:
    st.subheader("📊 Vergi ve Kazanç Analizi")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        gelir = st.number_input("Yıllık Toplam Gelir (TL)", value=1000000, step=50000)
        ihracat_mi = st.checkbox("Yazılım İhracatı mı? (%80 İndirim)", value=True)
        genc_girisimci = st.checkbox("Genç Girişimci İstisnası? (230.000 TL Muafiyet)", value=True)

    # Hesaplama Mantığı
    matrah = gelir
    if ihracat_mi:
        matrah = matrah * 0.20 # %80'i gitti
    
    if genc_girisimci:
        istisna_tutari = 230000
        matrah = max(0, matrah - istisna_tutari)
    
    vergi = matrah * 0.20 # Ortalama %20 vergi dilimi varsayımı
    net_kazanc = gelir - vergi

    with col2:
        # Grafik Verisi
        df_plot = pd.DataFrame({
            "Kategori": ["Net Kazanç", "Ödenecek Vergi"],
            "Tutar": [net_kazanc, vergi]
        })
        
        fig = px.pie(df_plot, values='Tutar', names='Kategori', 
                     title="Gelir Dağılımı (Vergi vs Net Kazanç)",
                     color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)

    # Özet Kartları
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Gelir", f"{gelir:,.0f} TL")
    c2.metric("Ödenecek Vergi", f"{vergi:,.0f} TL", delta="-70%" if ihracat_mi else "0%", delta_color="inverse")
    c3.metric("Cebine Kalan", f"{net_kazanc:,.0f} TL")

st.markdown("---")
st.caption("Cebimde Müşavir - Urla/İzmir 2026. Bilgiler resmi GİB rehberlerine dayanmaktadır.")
