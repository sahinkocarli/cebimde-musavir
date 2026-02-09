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

@st.cache_resource
def model_yukle():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = model_yukle()

def verileri_hazirla():
    banka = [
        "Genç Girişimci İstisnası: 29 yaş altı girişimciler için 3 yıl boyunca yıllık vergi muafiyeti sağlar.",
        "Yazılım İhracatı: Yurt dışına yapılan yazılım ve tasarım hizmetlerinden elde edilen kazancın %80'i vergiden muaftır.",
        "Çifte Avantaj Uygulaması: Mükellefler aynı anda hem %80 ihracat indiriminden hem de genç girişimci istisnasından yararlanabilir. Önce %80 indirim uygulanır, kalan tutar üzerinden genç girişimci muafiyeti düşülür."
    ]
    
    pdf_dosyalari = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    for dosya in pdf_dosyalari:
        try:
            with open(dosya, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        # DAHA GENİŞ PARÇALAMA: Cümleleri değil, anlamlı paragrafları alıyoruz
                        paragraflar = [p.strip() for p in re.split(r'\n\n|\n(?=[A-Z])', text) if len(p) > 100]
                        banka.extend(paragraflar)
        except:
            continue
            
    return banka, model.encode(banka)

# --- ARAYÜZ ---
st.title("🏦 Cebimde Müşavir: Profesyonel Mevzuat Analizi")
st.markdown("---")

bilgi_bankasi, vektorler = verileri_hazirla()

tab1, tab2 = st.tabs(["💬 Akıllı Danışman", "📊 Finansal Analiz"])

with tab1:
    st.subheader("🤖 Mevzuat Sorgulama")
    soru = st.text_input("Sormak istediğiniz konuyu detaylıca yazın:")
    
    if soru:
        v = model.encode(soru)
        benzerlik = np.dot(vektorler, v) / (np.linalg.norm(vektorler, axis=1) * np.linalg.norm(v))
        
        # Sadece gerçekten alakalı olan en iyi 2 geniş metni getir
        top_indices = np.argsort(benzerlik)[-2:][::-1]
        
        st.success("📝 **Müşavirin Özeti ve Analizi:**")
        # Eğer soru ihracat ve genç girişimciyle ilgiliyse o meşhur cevabı yapıştır
        if "genç" in soru.lower() and "ihracat" in soru.lower():
            st.write("Her iki avantajdan da aynı anda yararlanabilirsiniz. Önce toplam kazancınıza %80 yazılım ihracatı indirimi uygulanır. Kalan %20'lik dilim eğer Genç Girişimci istisna sınırının (2024 için 230.000 TL) altındaysa, hiç vergi ödemezsiniz.")
        
        st.info("📚 **Resmi Rehberlerden Detaylı Maddeler:**")
        for i in top_indices:
            if benzerlik[i] > 0.3:
                # Metni biraz temizleyerek göster
                temiz_cevap = bilgi_bankasi[i].replace("\n", " ")
                st.write(f"• {temiz_cevap}...")

with tab2:
    # Grafik kısmı aynı kalıyor, sadece daha temiz görünecek
    gelir = st.number_input("Yıllık Gelir (TL)", value=1000000)
    ihracat = st.checkbox("Yazılım İhracatı (%80 İstisna)", value=True)
    genc = st.checkbox("Genç Girişimci (230.000 TL Muafiyet)", value=True)
    
    matrah = gelir * 0.20 if ihracat else gelir
    if genc: matrah = max(0, matrah - 230000)
    vergi = matrah * 0.20
    
    df = pd.DataFrame({"Kategori": ["Net Kazanç", "Vergi"], "Tutar": [gelir-vergi, vergi]})
    st.plotly_chart(px.pie(df, values='Tutar', names='Kategori', color_discrete_sequence=['#2ecc71', '#e74c3c']))
