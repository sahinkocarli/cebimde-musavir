import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
import pandas as pd
import plotly.express as px
import re
import os
from datetime import datetime

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Cebimde Müşavir AI", page_icon="🏦", layout="wide")

# --- MODEL VE OTOMATİK PDF TARAMA ---
@st.cache_resource
def model_yukle():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = model_yukle()

def verileri_hazirla(uploaded_file=None):
    # Ana Bilgi Bankası
    banka = [
        "Yazılım İhracatı: Kazancın %80'i vergiden istisnadır. Genç girişimci desteğiyle birleşebilir.",
        "Genç Girişimci: 29 yaş altı için 3 yıl boyunca yıllık 230 bin TL (2024 yılı için) kazanç istisnası vardır."
    ]
    
    # GİB'den indirdiğin tüm PDF'leri otomatik oku
    current_dir = os.listdir('.')
    pdf_dosyalari = [f for f in current_dir if f.endswith('.pdf')]
    
    for dosya in pdf_dosyalari:
        try:
            with open(dosya, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        # Metni mantıklı parçalara böl (40 karakterden uzun cümleler)
                        temiz_metin = [s.strip() for s in re.split(r'\.|\n', text) if len(s) > 40]
                        banka.extend(temiz_metin)
        except:
            continue

    if uploaded_file:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text: banka.extend([s.strip() for s in re.split(r'\.|\n', text) if len(s) > 40])
            
    return banka, model.encode(banka)

# --- ARAYÜZ ---
st.title("🏦 Cebimde Müşavir: 2026 Mevzuat Uzmanı")
st.markdown("---")

bilgi_bankasi, vektorler = verileri_hazirla()

tab1, tab2 = st.tabs(["💬 Mevzuat Danışmanı", "📊 Vergi Analizi"])

with tab1:
    st.subheader("🤖 GİB Rehberlerine Göre Analiz")
    soru = st.text_input("Örn: Genç girişimci ve yazılım ihracatı aynı anda olur mu?")
    
    if soru:
        v = model.encode(soru)
        benzerlik = np.dot(vektorler, v) / (np.linalg.norm(vektorler, axis=1) * np.linalg.norm(v))
        en_yakin_index = np.argmax(benzerlik)
        
        st.info(f"🔍 **Mevzuat Kaydı:** {bilgi_bankasi[en_yakin_index]}")
        st.write("---")
        st.caption("Not: Bu cevap GİB rehberlerindeki en yakın maddeye göre oluşturulmuştur.")

with tab2:
    st.subheader("🔢 Hızlı Hesaplama")
    gelir = st.number_input("Yıllık Tahmini Gelir (TL)", value=1000000)
    ihracat_mi = st.checkbox("Yazılım İhracatı mı? (%80 İstisna)")
    
    matrah = gelir * 0.20 if ihracat_mi else gelir
    vergi = matrah * 0.20
    
    st.metric("Tahmini Ödenecek Vergi", f"{vergi:,.2f} TL")
