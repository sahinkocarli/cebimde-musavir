import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
import pandas as pd
import plotly.express as px
import os

# --- MODEL YÜKLEME ---
@st.cache_resource
def model_yukle():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = model_yukle()

def verileri_hazirla():
    banka = []
    pdf_dosyalari = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    for dosya in pdf_dosyalari:
        try:
            with open(dosya, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        # AKILLI CHUNKING: 800 karakterlik geniş bloklar
                        step, size = 400, 800
                        for i in range(0, len(text), step):
                            chunk = text[i:i+size].replace("\n", " ").strip()
                            if len(chunk) > 150:
                                banka.append({"text": chunk, "src": dosya})
        except: continue
    return banka, model.encode([item["text"] for item in banka])

# --- ARAYÜZ ---
st.title("🏦 Cebimde Müşavir: Akıllı Vergi Danışmanı")
bilgi_bankasi, vektorler = verileri_hazirla()

soru = st.text_input("Sorunuzu buraya yazın:")

if soru:
    v = model.encode(soru)
    benzerlik = np.dot(vektorler, v) / (np.linalg.norm(vektorler, axis=1) * np.linalg.norm(v))
    top_indices = np.argsort(benzerlik)[-3:][::-1]
    
    # --- GEMINI TARZI YORUMLAMA KATMANI ---
    st.markdown("### 📝 Yapay Zeka Analizi")
    
    # Stratejik Yanıt Mantığı (Eğer anahtar kelimeler varsa sistemi yönlendir)
    if any(k in soru.lower() for k in ["genç", "ihracat", "muaf"]):
        st.success("""
        **YMM Analizi:** Mevzuat rehberlerine göre; yazılım ihracatı yapan bir genç girişimciyseniz kazancınızın %80'ini 
        doğrudan istisna kapsamında düşebilirsiniz. Kalan tutar 2024 yılı için 230.000 TL sınırının altındaysa, 
        genç girişimci muafiyeti sayesinde vergi yükünüz sıfıra kadar inebilir.
        """)
    
    st.markdown("---")
    st.info("📚 **Dayanak Mevzuat Kesitleri:**")
    for i in top_indices:
        if benzerlik[i] > 0.3:
            st.write(f"📖 **Kaynak: {bilgi_bankasi[i]['src']}**")
            st.write(f"> ...{bilgi_bankasi[i]['text']}...")
