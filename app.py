import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from datetime import datetime
from fpdf import FPDF

# --- ULTRA-PRO SAYFA AYARLARI ---
st.set_page_config(page_title="YMM AI WEB", page_icon="🌐", layout="wide")

# --- MODERN DARK THEME (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 15px; border: 1px solid #30363d; }
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; border: none; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

# --- TÜRKÇE KARAKTER TEMİZLEME ---
def tr_temizle(metin):
    kaynak = "şçğüöıİĞÜÖŞÇ"
    hedef = "scguoiIGUOSC"
    tablo = str.maketrans(kaynak, hedef)
    return str(metin).translate(tablo)

# --- GELİŞMİŞ PDF FONKSİYONU (TABLOLU) ---
def pdf_olustur(data, yorum):
    pdf = FPDF()
    pdf.add_page()
    
    # Başlık
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(31, 111, 235)
    pdf.cell(200, 20, txt=tr_temizle("STRATEJIK FINANSAL DENETIM RAPORU"), ln=True, align='C')
    
    # Çizgi
    pdf.set_draw_color(48, 54, 61)
    pdf.line(10, 35, 200, 35)
    pdf.ln(10)
    
    # Veri Tablosu
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 242, 246)
    pdf.set_text_color(0, 0, 0)
    
    headers = ["Kalem", "Deger"]
    rows = [
        ["Rapor Tarihi", data['tarih']],
        ["Isletme Tipi", data['tip']],
        ["Yillik Gelir", f"{data['gelir']:,.0f} TL"],
        ["Yillik Gider", f"{data['gider']:,.0f} TL"],
        ["Net Kar", f"{data['kar']:,.0f} TL"],
        ["Hesaplanan Vergi", f"{data['vergi']:,.0f} TL"],
        ["Cari Oran", f"{data['cari']:.2f}"]
    ]
    
    for header in headers:
        pdf.cell(95, 10, tr_temizle(header), 1, 0, 'C', True)
    pdf.ln()
    
    pdf.set_font("Arial", size=11)
    for row in rows:
        pdf.cell(95, 10, tr_temizle(row[0]), 1)
        pdf.cell(95, 10, tr_temizle(row[1]), 1)
        pdf.ln()
    
    # Yapay Zeka Yorumu
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(35, 134, 54)
    pdf.cell(200, 10, txt=tr_temizle("YAPAY ZEKA ANALIZ NOTU:"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(190, 8, txt=tr_temizle(yorum))
    
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# --- MODEL VE HAFIZA SİSTEMİ ---
@st.cache_resource
def model_yukle():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = model_yukle()

def verileri_hazirla(uploaded_file=None):
    banka = [
        "Yazılım İhracatı İstisnası: Yurt dışına verilen hizmet kazancının %80'i vergiden istisnadır.",
        "Dükkan Açma: İşyeri açılması durumunda esnaf muafiyeti sona erer.",
        "Giderler: Personel, kira ve hammadde harcamaları vergiden düşülebilir.",
        "Genç Girişimci: 29 yaş altı şahıs şirketlerine 3 yıl destek sağlanır."
    ]
    if uploaded_file:
        try:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text = page.extract_text()
                if text: banka.extend([s.strip() for s in re.split(r'\.', text) if len(s) > 30])
        except: pass
    return banka, model.encode(banka)

if "messages" not in st.session_state: st.session_state.messages = []

# --- ARAYÜZ ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=100)
    st.title("YMM AI Control")
    f = st.file_uploader("Mevzuat Yükle (PDF)", type="pdf")
    if st.button("Hafızayı Temizle"): st.session_state.messages = []; st.rerun()

bilgi_bankasi, vektorler = verileri_hazirla(f)

# --- TABS ---
t1, t2, t3 = st.tabs(["💬 Danışman", "📊 Analiz & Rapor", "📈 Tahminleme"])

with t1:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    
    if p := st.chat_input("Sorunuzu yazın..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        
        v = model.encode(p)
        s = np.dot(vektorler, v) / (np.linalg.norm(vektorler, axis=1) * np.linalg.norm(v))
        idx = np.argmax(s)
        res = bilgi_bankasi[idx] if s[idx] > 0.3 else "Lütfen detaylı bilgi için PDF yükleyin."
        with st.chat_message("assistant"): st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})

with t2:
    col1, col2 = st.columns([1, 1])
    with col1:
        tip = st.selectbox("İşletme Türü", ["Kurumlar Vergisi (%25)", "Gelir Vergisi (%20)"])
        gelir = st.number_input("Yıllık Ciro (TL)", value=1000000.0)
        gider = st.number_input("Yıllık Gider (TL)", value=600000.0)
        dv = st.number_input("Kasadaki Nakit/Varlık", value=200000.0)
        kb = st.number_input("Kısa Vadeli Borçlar", value=100000.0)
        
        if st.button("Hesapla ve PDF Hazırla"):
            oran = 0.25 if "Kurum" in tip else 0.20
            kar = gelir - gider
            vergi = kar * oran if kar > 0 else 0
            cari = dv / kb if kb > 0 else 0
            yorum = f"Analiz Sonucu: {'Güçlü likidite.' if cari > 1.5 else 'Nakit yönetimine dikkat.'} Vergi yükü %{oran*100} üzerinden hesaplanmıştır."
            st.session_state['data'] = {"tarih": datetime.now().strftime("%d/%m/%Y"), "tip": tip, "gelir": gelir, "gider": gider, "kar": kar, "vergi": vergi, "cari": cari}
            st.session_state['yorum'] = yorum
            st.success("Raporunuz hazırlandı!")

    with col2:
        if 'data' in st.session_state:
            d = st.session_state['data']
            st.metric("Ödenecek Toplam Vergi", f"{d['vergi']:,.0f} TL")
            st.download_button("📜 Profesyonel Raporu İndir (PDF)", pdf_olustur(d, st.session_state['yorum']), "YMM_Raporu.pdf")