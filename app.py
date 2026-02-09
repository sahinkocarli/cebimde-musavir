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
st.set_page_config(page_title="Cebimde Musavir AI", page_icon="🏦", layout="wide")

# --- MODERN DARK THEME (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 15px; border: 1px solid #30363d; }
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; border: none; height: 3.5em; width: 100%; font-weight: bold; }
    .stButton>button:hover { background-color: #2ea043; border: 1px solid #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# --- TÜRKÇE KARAKTER TEMİZLEME ---
def tr_temizle(metin):
    kaynak = "şçğüöıİĞÜÖŞÇ"
    hedef = "scguoiIGUOSC"
    tablo = str.maketrans(kaynak, hedef)
    return str(metin).translate(tablo)

# --- PDF FONKSİYONU ---
def pdf_olustur(data, yorum):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=tr_temizle("CEBIMDE MUSAVIR - FINANSAL ANALIZ RAPORU"), ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for k, v in data.items():
        pdf.cell(100, 10, txt=tr_temizle(f"{k}: {v}"), ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=tr_temizle("YAPAY ZEKA STRATEJI NOTU:"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(190, 8, txt=tr_temizle(yorum))
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# --- MODEL VE HAFIZA SİSTEMİ ---
@st.cache_resource
def model_yukle():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = model_yukle()

def verileri_hazirla(uploaded_file=None):
    # Hafıza birimleri daha net ve ayrıştırılmış hale getirildi
    banka = [
        "EV HANIMLARI MUAFİYETİ: Evde imal edilen lif, dantel, nakış gibi ürünlerin internetten satışı yıllık belirli bir tutara kadar (GVK Md. 9) vergiden muaftır. Ancak bir işyeri veya dükkan açılırsa bu muafiyet tamamen sona erer.",
        "YAZILIM İHRACATI İSTİSNASI: Sadece yurt dışındaki müşterilere verilen yazılım, tasarım ve veri depolama hizmetlerinden elde edilen kazancın %80'i vergiden istisnadır (KVK Md. 10/ğ). Yurt içi satışlar bu kapsama girmez.",
        "GENÇ GİRİŞİMCİ DESTEĞİ: 29 yaş altı şahıs işletmesi kuranlara 3 yıl boyunca vergi muafiyeti sağlanır. Limited (LTD) veya Anonim (AŞ) şirketler bu haktan yararlanamaz.",
        "GİDERLER: Personel maaşları, işyeri kirası, hammadde alımları ve işle ilgili resmi faturalı harcamalar vergi matrahından düşülebilir.",
        "KURUMLAR VERGİSİ: Şirketler (LTD ve AŞ) için standart vergi oranı 2024 yılı itibarıyla %25'tir.",
        "CARİ ORAN: 1.5 ve üzeri değerler işletmenin borç ödeme gücünün yüksek olduğunu gösterir."
    ]
    if uploaded_file:
        try:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text = page.extract_text()
                if text: banka.extend([s.strip() for s in re.split(r'(?<!\d)\.(?=\s)', text) if len(s.strip()) > 30])
        except: pass
    return banka, model.encode(banka)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- ARAYÜZ TASARIMI ---
with st.sidebar:
    st.title("🏦 Cebimde Musavir")
    st.markdown("---")
    f = st.file_uploader("Mevzuat PDF'i Yükle", type="pdf")
    if st.button("Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.info("V10.0 Web Sürümü | Sahin Kocarlı")

bilgi_bankasi, vektorler = verileri_hazirla(f)

# --- SEKMELER ---
t1, t2, t3 = st.tabs(["💬 Akıllı Danışman", "📊 Finansal Analiz", "🔮 Gelecek Simülasyonu"])

# TAB 1: SOHBET (Eşik değeri 0.40'a çekildi)
with t1:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    
    if p := st.chat_input("Vergi veya muafiyet hakkında sorun..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        
        p_v = model.encode(p)
        sims = np.dot(vektorler, p_v) / (np.linalg.norm(vektorler, axis=1) * np.linalg.norm(p_v))
        idx = np.argmax(sims)
        
        # Daha kesin cevaplar için eşik 0.40 yapıldı
        ans = bilgi_bankasi[idx] if sims[idx] > 0.40 else "Bu sorunuzla ilgili veritabanımda tam eşleşme bulamadım. Lütfen ilgili mevzuat PDF'ini yan menüden yükleyin veya sorunuzu detaylandırın."
        
        full_ans = f"**Analiz:** {ans}"
        with st.chat_message("assistant"): st.markdown(full_ans)
        st.session_state.messages.append({"role": "assistant", "content": full_ans})

# TAB 2: ANALİZ
with t2:
    st.subheader("📋 Mevcut Durum Analizi")
    c1, c2 = st.columns([1, 1])
    with c1:
        tip = st.selectbox("İşletme Tipi", ["Kurumlar Vergisi (%25)", "Gelir Vergisi (%20)"])
        gelir = st.number_input("Yıllık Ciro", value=5000000.0)
        gider = st.number_input("Yıllık Gider", value=3000000.0)
        dv = st.number_input("Dönen Varlıklar", value=1500000.0)
        kb = st.number_input("Kısa Vadeli Borçlar", value=1000000.0)
        
        if st.button("Hesapla ve Rapor Hazırla"):
            kar = gelir - gider
            vergi = kar * (0.25 if "Kurum" in tip else 0.20)
            cari = dv / kb if kb > 0 else 0
            yorum = f"Cari oranınız {cari:.2f}. " + ("Finansal yapınız güçlü." if cari >= 1.5 else "Nakit akışına dikkat edilmeli.")
            st.session_state['report_data'] = {"Tarih": datetime.now().strftime("%d/%m/%Y"), "Isletme": tip, "Kar": f"{kar:,.0f} TL", "Vergi": f"{vergi:,.0f} TL", "Cari Oran": f"{cari:.2f}"}
            st.session_state['report_comment'] = yorum
            st.success("Analiz tamamlandı. Raporu aşağıdan indirebilirsiniz.")

    with c2:
        if 'report_data' in st.session_state:
            st.metric("Ödenecek Vergi", st.session_state['report_data']['Vergi'])
            st.download_button("📜 Raporu PDF Olarak İndir", pdf_olustur(st.session_state['report_data'], st.session_state['report_comment']), "YMM_Analiz_Raporu.pdf")
            
            # Cari Oran Grafiği
            fig = go.Figure(go.Indicator(mode="gauge+number", value=float(st.session_state['report_data']['Cari Oran']), title={'text': "Borç Ödeme Gücü (Cari Oran)"},
                gauge={'axis':{'range':[0,3]}, 'steps':[{'range':[0,1],'color':"red"},{'range':[1,2],'color':"orange"},{'range':[2,3],'color':"green"}]}))
            st.plotly_chart(fig, use_container_width=True)

# TAB 3: TAHMİNLEME
with t3:
    st.subheader("🔮 Gelecek Simülasyonu")
    if 'report_data' in st.session_state:
        d = st.session_state['report_data']
        oran = st.slider("Gelecek Ay Beklenen Satış Artışı (%)", -50, 100, 20)
        eski_gelir = float(d['Kar'].replace(' TL', '').replace(',', '')) + 3000000.0 # Tahmini gider ekli
        yeni_kar = (eski_gelir * (1 + oran/100)) - 3000000.0
        yeni_vergi = yeni_kar * (0.25 if "Kurum" in d['Isletme'] else 0.20)
        
        st.write(f"Satışlar %{oran} artarsa, tahmini yeni vergi yükü: **{max(0, yeni_vergi):,.0f} TL** olacaktır.")
        fig_bar = px.bar(x=["Mevcut Vergi", "Yeni Vergi"], y=[float(d['Vergi'].replace(' TL', '').replace(',', '')), yeni_vergi], color_discrete_sequence=['#238636'])
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Lütfen önce 'Finansal Analiz' sekmesinden hesaplama yapın.")
