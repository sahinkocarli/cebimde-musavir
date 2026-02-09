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

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Cebimde Musavir AI", page_icon="🏦", layout="wide")

# --- MODERN TASARIM (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 15px; border: 1px solid #30363d; }
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; border: none; height: 3.5em; width: 100%; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- TÜRKÇE KARAKTER TEMİZLEME ---
def tr_temizle(metin):
    kaynak = "şçğüöıİĞÜÖŞÇ"
    hedef = "scguoiIGUOSC"
    tablo = str.maketrans(kaynak, hedef)
    return str(metin).translate(tablo)

# --- PDF RAPOR FONKSİYONU ---
def pdf_olustur(data, yorum):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=tr_temizle("CEBIMDE MUSAVIR - ANALIZ RAPORU"), ln=True, align='C')
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

# --- MODEL VE OTOMATİK HAFIZA SİSTEMİ ---
@st.cache_resource
def model_yukle():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = model_yukle()

def verileri_hazirla():
    # Temel Mevzuat Bilgileri
    banka = [
        "EV HANIMLARI MUAFİYETİ: Evde imal edilen lif, dantel, nakış gibi ürünlerin internetten satışı yıllık belirli bir tutara kadar (GVK Md. 9) vergiden muaftır. Dükkan açılırsa muafiyet biter.",
        "YAZILIM İHRACATI: Yurt dışına verilen yazılım hizmet kazancının %80'i vergiden istisnadır (KVK Md. 10/ğ).",
        "GENÇ GİRİŞİMCİ: 29 yaş altı şahıs işletmelerine 3 yıl vergi istisnası sağlanır.",
        "KURUMLAR VERGİSİ: Şirketler için standart oran %25'tir."
    ]
    
    # GİZLİ TARAMA: GitHub klasöründeki tüm PDF'leri otomatik oku
    for dosya in os.listdir("."):
        if dosya.endswith(".pdf"):
            try:
                okuyucu = PdfReader(dosya)
                for sayfa in okuyucu.pages:
                    metin = sayfa.extract_text()
                    if metin:
                        # Metni cümlelere bölüp hafızaya ekle
                        cumleler = [s.strip() for s in re.split(r'(?<!\d)\.(?=\s)', metin) if len(s.strip()) > 40]
                        banka.extend(cumleler)
            except Exception as e:
                print(f"Hata: {dosya} okunamadı. {e}")
                
    return banka, model.encode(banka)

# --- ANA PROGRAM ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🏦 Cebimde Musavir")
    st.markdown("---")
    st.success("🤖 Mevzuat Hafızası Aktif")
    st.info("Sistem, yüklü olan tüm PDF belgelerini analiz ederek cevap vermektedir.")
    if st.button("Sohbeti Sıfırla"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.write("v11.0 | Profesyonel Sürüm")

bilgi_bankasi, vektorler = verileri_hazirla()

t1, t2, t3 = st.tabs(["💬 Akıllı Danışman", "📊 Finansal Analiz", "🔮 Gelecek Tahmini"])

with t1:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    
    if p := st.chat_input("Sorunuzu yazın..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        
        soru_v = model.encode(p)
        benzerlikler = np.dot(vektorler, soru_v) / (np.linalg.norm(vektorler, axis=1) * np.linalg.norm(soru_v))
        en_iyi_idx = np.argmax(benzerlikler)
        
        cevap = bilgi_bankasi[en_iyi_idx] if benzerlikler[en_iyi_idx] > 0.40 else "Bu konuda güncel mevzuat belgesi bulunamadı. Lütfen yeni bir PDF ekleyerek sistem hafızasını güncelleyin."
        
        final_cevap = f"**YMM Analizi:** {cevap}"
        with st.chat_message("assistant"): st.markdown(final_cevap)
        st.session_state.messages.append({"role": "assistant", "content": final_cevap})

with t2:
    st.subheader("📋 Bilanço ve Vergi Analizi")
    c1, c2 = st.columns(2)
    with c1:
        tip = st.selectbox("İşletme Tipi", ["Kurumlar Vergisi (%25)", "Gelir Vergisi (%20)"])
        gelir = st.number_input("Yıllık Gelir", value=1000000.0)
        gider = st.number_input("Yıllık Gider", value=600000.0)
        if st.button("Hesapla ve Raporla"):
            kar = gelir - gider
            vergi = kar * (0.25 if "Kurum" in tip else 0.20)
            st.session_state['data'] = {"Tarih": datetime.now().strftime("%d/%m/%Y"), "Isletme": tip, "Net Kar": f"{kar:,.0f} TL", "Hesaplanan Vergi": f"{vergi:,.0f} TL"}
            st.success("Analiz tamamlandı!")
    with c2:
        if 'data' in st.session_state:
            st.metric("Tahmini Vergi", st.session_state['data']['Hesaplanan Vergi'])
            st.download_button("📜 Raporu PDF İndir", pdf_olustur(st.session_state['data'], "Mevcut finansal verileriniz üzerinden vergi planlaması yapılmıştır."), "Analiz_Raporu.pdf")

with t3:
    st.info("Finansal Analiz sekmesinde hesaplama yaptıktan sonra burayı kullanabilirsiniz.")
    if 'data' in st.session_state:
        artis = st.slider("Satış Artış Tahmini (%)", 0, 100, 20)
        st.write(f"Satışlarınız %{artis} artarsa vergi yükünüzün değişimi grafikte gösterilmiştir.")
