import streamlit as st
import google.generativeai as genai
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Cebimde Müşavir", page_icon="🧾", layout="centered")

# --- API ANAHTARI KONTROLÜ ---
# Streamlit Secrets üzerinden Google API Key'i alıyoruz
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except Exception as e:
    st.error("🚨 HATA: Google API Key bulunamadı! Lütfen Streamlit ayarlarından Secrets kısmına ekleyin.")
    st.stop()

# Modeli Seç (Gemini 1.5 Flash - Hızlı ve Ucuz)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- VERİLERİ (INDEX) YÜKLE ---
@st.cache_resource
def load_index():
    try:
        if not os.path.exists("index.pkl"):
            return None, None, None, None
        
        with open("index.pkl", "rb") as f:
            data = pickle.load(f)
        return data["documents"], data["filenames"], data["vectorizer"], data["tfidf_matrix"]
    except Exception as e:
        st.error(f"İndeks dosyası yüklenirken hata oluştu: {e}")
        return None, None, None, None

documents, filenames, vectorizer, tfidf_matrix = load_index()

if documents is None:
    st.warning("⚠️ Sistem henüz hazır değil. Lütfen önce belgelerin işlenmesini bekleyin (build_index.py).")
    st.stop()

# --- GEMINI'YE DANIŞMA FONKSİYONU ---
def ask_gemini_advisor(soru, context_text):
    prompt = f"""
    Sen Türkiye vergi mevzuatına hakim, uzman bir "Dijital Mali Müşavirsin".
    
    GÖREVİN:
    Aşağıda sana verilen "RESMİ KAYNAK METİNLERİ" (CONTEXT) kullanarak, vatandaşın sorusunu net, doğru ve profesyonelce cevapla.
    
    KURALLAR:
    1. Sadece aşağıdaki KAYNAK METİNLERdeki bilgiyi kullan. Kendi kafandan kanun uydurma.
    2. Cevabın sohbet havasında olsun ama ciddiyetini koru.
    3. Varsa önemli tutarları (TL), oranları (%) ve tarihleri madde madde listele.
    4. Eğer metinlerde cevap yoksa "Bu konuda yüklenen resmi rehberlerde net bir bilgi bulamadım." de.
    
    RESMİ KAYNAK METİNLER:
    {context_text}
    
    VATANDAŞIN SORUSU:
    {soru}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Üzgünüm, şu an cevap üretemiyorum. Lütfen tekrar deneyin."

# --- ARAYÜZ (FRONTEND) ---
st.title("🧾 Cebimde Müşavir AI")
st.caption("Resmi GİB Rehberleri ile eğitilmiş Yapay Zeka Asistanı")

# Soru Kutusu
user_query = st.text_input("Mevzuat sorunuzu yazın:", placeholder="Örn: Kira geliri istisnası ne kadar?")

if st.button("Danış") and user_query:
    with st.spinner("🔍 Mevzuat taranıyor ve Müşavir yorumluyor..."):
        # 1. Hızlı Arama (TF-IDF)
        query_vec = vectorizer.transform([user_query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # En iyi 3 sonucu getir
        top_indices = scores.argsort()[-3:][::-1]
        
        found_docs = []
        context_data = ""
        has_relevant_data = False
        
        for idx in top_indices:
            score = scores[idx]
            if score > 0.05: # Çok alakasızları filtrele
                has_relevant_data = True
                doc_text = documents[idx]
                fname = filenames[idx]
                
                # Belge ismini temizle (arsiv_... kısmını at)
                clean_name = fname.replace("arsiv_fileadmin_", "").replace("arsiv_onceki-dokumanlar_", "")
                
                found_docs.append(f"📄 {clean_name}")
                context_data += f"\n--- KAYNAK: {clean_name} ---\n{doc_text}\n"

        if has_relevant_data:
            # 2. Gemini'ye Gönder (Yorumlama)
            ai_response = ask_gemini_advisor(user_query, context_data)
            
            # 3. Sonucu Göster
            st.markdown("### 🤖 Müşavir Cevabı:")
            st.info(ai_response)
            
            # 4. Kaynakları Göster
            with st.expander("📚 Kullanılan Resmi Kaynaklar"):
                for doc in found_docs:
                    st.write(doc)
                st.text_area("Ham Metin Verisi", context_data, height=150)
        else:
            st.warning("Bu konuyla ilgili mevzuat rehberlerinde eşleşen bir bilgi bulunamadı. Farklı kelimelerle aramayı deneyin.")

# Alt Bilgi
st.markdown("---")
st.markdown("⚠️ *Bu sistem bilgilendirme amaçlıdır. Resmi beyanname vermeden önce mutlaka gerçek bir Mali Müşavir ile görüşünüz.*")
