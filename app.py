import streamlit as st
import os
import requests
import json
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Cebimde Müşavir", page_icon="🧾", layout="centered")

# --- API ANAHTARI KONTROLÜ ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🚨 HATA: Streamlit Secrets ayarlarında 'GOOGLE_API_KEY' bulunamadı!")
    st.stop()

# --- GEMINI'YE BAĞLANMA (DİREKT REST API) ---
def call_google_api(model_name, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response

def ask_google_smartly(prompt):
    # 1. ÖNCE: En hızlı model (Gemini 1.5 Flash) dene
    response = call_google_api("gemini-1.5-flash", prompt)
    
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    
    # 2. EĞER HATA VERİRSE (404 vs): Klasik model (Gemini Pro) dene
    else:
        # st.toast("Flash modeli yanıt vermedi, Pro modeline geçiliyor...") # Bilgi ver
        response = call_google_api("gemini-pro", prompt)
        
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"🚨 HATA: Hiçbir model çalışmadı. Google Hatası ({response.status_code}): {response.text}"

# --- PDF OKUMA VE HAFIZA ---
@st.cache_resource(show_spinner=False)
def create_knowledge_base():
    documents = []
    filenames = []
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdf_files:
        return None, None, None, None

    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, pdf_file in enumerate(pdf_files):
        try:
            status_text.text(f"📚 İşleniyor: {pdf_file}...")
            reader = pypdf.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t: text += t + "\n"
            documents.append(text)
            filenames.append(pdf_file)
        except:
            pass
        progress_bar.progress((i + 1) / len(pdf_files))

    status_text.empty()
    progress_bar.empty()

    if documents:
        vectorizer = TfidfVectorizer(stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(documents)
        return documents, filenames, vectorizer, tfidf_matrix
    else:
        return None, None, None, None

# --- SİSTEM BAŞLANGICI ---
with st.spinner("🚀 Sistem başlatılıyor..."):
    documents, filenames, vectorizer, tfidf_matrix = create_knowledge_base()

if not documents:
    st.error("⚠️ PDF dosyası bulunamadı!")
    st.stop()

# --- MÜŞAVİR FONKSİYONU ---
def ask_advisor(soru, context_text):
    prompt = f"""
    Sen uzman bir Mali Müşavirsin. Aşağıdaki resmi kaynakları kullanarak vatandaşa cevap ver.
    
    KAYNAKLAR:
    {context_text}
    
    SORU:
    {soru}
    
    Cevabı Türkçe ver. Kaynaklarda bilgi yoksa "Bilgi yok" de.
    """
    return ask_google_smartly(prompt)

# --- ARAYÜZ ---
st.title("🧾 Cebimde Müşavir AI")
st.caption(f"📚 {len(filenames)} adet kaynak yüklendi.")

user_query = st.text_input("Sorunuzu yazın:", placeholder="Örn: Kira geliri istisnası ne kadar?")

if st.button("Danış") and user_query:
    with st.spinner("🔍 Müşavir düşünüyor..."):
        query_vec = vectorizer.transform([user_query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[-3:][::-1]
        
        found_docs = []
        context_data = ""
        has_data = False
        
        for idx in top_indices:
            if scores[idx] > 0.05:
                has_data = True
                fname = filenames[idx].replace("arsiv_fileadmin_", "").replace(".pdf", "")
                found_docs.append(f"📄 {fname}")
                context_data += f"\n--- KAYNAK: {fname} ---\n{documents[idx][:4000]}...\n"

        if has_data:
            ai_response = ask_advisor(user_query, context_data)
            st.markdown("### 🤖 Cevap:")
            if "🚨" in ai_response:
                st.error(ai_response)
            else:
                st.info(ai_response)
            
            with st.expander("Kaynaklar"):
                for doc in found_docs:
                    st.write(doc)
        else:
            st.warning("Bu konuda bilgi bulunamadı.")
