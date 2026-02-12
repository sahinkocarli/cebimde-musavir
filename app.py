import streamlit as st
import google.generativeai as genai
import os
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Cebimde Müşavir", page_icon="🧾", layout="centered")

# --- API KURULUMU VE OTOMATİK MODEL SEÇİMİ ---
try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
    else:
        st.error("🚨 HATA: Secrets içinde GOOGLE_API_KEY bulunamadı.")
        st.stop()

    # SİHİRLİ KISIM: Google'a soruyoruz, hangi modeller açık?
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
    
    # En iyiden başlayarak seçelim
    target_models = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
    active_model = None
    
    # Hedeflediklerimizden biri var mı?
    for target in target_models:
        if target in available_models:
            active_model = target
            break
            
    # Yoksa listenin başındakini al
    if not active_model and available_models:
        active_model = available_models[0]
        
    if not active_model:
        st.error(f"🚨 HATA: Bu anahtar ile hiçbir metin modeline erişilemiyor. (Liste boş)")
        st.stop()
        
    # Modeli Başlat
    model = genai.GenerativeModel(active_model)

except Exception as e:
    st.error(f"🚨 API Bağlantı Hatası: {str(e)}")
    st.stop()

# --- PDF OKUMA SİSTEMİ ---
@st.cache_resource(show_spinner=False)
def create_knowledge_base():
    documents = []
    filenames = []
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdf_files: return None, None, None, None

    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, pdf_file in enumerate(pdf_files):
        try:
            status_text.text(f"📚 Okunuyor: {pdf_file}...")
            reader = pypdf.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t: text += t + "\n"
            documents.append(text)
            filenames.append(pdf_file)
        except: pass
        progress_bar.progress((i + 1) / len(pdf_files))

    status_text.empty()
    progress_bar.empty()

    if documents:
        vectorizer = TfidfVectorizer(stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(documents)
        return documents, filenames, vectorizer, tfidf_matrix
    else:
        return None, None, None, None

# --- SİSTEM BAŞLATILIYOR ---
with st.spinner("🚀 Sistem başlatılıyor..."):
    documents, filenames, vectorizer, tfidf_matrix = create_knowledge_base()

if not documents:
    st.error("⚠️ Klasörde PDF bulunamadı! Lütfen GitHub'a dosya yükleyin.")
    st.stop()

# --- MÜŞAVİR FONKSİYONU ---
def ask_advisor(soru, context):
    prompt = f"""
    Sen uzman bir Mali Müşavirsin. Sadece aşağıdaki kaynakları kullan.
    
    KAYNAKLAR:
    {context}
    
    SORU: {soru}
    Cevabı Türkçe ver.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"🚨 HATA: {str(e)}"

# --- ARAYÜZ ---
st.title("🧾 Cebimde Müşavir AI")
st.success(f"✅ Bağlandı! Kullanılan Model: {active_model}") # Çalışan modeli ekranda göreceğiz

user_query = st.text_input("Sorunuz:", placeholder="Örn: Kira istisnası ne kadar?")

if st.button("Danış") and user_query:
    with st.spinner("🔍 İnceleniyor..."):
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
            response = ask_advisor(user_query, context_data)
            st.info(response)
            with st.expander("Kaynaklar"):
                for doc in found_docs: st.write(doc)
        else:
            st.warning("Bu konuda bilgi bulunamadı.")
