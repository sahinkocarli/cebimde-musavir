import streamlit as st
import google.generativeai as genai
import os
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Cebimde Müşavir AI", page_icon="🧾", layout="centered")

# --- API KURULUMU VE OTOMATİK MODEL SEÇİMİ ---
try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
    else:
        st.error("🚨 HATA: Secrets içinde GOOGLE_API_KEY bulunamadı.")
        st.stop()

    # Google'a soruyoruz: Hangi modeller açık?
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
    
    # En hızlı ve zeki olandan başlayarak seç
    target_models = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
    active_model = None
    
    for target in target_models:
        if target in available_models:
            active_model = target
            break
            
    if not active_model and available_models:
        active_model = available_models[0]
        
    if not active_model:
        st.error("🚨 HATA: Bu anahtar ile hiçbir yapay zeka modeline erişilemiyor.")
        st.stop()
        
    # Modeli Sessizce Başlat
    model = genai.GenerativeModel(active_model)

except Exception as e:
    st.error(f"🚨 Bağlantı Hatası: {str(e)}")
    st.stop()

# --- PDF OKUMA SİSTEMİ (ÖNBELLEKLİ) ---
@st.cache_resource(show_spinner=False)
def create_knowledge_base():
    documents = []
    filenames = []
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdf_files: return None, None, None, None

    # İlerleme çubuğu (Sadece ilk açılışta görünür)
    progress_text = "📚 Resmi Gazete ve Rehberler Taranıyor..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, pdf_file in enumerate(pdf_files):
        try:
            reader = pypdf.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t: text += t + "\n"
            documents.append(text)
            filenames.append(pdf_file)
        except: pass
        my_bar.progress((i + 1) / len(pdf_files), text=progress_text)

    my_bar.empty() # İş bitince çubuğu gizle

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
    st.error("⚠️ Klasörde PDF dosyası bulunamadı! Lütfen GitHub'a dosya yükleyin.")
    st.stop()

# --- MÜŞAVİR FONKSİYONU ---
def ask_advisor(soru, context):
    prompt = f"""
    Sen Türkiye Vergi Mevzuatına hakim, profesyonel bir Dijital Mali Müşavirsin.
    
    GÖREVİN:
    Aşağıda sana verilen "RESMİ KAYNAK METİNLERİ" (CONTEXT) kullanarak, vatandaşın sorusunu net, doğru ve profesyonelce cevapla.
    
    KURALLAR:
    1. Sadece aşağıdaki KAYNAK METİNLERdeki bilgiyi kullan. Harici bilgi ekleme.
    2. Cevabın Türkçe, nazik ve anlaşılır olsun.
    3. Önemli tarihleri, tutarları ve oranları madde madde yaz.
    4. Kaynaklarda bilgi yoksa, "Yüklenen rehberlerde bu konuyla ilgili net bir bilgi bulunmamaktadır." de.
    
    RESMİ KAYNAK METİNLER:
    {context}
    
    VATANDAŞIN SORUSU:
    {soru}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"🚨 Bir hata oluştu: {str(e)}"

# --- ARAYÜZ (FRONTEND) ---
st.title("🧾 Cebimde Müşavir AI")
st.caption(f"📚 Sistem hafızasında {len(filenames)} adet güncel mevzuat rehberi bulunmaktadır.")

# Soru Alanı
user_query = st.text_input("Mevzuat sorunuzu yazın:", placeholder="Örn: Genç girişimci istisnası şartları nelerdir?")

if st.button("Danış") and user_query:
    with st.spinner("🔍 Mevzuat taranıyor ve analiz ediliyor..."):
        # 1. Hızlı Arama (Vektör)
        query_vec = vectorizer.transform([user_query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[-3:][::-1]
        
        found_docs = []
        context_data = ""
        has_data = False
        
        for idx in top_indices:
            if scores[idx] > 0.05: # Alaka düzeyi filtresi
                has_data = True
                fname = filenames[idx].replace("arsiv_fileadmin_", "").replace("arsiv_onceki-dokumanlar_", "").replace(".pdf", "")
                # İsmi temizle ve listeye ekle
                clean_name = fname.replace("_", " ").title()
                found_docs.append(f"📄 {clean_name}")
                
                # İçeriği bağlama ekle (İlk 4000 karakter)
                context_data += f"\n--- KAYNAK: {clean_name} ---\n{documents[idx][:4000]}...\n"

        if has_data:
            # 2. Yapay Zeka Cevabı
            response = ask_advisor(user_query, context_data)
            
            # 3. Sonuç Gösterimi
            st.markdown("### 🤖 Müşavir Cevabı:")
            st.info(response)
            
            # 4. Kaynakça
            with st.expander("📚 Başvurulan Resmi Kaynaklar"):
                for doc in found_docs:
                    st.write(doc)
        else:
            st.warning("Bu konuyla ilgili yüklenen rehberlerde eşleşen bir bilgi bulunamadı. Lütfen sorunuzu farklı kelimelerle tekrar deneyin.")

st.markdown("---")
st.markdown("⚠️ *Bu sistem bilgilendirme amaçlıdır. Nihai kararlarınız için yeminli mali müşavirinize danışınız.*")
