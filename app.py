import streamlit as st
import google.generativeai as genai
import os
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Cebimde Müşavir PRO",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STİL AYARLARI ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- API VE MODEL ---
try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
    else:
        st.error("🚨 HATA: Secrets içinde GOOGLE_API_KEY bulunamadı.")
        st.stop()

    # Model Seçici (Otomatik)
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    target_models = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
    active_model = None
    
    for target in target_models:
        if target in available_models:
            active_model = target
            break
            
    if not active_model and available_models: active_model = available_models[0]
    
    if not active_model:
        st.error("🚨 HATA: Model bulunamadı.")
        st.stop()
        
    model = genai.GenerativeModel(active_model)

except Exception as e:
    st.error(f"🚨 Bağlantı Hatası: {str(e)}")
    st.stop()

# --- PDF OKUMA SİSTEMİ ---
@st.cache_resource(show_spinner=False)
def create_knowledge_base():
    documents = []
    filenames = []
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdf_files: return None, None, None, None

    # Yükleme ekranı (Sidebar)
    with st.sidebar:
        with st.status("📚 Kütüphane Taranıyor...", expanded=True) as status:
            progress_bar = st.progress(0)
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
                progress_bar.progress((i + 1) / len(pdf_files))
            status.update(label="✅ Hazır!", state="complete", expanded=False)

    if documents:
        vectorizer = TfidfVectorizer(stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(documents)
        return documents, filenames, vectorizer, tfidf_matrix
    else:
        return None, None, None, None

# --- BAŞLATMA ---
documents, filenames, vectorizer, tfidf_matrix = create_knowledge_base()

if not documents:
    st.error("⚠️ Klasörde PDF bulunamadı!")
    st.stop()

# --- MÜŞAVİR FONKSİYONU ---
def ask_advisor(soru, context):
    prompt = f"""
    Sen Türkiye Vergi Mevzuatına hakim, uzman bir Mali Müşavirsin.
    
    GÖREVİN:
    Aşağıdaki "RESMİ KAYNAK METİNLERİ" kullanarak vatandaşın sorusunu cevapla.
    
    KURALLAR:
    1. Sadece verilen kaynakları kullan.
    2. Cevabın Türkçe, net ve profesyonel olsun. "Sayın Mükellefimiz" diye başla.
    3. Önemli sayıları, yaş sınırlarını ve tarihleri madde madde yaz.
    4. Kaynaklarda bilgi yoksa "Bu konuda yüklenen rehberlerde bilgi bulunamadı" de.
    
    KAYNAKLAR:
    {context}
    
    SORU:
    {soru}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"🚨 Hata: {str(e)}"

# --- YAN MENÜ ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("İşlemler")
    
    if "query_input" not in st.session_state: st.session_state.query_input = ""
    def set_query(q): st.session_state.query_input = q

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚗 Araç Gider"): set_query("Binek otomobil gider kısıtlaması oranı nedir?")
    with col2:
        if st.button("🏠 Kira Geliri"): set_query("2024 mesken kira istisnası ne kadar?")
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🚀 Girişimci"): set_query("Genç girişimci istisnası yaş ve şartları?")
    with col4:
        if st.button("🍔 Yemek"): set_query("Günlük yemek bedeli istisnası kaç TL?")

    st.divider()
    with st.expander("📂 Yüklü Dosyalar"):
        for f in filenames:
            st.caption(f"📄 {f.replace('.pdf', '')}")

# --- ANA EKRAN ---
st.title("💼 Cebimde Müşavir PRO")
st.markdown("**Dijital Vergi Asistanınız (Genişletilmiş Hafıza)**")

user_query = st.text_input("Sorunuz:", key="query_input")

if st.button("Danış 🔎", type="primary") and user_query:
    with st.spinner("Dosyalar derinlemesine inceleniyor..."):
        # 1. Hızlı Arama
        query_vec = vectorizer.transform([user_query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # GÜNCELLEME: İlk 3 değil, ilk 5 dosyayı alıyoruz!
        top_indices = scores.argsort()[-5:][::-1]
        
        found_docs = []
        context_data = ""
        has_data = False
        
        for idx in top_indices:
            if scores[idx] > 0.05:
                has_data = True
                fname = filenames[idx].replace("arsiv_fileadmin_", "").replace(".pdf", "")
                found_docs.append(f"📄 {fname}")
                
                # GÜNCELLEME: [:4000] yerine [:50000] yaptık! (Yaklaşık 30 sayfa okur)
                # Artık metni kesmiyoruz, neredeyse tamamını yolluyoruz.
                doc_content = documents[idx][:50000] 
                context_data += f"\n--- KAYNAK: {fname} ---\n{doc_content}\n"

        if has_data:
            # 2. AI Cevabı
            response = ask_advisor(user_query, context_data)
            
            # 3. Sonuç
            st.success("✅ Cevap Hazır!")
            st.markdown(response)
            
            with st.expander("📚 İncelenen Belgeler"):
                for doc in found_docs: st.write(doc)
        else:
            st.warning("⚠️ İlgili konu yüklenen dosyalarda bulunamadı. Lütfen sol menüden dosya listesini kontrol edin.")

st.markdown("---")
st.caption("YASAL UYARI: Bu sistem bilgilendirme amaçlıdır. Nihai karar için YMM'ye danışınız.")
