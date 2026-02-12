import streamlit as st
import google.generativeai as genai
import os
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI (Geniş ve Şık) ---
st.set_page_config(
    page_title="Cebimde Müşavir PRO",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TASARIM İYİLEŞTİRMELERİ (CSS) ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
</style>
""", unsafe_allow_html=True)

# --- OTOMATİK MODEL SEÇİCİ ---
try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
    else:
        st.error("🚨 HATA: Secrets içinde GOOGLE_API_KEY bulunamadı.")
        st.stop()

    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
    
    target_models = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
    active_model = None
    
    for target in target_models:
        if target in available_models:
            active_model = target
            break
            
    if not active_model and available_models:
        active_model = available_models[0]
        
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

    with st.sidebar:
        with st.status("📚 Mevzuat Taranıyor...", expanded=True) as status:
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
            status.update(label="✅ Mevzuat Yüklendi!", state="complete", expanded=False)

    if documents:
        vectorizer = TfidfVectorizer(stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(documents)
        return documents, filenames, vectorizer, tfidf_matrix
    else:
        return None, None, None, None

# --- SİSTEM BAŞLATILIYOR ---
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
    1. Sadece aşağıdaki KAYNAK METİNLERdeki bilgiyi kullan.
    2. Cevabın Türkçe, nazik ve kurumsal olsun. "Sayın Mükellefimiz" diye başlayabilirsin.
    3. Önemli tarihleri, tutarları ve oranları **kalın** yaz veya madde madde listele.
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

# --- YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("Hızlı Erişim")
    st.markdown("Aşağıdaki konulara tıklayarak hızlıca bilgi alabilirsiniz:")
    
    # Hazır Sorular (Session State Kullanımı)
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Kira Geliri"):
            st.session_state.user_input = "2024 yılı mesken kira geliri istisna tutarı ne kadar?"
    with col2:
        if st.button("🚗 Araç Gideri"):
            st.session_state.user_input = "Binek otomobil gider kısıtlaması oranı nedir?"
            
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🚀 Girişimci"):
            st.session_state.user_input = "Genç girişimci istisnası şartları ve yaş sınırı nedir?"
    with col4:
        if st.button("🍔 Yemek Bedeli"):
            st.session_state.user_input = "2024 günlük yemek bedeli istisnası ne kadar?"

    st.markdown("---")
    st.info(f"📚 Sistemde {len(filenames)} adet resmi rehber taranmaktadır.")
    st.caption("v2.0 - Şahin Koçarlı")

# --- ANA SAYFA ---
st.title("💼 Cebimde Müşavir AI")
st.markdown("**Dijital Vergi Asistanınız 7/24 Hizmetinizde.**")
st.divider()

# Soru Alanı
user_query = st.text_input("Merak ettiğiniz konuyu yazın veya soldan seçin:", value=st.session_state.user_input)

if st.button("Danış 🔎") and user_query:
    with st.spinner("Dosyalar inceleniyor ve yanıt hazırlanıyor..."):
        # 1. Hızlı Arama
        query_vec = vectorizer.transform([user_query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[-3:][::-1]
        
        found_docs = []
        context_data = ""
        has_data = False
        
        for idx in top_indices:
            if scores[idx] > 0.05:
                has_data = True
                fname = filenames[idx].replace("arsiv_fileadmin_", "").replace("arsiv_onceki-dokumanlar_", "").replace(".pdf", "")
                clean_name = fname.replace("_", " ").title()
                found_docs.append(f"📄 {clean_name}")
                context_data += f"\n--- KAYNAK: {clean_name} ---\n{documents[idx][:4000]}...\n"

        if has_data:
            # 2. AI Cevabı
            response = ask_advisor(user_query, context_data)
            
            # 3. Şık Sonuç Gösterimi
            st.success("✅ Cevap Hazır!")
            st.markdown(response)
            
            # 4. Kaynakça
            with st.expander("📚 Bilginin Kaynağı Olan Resmi Belgeler"):
                for doc in found_docs:
                    st.write(doc)
        else:
            st.warning("⚠️ Bu konuyla ilgili yüklenen rehberlerde eşleşen bir bilgi bulunamadı. Lütfen farklı kelimelerle deneyin.")

# --- ALT BİLGİ (FOOTER) ---
st.markdown("---")
col_footer1, col_footer2 = st.columns([1, 4])
with col_footer1:
    st.markdown("🤖 **AI Powered**")
with col_footer2:
    st.caption("YASAL UYARI: Bu uygulama yapay zeka destekli bilgilendirme amaçlıdır. Verilen bilgilerin resmi geçerliliği yoktur. Nihai kararlarınız ve resmi işlemleriniz için lütfen Yeminli Mali Müşavirinize danışınız.")
