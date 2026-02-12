import streamlit as st
import google.generativeai as genai
import os
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI (Geniş ve Modern) ---
st.set_page_config(
    page_title="Mevzuat AI - Prototip",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TASARIM (Teknolojik Görünüm) ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        background-color: #0066cc; /* Kurumsal Mavi */
        color: white;
    }
    .stButton>button:hover {
        background-color: #0052a3;
    }
    .block-container {
        padding-top: 1.5rem;
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

    with st.sidebar:
        with st.status("🧠 Yapay Zeka Mevzuatı Tarıyor...", expanded=True) as status:
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
            status.update(label="✅ Veri Tabanı Hazır!", state="complete", expanded=False)

    if documents:
        vectorizer = TfidfVectorizer(stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(documents)
        return documents, filenames, vectorizer, tfidf_matrix
    else:
        return None, None, None, None

# --- BAŞLATMA ---
documents, filenames, vectorizer, tfidf_matrix = create_knowledge_base()

if not documents:
    st.error("⚠️ Klasörde PDF bulunamadı! Lütfen GitHub'a dosya yükleyin.")
    st.stop()

# --- MÜŞAVİR FONKSİYONU ---
def ask_advisor(soru, context):
    prompt = f"""
    Sen Türkiye Vergi Mevzuatına hakim, uzman bir Mali Müşavirsin.
    
    GÖREVİN:
    Aşağıdaki "RESMİ KAYNAK METİNLERİ" kullanarak soruyu cevapla.
    
    KURALLAR:
    1. Sadece verilen kaynakları kullan.
    2. Cevabın Türkçe, net ve profesyonel olsun.
    3. Önemli sayıları, limitleri ve tarihleri **kalın** yaz veya madde madde listele.
    4. Kaynaklarda bilgi yoksa "Mevcut yüklenen rehberlerde bu konu hakkında bilgi bulunmamaktadır." de.
    
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

# --- YAN MENÜ (VİZYON KISMI) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620541.png", width=70)
    st.title("Mevzuat AI")
    st.caption("Ar-Ge Prototip v1.0")
    
    st.info("""
    ℹ️ **Proje Hakkında:**
    Bu sistem, vergi mevzuatının yapay zeka ile **anlık olarak analiz edilebilirliğini** göstermek amacıyla hazırlanmış bir teknik demodu.
    
    Yüklü olan resmi PDF rehberleri üzerinden çalışır ve kaynak gösterir.
    """)

    st.divider()

    if "query_input" not in st.session_state: st.session_state.query_input = ""
    def set_query(q): st.session_state.query_input = q

    st.markdown("**⚡ Örnek Senaryolar:**")
    if st.button("🚗 Araç Gider Kısıtlaması"): set_query("Binek otomobil gider kısıtlaması oranı nedir?")
    if st.button("🏠 Kira Geliri İstisnası"): set_query("2024 mesken kira istisnası ne kadar?")
    if st.button("🚀 Genç Girişimci Şartları"): set_query("Genç girişimci istisnası yaş ve şartları?")

    st.divider()
    with st.expander("📂 Analiz Edilen Kaynaklar"):
        for f in filenames:
            st.caption(f"📄 {f.replace('.pdf', '')}")

# --- ANA EKRAN ---
st.title("⚖️ Mevzuat Analiz Sistemi")
st.markdown("""
**Hoş Geldiniz.** Bu uygulama, yüklenen resmi vergi rehberlerini tarayarak sorularınıza **kaynaklı ve gerekçeli** yanıtlar üretir.
""")

user_query = st.text_input("Analiz edilecek konuyu yazın:", key="query_input", placeholder="Örn: Asgari ücret istisnası nasıl uygulanır?")

if st.button("Analiz Et 🔎") and user_query:
    with st.spinner("Mevzuat taranıyor, ilgili maddeler analiz ediliyor..."):
        # 1. Hızlı Arama
        query_vec = vectorizer.transform([user_query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[-5:][::-1] # En iyi 5 sonuç (Turbo Mod)
        
        found_docs = []
        context_data = ""
        has_data = False
        
        for idx in top_indices:
            if scores[idx] > 0.05:
                has_data = True
                fname = filenames[idx].replace("arsiv_fileadmin_", "").replace(".pdf", "")
                found_docs.append(f"📄 {fname}")
                # Geniş okuma limiti (50.000 karakter)
                doc_content = documents[idx][:50000] 
                context_data += f"\n--- KAYNAK: {fname} ---\n{doc_content}\n"

        if has_data:
            # 2. AI Cevabı
            response = ask_advisor(user_query, context_data)
            
            # 3. Sonuç
            st.success("✅ Analiz Tamamlandı")
            st.markdown(response)
            
            with st.expander("📚 Referans Alınan Resmi Belgeler"):
                for doc in found_docs: st.write(doc)
        else:
            st.warning("⚠️ Aradığınız konu, sisteme yüklenen mevcut rehberlerde tespit edilemedi.")

st.markdown("---")
st.caption("YASAL UYARI: Bu bir Ar-Ge (Araştırma Geliştirme) prototipidir. Üretilen bilgiler resmi tavsiye niteliği taşımaz.")
