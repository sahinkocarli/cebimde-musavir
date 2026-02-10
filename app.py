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
# Anahtarı alıyoruz ama kütüphaneye vermiyoruz, kendimiz kullanacağız.
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🚨 HATA: Streamlit Secrets ayarlarında 'GOOGLE_API_KEY' bulunamadı!")
    st.stop()

# --- YENİ YÖNTEM: DİREKT HTTP İSTEĞİ (REST API) ---
def ask_google_directly(prompt):
    # Google'ın en standart ve çalışan model adresi
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"🚨 Google Hatası ({response.status_code}): {response.text}"
            
    except Exception as e:
        return f"🚨 Bağlantı Hatası: {str(e)}"

# --- FONKSİYON: PDF'LERİ OKU VE HAFIZAYA AT ---
@st.cache_resource(show_spinner=False)
def create_knowledge_base():
    documents = []
    filenames = []
    
    # Klasördeki tüm PDF'leri bul
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdf_files:
        return None, None, None, None

    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, pdf_file in enumerate(pdf_files):
        try:
            # Kullanıcıya bilgi ver
            status_text.text(f"📚 İşleniyor: {pdf_file}...")
            
            reader = pypdf.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Belgeyi listeye ekle
            documents.append(text)
            filenames.append(pdf_file)
        except Exception as e:
            print(f"Hata ({pdf_file}): {e}")
        
        # İlerleme çubuğunu güncelle
        progress_bar.progress((i + 1) / len(pdf_files))

    status_text.empty()
    progress_bar.empty()

    # TF-IDF Matrisini Oluştur (Arama Motoru)
    if documents:
        vectorizer = TfidfVectorizer(stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(documents)
        return documents, filenames, vectorizer, tfidf_matrix
    else:
        return None, None, None, None

# --- SİSTEM BAŞLANGICI ---
with st.spinner("🚀 Sistem başlatılıyor ve PDF'ler okunuyor... (Bu işlem bir kez yapılır)"):
    documents, filenames, vectorizer, tfidf_matrix = create_knowledge_base()

if documents is None or len(documents) == 0:
    st.error("⚠️ Klasörde hiç PDF dosyası bulunamadı! Lütfen GitHub'a PDF yüklediğinizden emin olun.")
    st.stop()

# --- MÜŞAVİR MANTIĞI ---
def ask_advisor(soru, context_text):
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
    
    # Artık kütüphaneyi değil, kendi yazdığımız direkt fonksiyonu çağırıyoruz
    return ask_google_directly(prompt)

# --- ARAYÜZ (FRONTEND) ---
st.title("🧾 Cebimde Müşavir AI")
st.caption(f"📚 {len(filenames)} adet resmi rehber hafızaya alındı.")

# Soru Kutusu
user_query = st.text_input("Mevzuat sorunuzu yazın:", placeholder="Örn: Kira geliri istisnası ne kadar?")

if st.button("Danış") and user_query:
    with st.spinner("🔍 Mevzuat taranıyor ve Müşavir yorumluyor..."):
        # 1. Hızlı Arama
        query_vec = vectorizer.transform([user_query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # En iyi 3 sonucu getir
        top_indices = scores.argsort()[-3:][::-1]
        
        found_docs = []
        context_data = ""
        has_relevant_data = False
        
        for idx in top_indices:
            score = scores[idx]
            if score > 0.05: # Filtre
                has_relevant_data = True
                doc_text = documents[idx]
                fname = filenames[idx]
                
                # Dosya ismini temizle
                clean_name = fname.replace("arsiv_fileadmin_", "").replace("arsiv_onceki-dokumanlar_", "").replace(".pdf", "")
                
                found_docs.append(f"📄 {clean_name}")
                # Çok uzun metinleri kısalt
                context_data += f"\n--- KAYNAK: {clean_name} ---\n{doc_text[:4000]}...\n"

        if has_relevant_data:
            # 2. Direkt Google'a Sor
            ai_response = ask_advisor(user_query, context_data)
            
            # 3. Sonucu Göster
            st.markdown("### 🤖 Müşavir Cevabı:")
            
            if "🚨" in ai_response:
                st.error(ai_response)
            else:
                st.info(ai_response)
            
            # 4. Kaynakları Göster
            with st.expander("📚 Kullanılan Resmi Kaynaklar"):
                for doc in found_docs:
                    st.write(doc)
        else:
            st.warning("Bu konuyla ilgili mevzuat rehberlerinde eşleşen bir bilgi bulunamadı.")

st.markdown("---")
st.markdown("⚠️ *Bu sistem bilgilendirme amaçlıdır.*")
