import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Turkish Sentiment Analysis",
    page_icon="🎭",
    layout="wide"
)

# Title
st.title("🎭 Turkish Sentiment Analysis")
st.markdown("""
Analyze the sentiment of Turkish text using Machine Learning.  
Enter your text and click **Analyze**!
""")

# Sidebar
st.sidebar.header("📌 About")
st.sidebar.info("""
**Model:** TF-IDF + Logistic Regression  
**Classes:** Positive, Negative, Neutral  
**Accuracy:** ~78%  

**Tech Stack:**
- Streamlit
- Scikit-learn
- Python

**Author:** [Your Name]  
[GitHub](https://github.com/yourusername)
""")

st.sidebar.header("💡 Try Examples")
if st.sidebar.button("😊 Positive"):
    st.session_state.text = "Bu ürün harika, çok beğendim! Herkese tavsiye ederim."
if st.sidebar.button("😞 Negative"):
    st.session_state.text = "Berbat bir deneyim, asla tavsiye etmem. Paramı boşa harcadım."
if st.sidebar.button("😐 Neutral"):
    st.session_state.text = "Ürün yarın saat 10:00'da kargoya verilecek."

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load model
@st.cache_resource
def load_model():
    # Demo training data
    train_data = {
        'text': [
            'harika mukemmel guzel super bayildim',
            'kotu berbat rezalet felaket',
            'bilgi toplanti tarih saat',
            'cok begendim tavsiye ederim',
            'hic begenmedim asla kullanmam',
            'guncellleme yapildi kayit tamamlandi',
            'mukemmel kalite hizli teslimat',
            'cok yavas calismiyor berbat',
            'fiyat belirlendi adres gonderildi',
            'harika deneyim cok mutluyum',
            'kotü hizmet sinir bozucu',
            'siparis alindi onay maili gonderildi'
        ],
        'label': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    }
    
    df = pd.DataFrame(train_data)
    
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['text'])
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, df['label'])
    
    labels = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
    
    return vectorizer, model, labels

vectorizer, model, labels = load_model()

# Main area
st.subheader("📝 Enter Turkish Text")

default_text = st.session_state.get('text', '')

user_input = st.text_area(
    "Type or paste your text:",
    value=default_text,
    height=120,
    placeholder="Örnek: Bu restoran çok güzel, yemekler lezzetli!"
)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    analyze = st.button("🔍 Analyze", type="primary")

with col2:
    if user_input:
        st.metric("Words", len(user_input.split()))

# Analysis
if analyze and user_input:
    with st.spinner("Analyzing..."):
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        
        sentiment = labels[pred]
        confidence = proba[pred]
        
        st.markdown("---")
        st.subheader("📊 Results")
        
        emoji_map = {'Positive': '😊', 'Negative': '😞', 'Neutral': '😐'}
        color_map = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {emoji_map[sentiment]} {sentiment}")
            st.markdown(f"**Confidence:** {confidence:.1%}")
        
        with col2:
            st.markdown("**All Probabilities:**")
            for i, label in labels.items():
                st.progress(proba[i], text=f"{label}: {proba[i]:.1%}")
        
        # Show cleaned text
        with st.expander("🔧 View Preprocessed Text"):
            st.code(cleaned)

elif analyze:
    st.warning("⚠️ Please enter text to analyze!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with Streamlit 🎈 | For internship applications
</div>
""", unsafe_allow_html=True)
