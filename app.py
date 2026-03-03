import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import json
import time
import io
import os
import base64
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from fpdf import FPDF
from datetime import datetime

# --- LOGO HELPER ---
def get_logo_base64():
    """Load logo and return as base64 string for inline display"""
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# --- PAGE CONFIG ---
st.set_page_config(page_title="FinBehavior AI", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")

# --- CUSTOM DARK THEME CSS ---
st.markdown("""
<style>
    /* ===== GLOBAL DARK THEME ===== */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
        color: #e0e0e0;
    }
    
    /* ===== LOGO & HEADER ===== */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 4px;
    }
    .logo-img {
        width: 56px;
        height: 56px;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0, 210, 255, 0.25);
        flex-shrink: 0;
    }
    .main-header {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 50%, #6c5ce7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -1px;
        line-height: 1.1;
        margin: 0;
    }
    .sub-header {
        color: #a0a0c0;
        font-size: 1.1rem;
        margin-top: 2px;
        font-weight: 400;
    }
    
    /* ===== METRIC CARDS ===== */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.15);
    }
    [data-testid="stMetricLabel"] {
        color: #a0a0c0 !important;
        font-size: 0.85rem;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141428 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #00d2ff !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #00d2ff, #6c5ce7) !important;
        border: none !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.4) !important;
    }
    
    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px !important;
    }
    
    /* ===== CONSENT CARD ===== */
    .consent-card {
        background: rgba(108, 92, 231, 0.1);
        border: 1px solid rgba(108, 92, 231, 0.3);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 40px auto;
        max-width: 600px;
    }
    .consent-card h2 { color: #00d2ff; margin-bottom: 16px; }
    .consent-card p { color: #a0a0c0; line-height: 1.7; }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #a0a0c0;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 210, 255, 0.1) !important;
        color: #00d2ff !important;
        border-bottom: 2px solid #00d2ff;
    }
    
    /* ===== ARCHITECTURE FLOW ===== */
    .flow-step {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .flow-step:hover {
        border-color: #00d2ff;
        transform: translateY(-3px);
    }
    .flow-step .icon { font-size: 2rem; margin-bottom: 8px; }
    .flow-step .title { color: #00d2ff; font-weight: 700; font-size: 0.95rem; }
    .flow-step .desc { color: #a0a0c0; font-size: 0.8rem; margin-top: 6px; }

    /* ===== SUCCESS/ERROR BOXES ===== */
    .stAlert {
        border-radius: 10px !important;
    }
    
    /* ===== PLOTLY CHARTS ===== */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    
    /* ===== HIDE STREAMLIT BRANDING (keep sidebar toggle visible) ===== */
    footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    
    /* ===== ENSURE SIDEBAR IS VISIBLE ===== */
    [data-testid="stSidebar"] {
        min-width: 320px !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        visibility: visible !important;
    }
</style>
""", unsafe_allow_html=True)


# --- INITIALIZE SESSION STATE ---
if 'consent_given' not in st.session_state:
    st.session_state.consent_given = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_scores' not in st.session_state:
    st.session_state.analysis_scores = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []


# --- PREPROCESSING ---
def preprocess_text(text):
    """Clean social media posts for NLP"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s₹]', '', text)  # Keep ₹ for amount detection
    text = ' '.join(text.split())
    return text


def extract_amount(text):
    """Extract monetary amounts from post"""
    patterns = [
        r'₹\s?([\d,]+(?:\.\d+)?)\s*(crore|crores|cr|lakh|lakhs|lac|l|k)?',
        r'(?:Rs\.?|INR)\s?([\d,]+(?:\.\d+)?)\s*(crore|crores|cr|lakh|lakhs|lac|l|k)?',
        r'([\d,]+(?:\.\d+)?)\s*(?:rupees|rs)\s*(crore|crores|cr|lakh|lakhs|lac|l|k)?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '')
            amount = float(amount_str)
            # Handle suffix captured by regex (group 2)
            suffix = (match.group(2) or '').lower().strip()
            if suffix in ('lakh', 'lakhs', 'lac', 'l'):
                amount *= 100000
            elif suffix in ('crore', 'crores', 'cr'):
                amount *= 10000000
            elif suffix == 'k':
                amount *= 1000
            return amount
    return 0


# --- LOAD MODELS (CACHED) ---
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Models are loaded lazily inside the analysis block with visual feedback
# (removed top-level blocking call that prevented page from rendering)


# --- SCORING LOGIC (ENHANCED) ---
def calculate_scores(classifications):
    """Enhanced scoring with confidence weighting and amount extraction"""
    scores = {"Spending": 0, "Investment": 0, "Loan": 0, "Savings": 0, "Risk": 0}
    weighted_scores = {"Spending": 0, "Investment": 0, "Loan": 0, "Savings": 0, "Risk": 0}
    total_amount = {"Spending": 0, "Investment": 0, "Loan": 0, "Savings": 0, "Risk": 0}
    
    confident_items = [c for c in classifications if c['label'] != "Uncertain"]
    
    for item in confident_items:
        label = item['label']
        if label in scores:
            scores[label] += 1
            weighted_scores[label] += item.get('confidence', 0.5)
            total_amount[label] += item.get('amount', 0)
    
    total = sum(scores.values())
    if total == 0:
        return scores, 0, 0, weighted_scores, total_amount
    
    # Enhanced risk: weighted by confidence + amount factor
    amount_factor = 1.0
    if total_amount['Spending'] + total_amount['Loan'] + total_amount['Risk'] > 0:
        risky_amount = total_amount['Spending'] + total_amount['Loan'] + total_amount['Risk']
        safe_amount = total_amount['Investment'] + total_amount['Savings'] + 1
        amount_factor = min(risky_amount / safe_amount, 3.0)  # Cap at 3x
    
    base_risk = (weighted_scores['Spending'] * 1 + weighted_scores['Loan'] * 1.5 + weighted_scores['Risk'] * 2)
    stabilizer = (weighted_scores['Investment'] * 1.2 + weighted_scores['Savings'] * 1.0)
    
    risk_score = ((base_risk - stabilizer * 0.5) / total) * 50 * (1 + amount_factor * 0.1)
    risk_score = max(0, min(risk_score, 100))
    
    uncertainty_rate = (len(classifications) - len(confident_items)) / len(classifications) * 100
    
    return scores, risk_score, uncertainty_rate, weighted_scores, total_amount


# --- PDF REPORT GENERATION ---
def _pdf_safe(text):
    """Sanitize text for PDF output — replace Unicode chars unsupported by Helvetica"""
    replacements = {
        '₹': 'Rs.', '😍': '', '📈': '', '✅': '', '🎉': '', '✈️': '',
        '🚀': '', '💰': '', '😅': '', '📦': '', '🤞': '', '🙌': '',
        '💪': '', '🎯': '', '🎲': '', '⚠️': '', '❓': '', '💸': '',
        '🏦': '', '📊': '', '💡': '', '🔒': '', '🔴': '', '🟢': '',
        '🟡': '', '🕸️': '', '📋': '', '🔬': '', '📜': '', '⚡': '',
        '\u200d': '', '\ufe0f': '',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Remove any remaining non-latin1 characters
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text


def generate_pdf_report(results, scores, risk_score, uncertainty_rate):
    """Generate a downloadable PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 15, "FinBehavior AI Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    
    # Risk Score
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Risk Score: {int(risk_score)}/100", ln=True)
    level = "HIGH RISK" if risk_score > 70 else "LOW RISK" if risk_score < 30 else "MEDIUM RISK"
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Profile: {level}", ln=True)
    pdf.cell(0, 8, f"Uncertainty Rate: {uncertainty_rate:.0f}%", ln=True)
    pdf.ln(5)
    
    # Category Breakdown
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Category Breakdown", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for cat, count in scores.items():
        pdf.cell(0, 7, f"  {cat}: {count} posts", ln=True)
    pdf.ln(5)
    
    # Detailed Classifications
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Post Classifications", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for r in results:
        post_text = _pdf_safe(r['Post'][:80] + ('...' if len(r['Post']) > 80 else ''))
        pdf.cell(0, 6, f"[{r['Category']}] (conf: {r['Confidence']}) {post_text}", ln=True)
    
    # Privacy Notice
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, "Privacy Notice: This report was generated from consent-based behavioral analysis. Data is not stored or shared.")
    
    return bytes(pdf.output())


# --- CONFUSION MATRIX ---
def run_model_evaluation(classifier_model):
    """Evaluate model accuracy on synthetic data"""
    try:
        with open("synthetic_posts.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        return None, None, None
    
    labels = ["Spending", "Investment", "Loan", "Savings", "Risk"]
    true_labels = []
    pred_labels = []
    
    sample = test_data[:25]  # Use 25 samples for speed
    
    progress = st.progress(0)
    for i, item in enumerate(sample):
        clean = preprocess_text(item['text'])
        output = classifier_model(clean, candidate_labels=labels, multi_label=False)
        pred_labels.append(output['labels'][0])
        true_labels.append(item['true_label'])
        progress.progress((i + 1) / len(sample))
    progress.empty()
    
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    report = classification_report(true_labels, pred_labels, labels=labels, output_dict=True, zero_division=0)
    
    return cm, report, labels


# ========================
# MAIN APPLICATION
# ========================
def main():
    # --- CONSENT FLOW ---
    if not st.session_state.consent_given:
        logo_b64 = get_logo_base64()
        if logo_b64:
            st.markdown(f'''
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_b64}" class="logo-img" alt="FinBehavior AI logo">
                <div>
                    <h1 class="main-header">FinBehavior AI</h1>
                    <p class="sub-header">Social Behavioral Credit Intelligence Platform</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('<h1 class="main-header">FinBehavior AI</h1>', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Social Behavioral Credit Intelligence Platform</p>', unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("")
        
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.markdown("""
            <div class="consent-card">
                <h2>🔐 User Consent Required</h2>
                <p>
                    This tool analyzes social media posts to assess financial behavior patterns.<br><br>
                    <strong>Before proceeding, please confirm:</strong><br>
                    ✅ You consent to analyze the text you provide<br>
                    ✅ You understand this is a research prototype<br>
                    ✅ No data is stored or shared externally<br>
                    ✅ All scores are explainable and appealable
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("✅ I Consent — Proceed to Dashboard", type="primary", use_container_width=True):
                st.session_state.consent_given = True
                st.rerun()
            
            st.markdown("")
            st.caption("🔒 Compliant with GDPR & India's DPDP Act 2023")
        return
    
    # --- MAIN DASHBOARD ---
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.markdown(f'''
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_b64}" class="logo-img" alt="FinBehavior AI logo">
            <div>
                <h1 class="main-header">FinBehavior AI</h1>
                <p class="sub-header">Social Behavioral Credit Intelligence Dashboard — Consent-Based & Privacy-First</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<h1 class="main-header">FinBehavior AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Social Behavioral Credit Intelligence Dashboard — Consent-Based & Privacy-First</p>', unsafe_allow_html=True)

    # --- SIDEBAR ---
    st.sidebar.markdown("## 📥 Input Data")
    
    input_method = st.sidebar.radio("Input Method", ["📝 Text Input", "📁 Upload File"], horizontal=True)
    
    if input_method == "📝 Text Input":
        demo_mode = st.sidebar.checkbox("✅ Use Demo Data", value=True)
        
        if demo_mode:
            default_text = """Bought new iPhone 15 Pro on EMI ₹5000/month 😍
Invested ₹50,000 in Mutual Funds today 📈
Paid credit card bill of ₹12,500 on time ✅
Personal loan of ₹2L approved finally! 🎉
Trip to Dubai booked for ₹80,000 - can't wait! ✈️
Transferred ₹15,000 rent to landlord
Bought crypto because FOMO - ₹25,000 in Shiba Inu 🚀
Saved ₹10,000 this month for emergency fund 💰
Lost ₹5000 in options trading... lesson learned 😅
Started SIP of ₹2000/month in NPS"""
            user_input = st.sidebar.text_area("Social Posts (one per line)", value=default_text, height=250)
        else:
            user_input = st.sidebar.text_area("Paste Social Posts (one per line)", height=250,
                                              placeholder="Example: Bought new laptop on EMI ₹3000/month")
        posts = [line.strip() for line in user_input.split('\n') if line.strip()] if user_input else []
    
    else:  # File Upload
        uploaded_file = st.sidebar.file_uploader("Upload CSV or JSON", type=['csv', 'json'])
        posts = []
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                text_col = st.sidebar.selectbox("Select text column", df.columns)
                posts = df[text_col].dropna().tolist()
            elif uploaded_file.name.endswith('.json'):
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    if isinstance(data[0], dict) and 'text' in data[0]:
                        posts = [item['text'] for item in data]
                    else:
                        posts = [str(item) for item in data]
            st.sidebar.success(f"✅ Loaded {len(posts)} posts")
        user_input = "\n".join(posts) if posts else ""

    # Advanced Settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        confidence_threshold = st.slider("Confidence Threshold", 0.3, 0.9, 0.5, 0.05,
                                         help="Posts below this confidence → 'Uncertain'")
        include_sentiment = st.checkbox("🎭 Include Sentiment Analysis", value=True,
                                        help="Adds emotional context to risk scoring")

    # --- ROI CALCULATOR (Add-On #3) ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Business Impact Calculator")
    loan_amount = st.sidebar.number_input("Avg Loan Size (₹)", value=500000, step=50000, format="%d")
    default_rate_reduction = st.sidebar.slider("Expected Default Reduction (%)", 1, 20, 5)
    customers = st.sidebar.number_input("Customers Screened/Year", value=1000, step=100, format="%d")
    potential_savings = (loan_amount * (default_rate_reduction / 100) * customers)
    st.sidebar.metric("💎 Potential Savings/Year", f"₹{potential_savings:,.0f}")
    if potential_savings >= 10000000:
        st.sidebar.success(f"= ₹{potential_savings/10000000:.1f} Crores")
    elif potential_savings >= 100000:
        st.sidebar.info(f"= ₹{potential_savings/100000:.1f} Lakhs")
    st.sidebar.caption("_Projected savings from reduced loan defaults using behavioral screening_")

    analyze_btn = st.sidebar.button("🚀 Analyze Behavior", type="primary", use_container_width=True)

    # --- ANALYSIS LOGIC ---
    if analyze_btn and posts:
        if not posts:
            st.warning("⚠️ Please enter at least one post.")
            return
        
        labels = ["Spending", "Investment", "Loan", "Savings", "Risk"]
        results = []
        
        # Load classification model with visible feedback
        with st.spinner("🧠 Loading NLP classification model... (first time may take a minute)"):
            classifier_model = load_classifier()
        
        # Load sentiment model if needed
        sentiment_model = None
        if include_sentiment:
            with st.spinner("🎭 Loading sentiment model..."):
                sentiment_model = load_sentiment_model()
        
        # Progress UX
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_messages = [
            "🔍 Parsing financial intent...",
            "🧠 Running NLP classification...",
            "📊 Computing behavior vectors...",
            "⚡ Calculating risk metrics...",
            "🎯 Generating recommendations..."
        ]
        
        start_time = time.time()
        
        with st.spinner('🤖 AI analysis in progress...'):
            for i, post in enumerate(posts):
                # Status message rotation
                msg_idx = min(i * len(status_messages) // len(posts), len(status_messages) - 1)
                status_text.text(f"{status_messages[msg_idx]} ({i+1}/{len(posts)})")
                
                clean_post = preprocess_text(post)
                amount = extract_amount(post)
                
                # Zero-shot classification
                output = classifier_model(clean_post, candidate_labels=labels, multi_label=False)
                confidence = output['scores'][0]
                category = output['labels'][0] if confidence >= confidence_threshold else "Uncertain"
                
                # Sentiment analysis
                sentiment = {"label": "N/A", "score": 0}
                if sentiment_model:
                    try:
                        sent_result = sentiment_model(clean_post[:512])
                        sentiment = sent_result[0]
                    except:
                        pass
                
                # Sentiment-adjusted confidence
                adjusted_confidence = confidence
                if include_sentiment and sentiment['label'] != "N/A":
                    if sentiment['label'] == 'NEGATIVE' and category in ['Spending', 'Loan', 'Risk']:
                        adjusted_confidence = min(confidence * 1.15, 1.0)  # Boost risky items with negative sentiment
                    elif sentiment['label'] == 'POSITIVE' and category in ['Investment', 'Savings']:
                        adjusted_confidence = min(confidence * 1.1, 1.0)  # Boost stable items with positive sentiment
                
                results.append({
                    "Post": post,
                    "Category": category,
                    "Confidence": round(adjusted_confidence, 3),
                    "Raw_Confidence": round(confidence, 3),
                    "Amount": amount,
                    "Sentiment": sentiment['label'],
                    "Sentiment_Score": round(sentiment.get('score', 0), 3),
                    "All_Scores": {k: round(v, 3) for k, v in zip(output['labels'], output['scores'])}
                })
                
                progress_bar.progress((i + 1) / len(posts))
        
        elapsed = time.time() - start_time
        progress_bar.empty()
        status_text.empty()
        
        # Store in session state
        scores, risk_score, uncertainty_rate, weighted_scores, total_amounts = calculate_scores([
            {"label": r["Category"], "confidence": r["Confidence"], "amount": r["Amount"]} for r in results
        ])
        
        st.session_state.analysis_results = results
        st.session_state.analysis_scores = {
            "scores": scores, "risk_score": risk_score,
            "uncertainty_rate": uncertainty_rate,
            "weighted_scores": weighted_scores,
            "total_amounts": total_amounts,
            "elapsed": elapsed,
            "confidence_threshold": confidence_threshold,
            "include_sentiment": include_sentiment
        }
        st.session_state.analysis_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "risk_score": risk_score,
            "posts": len(posts)
        })
        st.rerun()
    
    # --- DISPLAY RESULTS (from session state) ---
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        data = st.session_state.analysis_scores
        scores = data['scores']
        risk_score = data['risk_score']
        uncertainty_rate = data['uncertainty_rate']
        weighted_scores = data['weighted_scores']
        total_amounts = data['total_amounts']
        elapsed = data['elapsed']
        confidence_threshold = data['confidence_threshold']
        
        st.success(f"✅ Analysis Complete — {len(results)} posts in {elapsed:.1f}s ({elapsed/len(results)*1000:.0f}ms/post)")
        
        # --- TABS ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Details", "🔬 Model Eval", "📜 History"])
        
        with tab1:
            # Top Metrics
            st.subheader("📊 Financial Behavior Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                risk_emoji = "🔴" if risk_score > 70 else "🟢" if risk_score < 30 else "🟡"
                st.metric("Risk Score", f"{int(risk_score)}/100",
                          delta=f"{risk_emoji} High" if risk_score > 70 else f"{risk_emoji} Low" if risk_score < 30 else f"{risk_emoji} Medium")
            with col2:
                st.metric("💸 Spending", scores['Spending'])
            with col3:
                st.metric("📈 Investment", scores['Investment'])
            with col4:
                st.metric("🏦 Loan", scores['Loan'])
            with col5:
                st.metric("❓ Uncertain", f"{uncertainty_rate:.0f}%",
                          delta="Good" if uncertainty_rate < 20 else "Review needed")
            
            # Charts
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("🥧 Behavior Distribution")
                df = pd.DataFrame(list(scores.items()), columns=['Category', 'Count'])
                df = df[df['Count'] > 0]
                if not df.empty:
                    fig = px.pie(df, values='Count', names='Category', hole=0.45,
                                color_discrete_sequence=['#00d2ff', '#6c5ce7', '#fd79a8', '#00b894', '#e17055'])
                    fig.update_layout(
                        showlegend=True, height=350,
                        margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#a0a0c0',
                        legend=dict(font=dict(color='#e0e0e0'))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No confident classifications to display.")
            
            with c2:
                st.subheader("🕸️ Behavior Profile Radar")
                # Radar Chart
                categories = list(scores.keys())
                values = list(scores.values())
                max_val = max(values) if max(values) > 0 else 1
                normalized = [v / max_val * 100 for v in values]
                
                # Add benchmark
                benchmark = [50, 50, 30, 60, 20]  # "Ideal" profile
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized + [normalized[0]],
                    theta=categories + [categories[0]],
                    fill='toself', name='Your Profile',
                    fillcolor='rgba(0, 210, 255, 0.2)',
                    line=dict(color='#00d2ff', width=2)
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=benchmark + [benchmark[0]],
                    theta=categories + [categories[0]],
                    fill='toself', name='Ideal Benchmark',
                    fillcolor='rgba(0, 184, 148, 0.1)',
                    line=dict(color='#00b894', width=1, dash='dot')
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor='rgba(255,255,255,0.1)'),
                        angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#e0e0e0')
                    ),
                    showlegend=True, height=350,
                    margin=dict(t=30, b=10, l=40, r=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#a0a0c0',
                    legend=dict(font=dict(color='#e0e0e0'))
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Risk Factors
            c3, c4 = st.columns(2)
            
            with c3:
                st.subheader("📈 Risk Factor Weights")
                risk_data = pd.DataFrame({
                    'Factor': ['Spending', 'Loan', 'Risk Behavior', 'Investment (-)', 'Savings (-)'],
                    'Weight': [
                        weighted_scores['Spending'] * 1,
                        weighted_scores['Loan'] * 1.5,
                        weighted_scores['Risk'] * 2,
                        -weighted_scores['Investment'] * 0.6,
                        -weighted_scores['Savings'] * 0.5
                    ],
                    'Type': ['Risk', 'Risk', 'Risk', 'Stabilizer', 'Stabilizer']
                })
                colors = {'Risk': '#e17055', 'Stabilizer': '#00b894'}
                fig2 = px.bar(risk_data, x='Factor', y='Weight', color='Type',
                              color_discrete_map=colors)
                fig2.update_layout(
                    showlegend=True, height=300,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#a0a0c0',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    legend=dict(font=dict(color='#e0e0e0'))
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with c4:
                st.subheader("💰 Amount Distribution")
                amount_data = pd.DataFrame(list(total_amounts.items()), columns=['Category', 'Amount'])
                amount_data = amount_data[amount_data['Amount'] > 0]
                if not amount_data.empty:
                    fig_amt = px.bar(amount_data, x='Category', y='Amount',
                                    color='Category',
                                    color_discrete_sequence=['#00d2ff', '#6c5ce7', '#fd79a8', '#00b894', '#e17055'])
                    fig_amt.update_layout(
                        showlegend=False, height=300,
                        margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#a0a0c0',
                        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='₹ Amount')
                    )
                    st.plotly_chart(fig_amt, use_container_width=True)
                else:
                    st.info("No monetary amounts detected in posts.")
            
            # Recommendations
            st.subheader("💡 AI-Powered Bank Recommendations")
            
            if risk_score > 70:
                st.error("⚠️ **High Risk Profile Detected**")
                rec_cols = st.columns(4)
                recs = [
                    ("🔒", "Secured Credit Card", "Deposit-backed, low limit"),
                    ("📚", "Financial Literacy", "Budgeting courses"),
                    ("🔍", "3-Month Monitor", "Review before upgrade"),
                    ("💰", "Micro-Savings Plan", "Auto-debit ₹500/week")
                ]
            elif scores['Investment'] > scores['Spending'] and scores['Savings'] > 0:
                st.success("✅ **Stable Investor Profile**")
                recs = [
                    ("💎", "Premium Credit Card", "Rewards + lounge access"),
                    ("📊", "Wealth Advisory", "Portfolio management"),
                    ("📈", "Higher Credit Limit", "Pre-approved upgrade"),
                    ("🛡️", "Insurance Cross-sell", "Term + health cover")
                ]
            elif scores['Loan'] > 0 and scores['Savings'] == 0:
                st.warning("⚡ **Active Borrower, Low Savings**")
                recs = [
                    ("🔄", "Debt Consolidation", "Single lower-rate loan"),
                    ("🏦", "Auto Savings", "₹1000/month auto-debit"),
                    ("🎓", "Financial Coaching", "Emergency fund guide"),
                    ("💳", "Balance Transfer", "0% intro rate offer")
                ]
            else:
                st.info("ℹ️ **Average Profile — Growth Opportunity**")
                recs = [
                    ("💳", "Standard Credit Card", "2% cashback"),
                    ("📱", "Micro-Investments", "Start with ₹100/day"),
                    ("🎮", "Savings Challenge", "Gamified goals"),
                    ("📊", "Spending Insights", "Category-wise breakdown")
                ]
            
            rec_cols = st.columns(4)
            for col, (icon, title, desc) in zip(rec_cols, recs):
                with col:
                    st.markdown(f"""
                    <div class="flow-step">
                        <div class="icon">{icon}</div>
                        <div class="title">{title}</div>
                        <div class="desc">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Explainability
            with st.expander("🔍 Why This Risk Score? (Explainable AI)"):
                st.markdown(f"""
                **Risk Score Calculation (Confidence-Weighted):**
                | Factor | Posts | Weighted Score | Weight | Contribution |
                |--------|-------|---------------|--------|-------------|
                | Spending | {scores['Spending']} | {weighted_scores['Spending']:.2f} | ×1.0 | +{weighted_scores['Spending']*1:.2f} |
                | Loan | {scores['Loan']} | {weighted_scores['Loan']:.2f} | ×1.5 | +{weighted_scores['Loan']*1.5:.2f} |
                | Risk | {scores['Risk']} | {weighted_scores['Risk']:.2f} | ×2.0 | +{weighted_scores['Risk']*2:.2f} |
                | Investment | {scores['Investment']} | {weighted_scores['Investment']:.2f} | ×-0.6 | {-weighted_scores['Investment']*0.6:.2f} |
                | Savings | {scores['Savings']} | {weighted_scores['Savings']:.2f} | ×-0.5 | {-weighted_scores['Savings']*0.5:.2f} |
                
                **Final Score:** {int(risk_score)}/100
                **Sentiment Integration:** {'Enabled — negative sentiment boosts risk labels' if data.get('include_sentiment') else 'Disabled'}
                **Uncertain Posts:** {len([r for r in results if r['Category'] == 'Uncertain'])} (threshold: {confidence_threshold})
                """)
            
            # --- TRADITIONAL vs FINBEHAVIOR COMPARISON (Add-On #4) ---
            st.markdown("---")
            st.subheader("🆚 Traditional vs. FinBehavior Scoring")
            cmp1, cmp2 = st.columns(2)
            with cmp1:
                st.markdown("""
                <div style="background: rgba(255,100,100,0.08); border: 1px solid rgba(255,100,100,0.2); border-radius: 12px; padding: 20px;">
                    <h4 style="color: #ff6b6b;">📋 Traditional (CIBIL/Bureau)</h4>
                    <ul style="color: #a0a0c0;">
                        <li>Based on past loan repayment only</li>
                        <li>❌ <b>Rejects thin-file</b> — 400M Indians excluded</li>
                        <li>Lagging indicator (30–90 day delay)</li>
                        <li>No behavioral or lifestyle signals</li>
                        <li>Opaque scoring — black box</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with cmp2:
                st.markdown("""
                <div style="background: rgba(0,210,255,0.08); border: 1px solid rgba(0,210,255,0.2); border-radius: 12px; padding: 20px;">
                    <h4 style="color: #00d2ff;">🧠 FinBehavior AI</h4>
                    <ul style="color: #a0a0c0;">
                        <li>Based on <b>real-time behavior</b> signals</li>
                        <li>✅ <b>Includes underbanked</b> & first-time borrowers</li>
                        <li>Leading indicator — catches trends early</li>
                        <li>Consent-based social + spending signals</li>
                        <li>Fully explainable — open formula</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Download Report
            st.markdown("---")
            dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1])
            with dl_col2:
                pdf_bytes = generate_pdf_report(results, scores, risk_score, uncertainty_rate)
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"finbehavior_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        # --- TAB 2: DETAILS ---
        with tab2:
            st.subheader("📋 Detailed Post Classification")
            
            # Summary table
            results_df = pd.DataFrame(results)
            display_df = results_df[['Post', 'Category', 'Confidence', 'Amount', 'Sentiment']].copy()
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"₹{x:,.0f}" if x > 0 else "—")
            st.dataframe(display_df, use_container_width=True, height=300)
            
            # Expandable details
            for idx, row in results_df.iterrows():
                emoji_map = {"Spending": "💸", "Investment": "📈", "Loan": "🏦", "Savings": "💰", "Risk": "⚠️", "Uncertain": "❓"}
                emoji = emoji_map.get(row['Category'], "📝")
                
                with st.expander(f"{emoji} {row['Post'][:70]}{'...' if len(row['Post'])>70 else ''}"):
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Category", row['Category'])
                    d2.metric("Confidence", f"{row['Confidence']:.1%}")
                    d3.metric("Sentiment", row['Sentiment'])
                    d4.metric("Amount", f"₹{row['Amount']:,.0f}" if row['Amount'] > 0 else "—")
                    
                    # --- HUMAN-IN-THE-LOOP FLAGGING (Add-On #2) ---
                    if row['Category'] == 'Risk' or row['Confidence'] < 0.6:
                        flag_key = f"flag_{idx}_{row['Post'][:20]}"
                        if st.button("🚩 Flag for Manual Review", key=flag_key):
                            st.success("✅ Flagged for Loan Officer Review — alert sent to CRM")
                            st.caption("_In production, this triggers an alert in the bank's Case Management System_")
                    
                    if row['All_Scores']:
                        score_df = pd.DataFrame(list(row['All_Scores'].items()), columns=['Label', 'Score'])
                        fig_bar = px.bar(score_df, x='Label', y='Score',
                                        color='Score', color_continuous_scale='Viridis')
                        fig_bar.update_layout(
                            height=200, margin=dict(t=10, b=10, l=10, r=10),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#a0a0c0',
                            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                        )
                        st.plotly_chart(fig_bar, use_container_width=True, key=f"detail_bar_{idx}")
        
        # --- TAB 3: MODEL EVALUATION ---
        with tab3:
            st.subheader("🔬 Model Accuracy Evaluation")
            st.caption("Evaluates the zero-shot model against synthetic labeled data")
            
            if st.button("▶️ Run Evaluation (25 samples)", type="primary"):
                with st.spinner("🧠 Loading model for evaluation..."):
                    eval_classifier = load_classifier()
                with st.spinner("Running model on labeled test data..."):
                    cm, report, eval_labels = run_model_evaluation(eval_classifier)
                
                if cm is not None:
                    ev1, ev2 = st.columns(2)
                    
                    with ev1:
                        st.markdown("**Confusion Matrix**")
                        fig_cm, ax = plt.subplots(figsize=(6, 5))
                        fig_cm.patch.set_facecolor('#0f0c29')
                        ax.set_facecolor('#0f0c29')
                        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                        ax.set_xticks(range(len(eval_labels)))
                        ax.set_yticks(range(len(eval_labels)))
                        ax.set_xticklabels(eval_labels, rotation=45, ha='right', color='#e0e0e0', fontsize=9)
                        ax.set_yticklabels(eval_labels, color='#e0e0e0', fontsize=9)
                        ax.set_xlabel('Predicted', color='#a0a0c0')
                        ax.set_ylabel('True', color='#a0a0c0')
                        for i in range(len(eval_labels)):
                            for j in range(len(eval_labels)):
                                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                                       color='white' if cm[i, j] > cm.max()/2 else '#333', fontweight='bold')
                        fig_cm.colorbar(im, ax=ax, shrink=0.8)
                        plt.tight_layout()
                        st.pyplot(fig_cm)
                    
                    with ev2:
                        st.markdown("**Per-Category Metrics**")
                        metrics_data = []
                        for label in eval_labels:
                            if label in report:
                                metrics_data.append({
                                    "Category": label,
                                    "Precision": f"{report[label]['precision']:.0%}",
                                    "Recall": f"{report[label]['recall']:.0%}",
                                    "F1-Score": f"{report[label]['f1-score']:.0%}",
                                    "Support": int(report[label]['support'])
                                })
                        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                        
                        # Overall accuracy
                        if 'accuracy' in report:
                            st.metric("Overall Accuracy", f"{report['accuracy']:.0%}")
                        
                        st.markdown("""
                        **What this means:**
                        - **Precision**: Of predicted X, how many were actually X?
                        - **Recall**: Of actual X, how many did we find?
                        - **F1-Score**: Harmonic mean of precision & recall
                        - Higher is better. >70% is good for zero-shot.
                        """)
                else:
                    st.warning("⚠️ `synthetic_posts.json` not found. Run `generate_data.py` first.")
        
        # --- TAB 4: HISTORY ---
        with tab4:
            st.subheader("📜 Analysis History")
            if st.session_state.analysis_history:
                hist_df = pd.DataFrame(st.session_state.analysis_history)
                fig_hist = px.line(hist_df, x='timestamp', y='risk_score', markers=True,
                                  labels={'risk_score': 'Risk Score', 'timestamp': 'Time'})
                fig_hist.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#a0a0c0',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 100])
                )
                fig_hist.update_traces(line_color='#00d2ff', marker_color='#6c5ce7')
                st.plotly_chart(fig_hist, use_container_width=True)
                st.dataframe(hist_df, use_container_width=True)
            else:
                st.info("No analysis history yet. Run an analysis to see trends.")
        
        # Performance Metrics
        with st.expander("⚡ Performance Metrics"):
            perf1, perf2, perf3 = st.columns(3)
            perf1.metric("Total Time", f"{elapsed:.1f}s")
            perf2.metric("Avg per Post", f"{elapsed/len(results)*1000:.0f}ms")
            perf3.metric("Posts Analyzed", len(results))
        
        # --- REGULATORY COMPLIANCE BADGES (Add-On #5) ---
        st.markdown("---")
        badge1, badge2, badge3, badge4 = st.columns(4)
        with badge1:
            st.markdown("""
            <div style="text-align:center; padding:10px; background:rgba(0,210,255,0.06); border-radius:10px; border:1px solid rgba(0,210,255,0.15);">
                <div style="font-size:1.5rem;">🛡️</div>
                <div style="color:#00d2ff; font-weight:600; font-size:0.85rem;">DPDP Act 2023</div>
                <div style="color:#666; font-size:0.7rem;">Compliant</div>
            </div>
            """, unsafe_allow_html=True)
        with badge2:
            st.markdown("""
            <div style="text-align:center; padding:10px; background:rgba(108,92,231,0.06); border-radius:10px; border:1px solid rgba(108,92,231,0.15);">
                <div style="font-size:1.5rem;">🏛️</div>
                <div style="color:#6c5ce7; font-weight:600; font-size:0.85rem;">RBI Digital Lending</div>
                <div style="color:#666; font-size:0.7rem;">Guidelines Ready</div>
            </div>
            """, unsafe_allow_html=True)
        with badge3:
            st.markdown("""
            <div style="text-align:center; padding:10px; background:rgba(0,200,83,0.06); border-radius:10px; border:1px solid rgba(0,200,83,0.15);">
                <div style="font-size:1.5rem;">🔐</div>
                <div style="color:#00c853; font-weight:600; font-size:0.85rem;">ISO 27001</div>
                <div style="color:#666; font-size:0.7rem;">Architecture Ready</div>
            </div>
            """, unsafe_allow_html=True)
        with badge4:
            st.markdown("""
            <div style="text-align:center; padding:10px; background:rgba(255,193,7,0.06); border-radius:10px; border:1px solid rgba(255,193,7,0.15);">
                <div style="font-size:1.5rem;">🇪🇺</div>
                <div style="color:#ffc107; font-weight:600; font-size:0.85rem;">GDPR</div>
                <div style="color:#666; font-size:0.7rem;">By Design</div>
            </div>
            """, unsafe_allow_html=True)
        st.caption("🔒 **Privacy & Ethics**: Consent-based analysis • No data stored • Explainable scores • Bias-audited • Human-in-the-loop oversight")
    
    else:
        # --- WELCOME / LANDING PAGE ---
        st.markdown("")
        
        # Architecture Flow
        st.subheader("🏗️ How It Works")
        flow_cols = st.columns(5)
        steps = [
            ("📱", "1. Input", "Social media posts with user consent"),
            ("🧹", "2. Preprocess", "Clean text, extract amounts"),
            ("🤖", "3. NLP Engine", "Zero-shot + sentiment classification"),
            ("📊", "4. Score", "Confidence-weighted risk calculation"),
            ("💡", "5. Recommend", "Personalized bank products")
        ]
        for col, (icon, title, desc) in zip(flow_cols, steps):
            with col:
                st.markdown(f"""
                <div class="flow-step">
                    <div class="icon">{icon}</div>
                    <div class="title">{title}</div>
                    <div class="desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            ### 🎯 Features
            - **Zero-Shot NLP** — No training data needed
            - **Sentiment-Aware Scoring** — Emotional context matters
            - **Amount Extraction** — ₹ values impact risk weight
            - **Confidence Thresholds** — Filter uncertain predictions
            - **Explainable AI** — See exactly why each score
            - **PDF Reports** — Download professional reports
            - **Model Evaluation** — Confusion matrix & F1 scores
            """)
        with col_b:
            st.markdown("""
            ### 🔒 Ethics by Design
            - ✅ Explicit **user consent** before analysis
            - ✅ **No data storage** — ephemeral processing
            - ✅ **Explainable scores** — full calculation visible
            - ✅ **Bias monitoring** — model evaluation tab
            - ✅ **GDPR & DPDP** Act 2023 compliant
            - ✅ **Appealable** — scores can be challenged
            - ✅ **Open model** — no black-box proprietary AI
            """)
        
        st.markdown("")
        st.info("👈 **Get Started**: Enter posts in the sidebar and click **'🚀 Analyze Behavior'**")


if __name__ == "__main__":
    main()
