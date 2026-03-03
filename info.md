# 🏦 FinBehavior AI — Project Documentation

> **Social Behavioral Credit Intelligence Platform**
> Analyze financial behavior from social media posts using AI — Consent-Based & Privacy-First

---

## 📁 Project Structure

```
FinBehaviorAI/
├── app.py                  # Main Streamlit application (~1000 lines)
│                            ├── Custom CSS dark theme
│                            ├── Consent flow
│                            ├── Preprocessing & amount extraction
│                            ├── NLP classification engine
│                            ├── Confidence-weighted scoring
│                            ├── Dashboard with charts & radar
│                            ├── PDF report generation
│                            ├── Model evaluation (confusion matrix)
│                            ├── ROI Calculator, HITL flagging
│                            ├── Traditional vs FinBehavior comparison
│                            └── Regulatory compliance badges
│
├── api.py                  # FastAPI REST API endpoint
│                            ├── POST /analyze — single post
│                            ├── POST /analyze/batch — multiple posts + risk score
│                            ├── GET /health — health check
│                            └── GET /docs — Swagger UI
│
├── generate_data.py        # Synthetic data generator (88 lines)
│                            └── Generates 75 labeled posts across 5 categories
│
├── synthetic_posts.json    # Generated demo dataset (75 posts)
│                            └── Each post has: text, true_label, timestamp, source
│
├── app_minimal.py          # Emergency fallback app (9 lines)
│                            └── Static demo if main app fails
│
├── requirements.txt        # Python dependencies (10 packages)
├── info.md                 # This file — project documentation
└── venv/                   # Python virtual environment
```

---

## 🧠 How It Works — Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. INPUT    │───▶│ 2. PREPROCESS│───▶│  3. NLP ENGINE  │───▶│  4. SCORING  │───▶│ 5. OUTPUT    │
│              │    │              │    │                 │    │              │    │              │
│ • Text paste │    │ • Lowercase  │    │ • Zero-shot     │    │ • Confidence │    │ • Risk score │
│ • CSV upload │    │ • Remove URLs│    │   classification│    │   weighting  │    │ • Radar chart│
│ • JSON upload│    │ • Remove @/# │    │ • Sentiment     │    │ • Amount     │    │ • PDF report │
│ • Demo data  │    │ • Extract ₹  │    │   analysis      │    │   extraction │    │ • Recommend. │
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘    └──────────────┘
```

---

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit 1.28+ | Dashboard UI, widgets, layout |
| **REST API** | FastAPI + Uvicorn | Bank integration endpoint |
| **Styling** | Custom CSS | Dark glassmorphism theme, animations |
| **NLP Classifier** | `typeform/distilbert-base-uncased-mnli` | Zero-shot classification into 5 categories |
| **Sentiment Model** | `distilbert-base-uncased-finetuned-sst-2-english` | Positive/Negative emotional tone |
| **Charts** | Plotly Express + Graph Objects | Pie chart, radar chart, bar charts, line chart |
| **Data Processing** | Pandas, NumPy | Dataframes, numerical ops |
| **ML Evaluation** | scikit-learn | Confusion matrix, precision/recall/F1 |
| **PDF Generation** | fpdf2 | Downloadable PDF reports |
| **Visualization** | Matplotlib | Confusion matrix heatmap |
| **Language** | Python 3.10+ | Core language |

---

## 🧹 Preprocessing Pipeline

The `preprocess_text()` function cleans social media posts before NLP analysis:

```python
def preprocess_text(text):
    text = text.lower()                                    # Step 1: Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)   # Step 2: Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)                 # Step 3: Remove @mentions & #hashtags
    text = re.sub(r'[^\w\s₹]', '', text)                  # Step 4: Remove special chars (keep ₹)
    text = ' '.join(text.split())                          # Step 5: Normalize whitespace
    return text
```

**Why each step:**
1. **Lowercase** → Normalizes "BOUGHT" and "bought" to same token
2. **Remove URLs** → Social posts often have links that add noise
3. **Remove @/#** → Mentions/hashtags don't carry financial intent
4. **Keep ₹** → Needed for the amount extraction step
5. **Normalize spaces** → Reduces tokenization issues

### Amount Extraction (`extract_amount()`)

```python
# Detects monetary values in Indian formats:
# ₹50,000  →  50000
# Rs. 2L   →  200000
# 5000 rupees → 5000
# ₹10k     →  10000
# ₹1cr     →  10000000
```

Supports: `₹`, `Rs.`, `INR`, `rupees` + suffixes `k`, `L/lakh/lac`, `cr`

---

## 🤖 AI Models Used

### 1. Zero-Shot Classifier
- **Model**: `typeform/distilbert-base-uncased-mnli`
- **Type**: Zero-shot text classification (no training needed)
- **How it works**: Given a text and candidate labels, predicts which label fits best
- **Labels**: `["Spending", "Investment", "Loan", "Savings", "Risk"]`
- **Why zero-shot**: No labeled financial data required; works out-of-the-box

### 2. Sentiment Analysis
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Type**: Binary sentiment (POSITIVE / NEGATIVE)
- **Integration**: Sentiment adjusts risk scores:
  - Negative sentiment + Spending/Loan/Risk → **boost confidence by 15%** (higher risk)
  - Positive sentiment + Investment/Savings → **boost confidence by 10%** (lower risk)

---

## 📊 Scoring Algorithm

### Step 1: Classify each post
```
Post → Preprocess → Zero-Shot Model → Category + Confidence Score
```

### Step 2: Confidence Weighting
Instead of just counting posts per category, each post's contribution is weighted by its confidence:
```python
weighted_scores[label] += confidence  # e.g., "Risk" at 0.92 contributes 0.92, not just 1
```

### Step 3: Amount Factor
If monetary amounts are detected, the ratio of risky-to-safe spending amplifies risk:
```python
amount_factor = min(risky_amount / safe_amount, 3.0)  # Capped at 3x
```

### Step 4: Final Risk Score
```python
base_risk = (spending × 1.0) + (loan × 1.5) + (risk × 2.0)
stabilizer = (investment × 1.2) + (savings × 1.0)
risk_score = ((base_risk - stabilizer × 0.5) / total_posts) × 50 × (1 + amount_factor × 0.1)
```
- Range: **0–100**
- < 30 = Low Risk 🟢
- 30–70 = Medium Risk 🟡
- > 70 = High Risk 🔴

---

## 🔐 Consent & Ethics Design

| Principle | Implementation |
|-----------|---------------|
| **User Consent** | Mandatory consent screen before any analysis |
| **No Data Storage** | All processing is ephemeral — nothing saved to disk/cloud |
| **Explainable AI** | "Why This Risk Score?" section shows full calculation |
| **Bias Monitoring** | Model Evaluation tab with confusion matrix & F1 scores |
| **Appealable Scores** | Scores are transparent and can be challenged |
| **GDPR/DPDP** | No PII collected, no external API calls, local-only processing |

---

## 📋 Features List

### Core Features
- ✅ Zero-shot NLP classification (5 financial categories)
- ✅ Sentiment analysis integration (positive/negative)
- ✅ Confidence-weighted risk scoring (0–100)
- ✅ Monetary amount extraction (₹ values)
- ✅ Adjustable confidence threshold slider

### Dashboard & Visualization
- ✅ Donut chart — behavior distribution
- ✅ Radar/spider chart — profile vs benchmark comparison
- ✅ Bar chart — risk factor weights (risk vs stabilizers)
- ✅ Bar chart — amount distribution by category
- ✅ Line chart — analysis history (risk score trend)
- ✅ Metric cards — risk score, spending, investment, loan, uncertainty

### Input Options
- ✅ Text input (paste posts line-by-line)
- ✅ CSV file upload (with column selector)
- ✅ JSON file upload (auto-detects format)
- ✅ Pre-loaded demo data with 10 realistic Indian financial posts

### AI & Analysis
- ✅ Model evaluation tab — confusion matrix + precision/recall/F1
- ✅ Per-post expandable details with score breakdown bar chart
- ✅ Sentiment-adjusted confidence scoring
- ✅ AI-powered bank product recommendations (4 profile types)

### Output & Export
- ✅ Downloadable PDF report with full analysis
- ✅ Explainable AI section — full risk calculation breakdown
- ✅ Performance metrics — total time, avg per post

### UX & Design
- ✅ Dark glassmorphism CSS theme
- ✅ Gradient header with branded colors
- ✅ Hover animations on metric cards
- ✅ Consent flow before dashboard access
- ✅ Session state persistence (results survive widget interactions)
- ✅ Rotating status messages during analysis
- ✅ Hidden Streamlit branding (MainMenu, footer, header)
- ✅ 5-step architecture flow on landing page

### 🚀 5 Killer Add-Ons
- ✅ **FastAPI REST API** (`api.py`) — `/analyze` and `/analyze/batch` endpoints with Swagger UI
- ✅ **Human-in-the-Loop** — 🚩 "Flag for Manual Review" button on risky/low-confidence posts
- ✅ **ROI Calculator** — Sidebar widget showing potential bank savings in ₹ Crores
- ✅ **Traditional vs FinBehavior** — Side-by-side comparison (CIBIL vs our approach)
- ✅ **Regulatory Badges** — DPDP 2023, RBI Digital Lending, ISO 27001, GDPR compliance badges

---

## 🏗️ Key Functions Reference

| Function | File | Purpose |
|----------|------|---------|
| `preprocess_text(text)` | app.py:174 | Clean social media text for NLP |
| `extract_amount(text)` | app.py:184 | Extract ₹ values from post text |
| `load_classifier()` | app.py:209 | Load & cache zero-shot model |
| `load_sentiment_model()` | app.py:213 | Load & cache sentiment model |
| `calculate_scores(classifications)` | app.py:221 | Confidence-weighted risk scoring |
| `_pdf_safe(text)` | app.py:259 | Sanitize Unicode for PDF output |
| `generate_pdf_report(...)` | app.py:276 | Generate downloadable PDF |
| `run_model_evaluation(classifier)` | app.py:323 | Confusion matrix on synthetic data |
| `main()` | app.py:355 | Main application entry point |
| `generate_post(category)` | generate_data.py:53 | Generate synthetic social posts |

---

## 🚀 How to Run

```bash
# 1. Navigate to project
cd FinBehaviorAI

# 2. Activate virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 3. Install dependencies (first time only)
pip install -r requirements.txt

# 4. Generate synthetic data (optional, already pre-generated)
python generate_data.py

# 5. Launch the Streamlit dashboard
streamlit run app.py
# → http://localhost:8501

# 6. (Optional) Launch the REST API
uvicorn api:app --reload
# → http://localhost:8000/docs (Swagger UI)
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.28.0 | Web dashboard framework |
| `transformers` | ≥4.35.0 | Hugging Face NLP models |
| `torch` | ≥2.0.0 | PyTorch (model backend) |
| `pandas` | ≥2.0.0 | DataFrames & data manipulation |
| `plotly` | ≥5.18.0 | Interactive charts |
| `scikit-learn` | ≥1.3.0 | Confusion matrix, metrics |
| `fpdf2` | ≥2.7.0 | PDF report generation |
| `matplotlib` | ≥3.8.0 | Confusion matrix heatmap |
| `fastapi` | ≥0.100.0 | REST API framework |
| `uvicorn` | ≥0.23.0 | ASGI server for FastAPI |

---

## 🎯 Business Use Cases

1. **Banks**: Alternative credit scoring for thin-file/underbanked customers
2. **Neo-banks**: Behavior-based product recommendations
3. **Insurance**: Risk profiling from lifestyle signals
4. **FinTech**: Automated financial health assessments
5. **Regulators**: Consent-first behavioral analytics framework

---

## 📝 License & Disclaimer

This is a **research prototype** built for educational/hackathon purposes.
- Not intended for production credit decisions
- All AI models are open-source (Hugging Face)
- No real user data is processed or stored
- Scores are experimental and should not be used for actual lending decisions

---

## 🔌 REST API Reference (`api.py`)

### Start the API
```bash
uvicorn api:app --reload
# → Swagger UI at http://localhost:8000/docs
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Analyze single post |
| `POST` | `/analyze/batch` | Analyze multiple posts + risk score |

### Example Request
```json
POST /analyze
{
  "text": "Just invested ₹50,000 in Nifty 50 index fund",
  "consent": true,
  "include_sentiment": true,
  "confidence_threshold": 0.5
}
```

### Example Response
```json
{
  "text": "Just invested ₹50,000 in Nifty 50 index fund",
  "category": "Investment",
  "confidence": 0.891,
  "sentiment": "POSITIVE",
  "sentiment_score": 0.987,
  "all_scores": {
    "Investment": 0.891,
    "Savings": 0.042,
    "Spending": 0.031,
    "Loan": 0.021,
    "Risk": 0.015
  }
}
```

> ⚠️ Consent must be `true` or the API returns 403. This enforces ethical use.

---

## 🎤 Pitch Script

> *"Meet Rahul. He earns ₹50k/month, pays rent on time, invests in SIPs. But he has no credit card. CIBIL rejects him. Our tool reads his **consented** financial behavior, gives him a score, and gets him the loan. Banks reduce defaults; Rahul gets access. Everyone wins."*

