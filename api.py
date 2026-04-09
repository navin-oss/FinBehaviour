"""
FinBehavior AI — REST API Endpoint
Run with: uvicorn api:app --reload
Swagger UI: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
from typing import Optional
import re

# --- Initialize FastAPI ---
app = FastAPI(
    title="FinBehavior AI API",
    description="Consent-Based Social Behavioral Credit Intelligence API",
    version="1.0.0",
    docs_url="/docs"
)

# --- Load Model (once at startup) ---
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

LABELS = ["Spending", "Investment", "Loan", "Savings", "Risk"]


# --- Request/Response Models ---
class PostInput(BaseModel):
    text: str = Field(..., description="Social media post text to analyze", min_length=3)
    consent: bool = Field(..., description="User must consent to analysis (must be True)")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    confidence_threshold: float = Field(default=0.5, ge=0.3, le=0.9, description="Minimum confidence threshold")

class BatchInput(BaseModel):
    posts: list[str] = Field(..., description="List of social media posts", min_length=1, max_length=100)
    consent: bool = Field(..., description="User must consent to analysis")
    include_sentiment: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.5, ge=0.3, le=0.9)

class PostResult(BaseModel):
    text: str
    category: str
    confidence: float
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    all_scores: dict

class BatchResult(BaseModel):
    results: list[PostResult]
    risk_score: float
    profile: str
    total_posts: int
    confident_posts: int
    recommendation: str


# --- Helper Functions ---
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s₹]', '', text)
    return ' '.join(text.split())


def analyze_single(text: str, include_sentiment: bool, threshold: float) -> PostResult:
    clean = preprocess(text)
    output = classifier(clean, candidate_labels=LABELS, multi_label=False)
    
    # Apply heuristic boosts based on explicit keywords
    scores_dict = {k: v for k, v in zip(output['labels'], output['scores'])}
    
    # Rule 1: Explicit spending keywords
    spending_keywords = ['bought', 'paid', 'purchased', 'spent', 'splurged', 'ordered', 'booked']
    if any(word in clean for word in spending_keywords):
        # Boost Spending significantly to ensure it exceeds threshold
        scores_dict['Spending'] += 0.5
        
        # Normalize scores to sum to 1
        total_score = sum(scores_dict.values())
        for k in scores_dict:
            scores_dict[k] /= total_score
            
        # Re-sort lists
        sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        output['labels'] = [k for k, v in sorted_items]
        output['scores'] = [v for k, v in sorted_items]
    
    confidence = output['scores'][0]
    category = output['labels'][0] if confidence >= threshold else "Uncertain"
    
    sentiment = None
    sentiment_score = None
    if include_sentiment:
        sent = sentiment_model(clean[:512])[0]
        sentiment = sent['label']
        sentiment_score = round(sent['score'], 3)
    
    return PostResult(
        text=text,
        category=category,
        confidence=round(confidence, 3),
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        all_scores={k: round(v, 3) for k, v in zip(output['labels'], output['scores'])}
    )


# --- Endpoints ---
@app.get("/")
def root():
    return {
        "service": "FinBehavior AI API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": ["/analyze", "/analyze/batch", "/health"]
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": True}


@app.post("/analyze", response_model=PostResult)
def analyze_post(post: PostInput):
    """Analyze a single social media post for financial behavior"""
    if not post.consent:
        raise HTTPException(status_code=403, detail="User consent is required. Set consent=true.")
    return analyze_single(post.text, post.include_sentiment, post.confidence_threshold)


@app.post("/analyze/batch", response_model=BatchResult)
def analyze_batch(batch: BatchInput):
    """Analyze multiple posts and return aggregated risk score"""
    if not batch.consent:
        raise HTTPException(status_code=403, detail="User consent is required. Set consent=true.")
    
    results = [analyze_single(text, batch.include_sentiment, batch.confidence_threshold) for text in batch.posts]
    
    # Calculate risk score
    scores = {"Spending": 0, "Investment": 0, "Loan": 0, "Savings": 0, "Risk": 0}
    for r in results:
        if r.category in scores:
            scores[r.category] += 1
    
    total = sum(scores.values())
    confident = len([r for r in results if r.category != "Uncertain"])
    
    if total == 0:
        risk_score = 0.0
    else:
        risk_score = min(((scores['Spending'] * 1 + scores['Loan'] * 1.5 + scores['Risk'] * 2) / total) * 100, 100)
    
    profile = "HIGH_RISK" if risk_score > 70 else "LOW_RISK" if risk_score < 30 else "MEDIUM_RISK"
    
    rec_map = {
        "HIGH_RISK": "Secured Credit Card + Financial Literacy Program",
        "MEDIUM_RISK": "Standard Credit Card + Spending Insights",
        "LOW_RISK": "Premium Credit Card + Wealth Advisory"
    }
    
    return BatchResult(
        results=results,
        risk_score=round(risk_score, 1),
        profile=profile,
        total_posts=len(batch.posts),
        confident_posts=confident,
        recommendation=rec_map[profile]
    )
