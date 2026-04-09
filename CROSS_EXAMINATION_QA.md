# FinBehaviour AI — Extreme-Level Cross-Examination Q&A

> **For Judges, Examiners & Viva Voce**  
> Hard-hitting questions on feasibility, ethics, scalability, and real-world applicability — with prepared answers.

---

## 1. FEASIBILITY & REAL-WORLD APPLICABILITY

### Q1.1: "Social media posts are inherently noisy and sarcastic. How can you claim your risk score is reliable when 'Just blew ₹50k on crypto, no regrets' could be a joke, a flex, or genuine distress?"

**Answer:**  
We acknowledge this is a fundamental limitation. Our system does three things to mitigate it:

1. **Confidence-weighted scoring** — Low-confidence classifications contribute less to the final risk score. Posts with ambiguous intent get down-weighted.
2. **Human-in-the-Loop (HITL)** — We explicitly flag low-confidence or high-risk posts for manual review. Banks don't auto-reject; they get a "Flag for Review" signal.
3. **Transparency** — We show the full breakdown (category, confidence, sentiment) so loan officers can see *why* a score is high and override if context suggests sarcasm.

We position this as a **supplementary signal**, not a replacement for CIBIL. It's meant for thin-file customers where traditional data is absent — any signal is better than none, as long as it's explainable and appealable.

---

### Q1.2: "Your synthetic dataset has only 75 posts. How can you validate a production system on such a tiny sample?"

**Answer:**  
The 75-post synthetic set is for **demo and model evaluation only** — to show the confusion matrix and F1 scores in the UI. It is not used for training (we use zero-shot models).

For real validation, we would need:
- **Partner bank pilot** with consented, anonymized real posts
- **Benchmark against CIBIL** — correlate our risk score with actual default rates over 12–24 months
- **A/B testing** — approve some "marginal" applicants based on our score and track repayment

We explicitly state in our documentation that this is a **research prototype** and not production-ready. The evaluation tab is illustrative, not conclusive.

---

### Q1.3: "Zero-shot models weren't trained on financial text. Why should a bank trust them for credit decisions?"

**Answer:**  
Zero-shot models like MNLI-based classifiers work by **natural language inference** — they map text to semantic concepts. "I'm drowning in EMI" implies entailment with "Loan" and "Risk" even without financial fine-tuning.

**Trade-off we accept:**  
- **Pro:** No labeled financial data needed; works out-of-the-box; no cold-start.
- **Con:** Lower precision on domain-specific slang (e.g., "chillar" for small change, "FD" for fixed deposit).

**Our roadmap** explicitly includes migrating to **FinBERT** or a custom fine-tuned model on Indian financial corpus. For an MVP, zero-shot gives us a working baseline that can be improved iteratively.

---

## 2. ETHICS & BIAS

### Q2.1: "Your system could discriminate against people who post about loans or spending more often — even if they're financially stable. Isn't this algorithmic bias?"

**Answer:**  
Yes, there is a risk. We mitigate it in several ways:

1. **Consent-first** — We only analyze data the user explicitly consents to share. No scraping without permission.
2. **Explainable AI** — Every score shows the full calculation: which posts contributed, with what confidence. Users can appeal and understand *why* they were scored.
3. **Bias monitoring** — The Model Evaluation tab shows confusion matrix and per-class F1. We can detect if certain categories (e.g., "Loan") are over-predicted.
4. **No protected attributes** — We do not use age, gender, religion, or location. The input is text only.

We do **not** claim to be bias-free. We claim to be **transparent and auditable**. Regulators (RBI, DPDP) require explainability — we provide it.

---

### Q2.2: "What if someone has a mental health crisis and posts 'I'm going to max out my cards'? You'd flag them as high risk and potentially deny credit when they need help."

**Answer:**  
This is a valid ethical concern. Our design choices:

1. **HITL flagging** — High-risk + high-confidence posts get a "Flag for Manual Review" button. A human can see the full context and escalate to support/counseling instead of auto-rejecting.
2. **No auto-denial** — We never recommend "Reject." We recommend "Review" or "Proceed with caution." The final decision stays with the bank.
3. **Future enhancement** — We could add a "Distress" or "Crisis" category that triggers a different workflow (e.g., connect to financial wellness programs) instead of pure risk scoring.

We acknowledge this edge case and position our tool as **decision support**, not decision automation.

---

## 3. TECHNICAL & SCALABILITY

### Q3.1: "DistilBERT runs on CPU and takes ~2–3 seconds per post. How will you scale to millions of users?"

**Answer:**  
Current limitations:
- **Single-threaded inference** — One post at a time per worker.
- **Model size** — ~250MB per model (classifier + sentiment). Fits in memory but adds latency.

**Scaling strategies:**
1. **Batch inference** — Process 8–16 posts per GPU batch. Our API already supports `/analyze/batch`.
2. **Model quantization** — Use ONNX or TensorRT to reduce size and latency by 2–3x.
3. **Horizontal scaling** — Stateless API; we can run multiple Uvicorn workers behind a load balancer.
4. **Caching** — For repeated posts (e.g., same user re-checking), cache results with a short TTL.
5. **Async queue** — For non-real-time use, push to Redis/RabbitMQ and process asynchronously.

For a pilot with 10K users and ~50 posts each, we'd need ~500K inferences. At 2 sec/post on 10 workers, that's ~28 hours. With batching and GPU, we can bring it to under 2 hours.

---

### Q3.2: "You remove URLs, @mentions, and hashtags. Don't you lose critical context? A link to a gambling site is very different from a link to a mutual fund."

**Answer:**  
Yes, we lose that context. Our preprocessing is intentionally aggressive to:
1. **Reduce noise** — URLs often point to unrelated content; including them can confuse the model.
2. **Avoid external calls** — We don't fetch URL content (privacy, latency, security).
3. **Keep pipeline simple** — No dependency on external services.

**Trade-off:** We sacrifice URL-based signals for simplicity and privacy. A future enhancement could use a **URL classifier** (e.g., categorize domains as gambling, investment, news) and feed that as a separate feature. For MVP, we rely on text semantics only.

---

## 4. LEGAL & REGULATORY

### Q4.1: "RBI's digital lending guidelines require lenders to have a clear credit assessment framework. Does your black-box AI comply?"

**Answer:**  
We are **not** a black box. Our "Explainable AI" section shows:
- Per-post category and confidence
- Sentiment and its impact
- Amount extraction and risky/safe ratio
- Full formula: `base_risk`, `stabilizer`, `amount_factor`

RBI guidelines emphasize **transparency** and **appealability**. We provide both. The score is reproducible — given the same inputs, the same formula yields the same output. We don't use opaque deep learning for the final score; we use a deterministic formula on top of model outputs.

---

### Q4.2: "DPDP 2023 requires consent, purpose limitation, and data minimization. How do you comply?"

**Answer:**  
1. **Consent** — Mandatory consent screen before any analysis. API returns 403 if `consent=false`.
2. **Purpose limitation** — We only use data for behavioral risk assessment. No secondary use, no marketing.
3. **Data minimization** — We process only the text provided. No PII collection (no name, phone, Aadhaar).
4. **Zero storage** — All processing is in-memory. Nothing is persisted to disk or cloud. Data is purged after the session.

We display DPDP 2023 and GDPR badges in the UI to signal alignment. For production, a formal DPIA (Data Protection Impact Assessment) would be required.

---

## 5. SECURITY & PRIVACY

### Q5.1: "You claim 'zero storage.' What about logs? Could a malicious insider or attacker extract user posts from your system?"

**Answer:**  
- **Streamlit app** — Session state is in-memory; no database. On server restart, data is gone.
- **API** — No logging of post content. We could add audit logs (hash of post, timestamp, risk score) for compliance, but we'd need to ensure logs don't contain reconstructable PII.
- **Production hardening** — We'd need: encrypted transit (HTTPS), rate limiting, API key auth, and a clear data retention policy for any logs we do keep.

For the prototype, we minimize attack surface by not storing anything. Production would require a full security review.

---

### Q5.2: "What if someone feeds your API millions of posts to infer training data or reverse-engineer your model?"

**Answer:**  
- **Rate limiting** — Not implemented in prototype; would be essential in production (e.g., 100 requests/minute per API key).
- **Model extraction** — Our models are public (Hugging Face). There's nothing to steal. The value is in the pipeline (preprocessing, scoring formula, thresholds), not the model weights.
- **Abuse detection** — Unusual patterns (e.g., 10K requests from one IP) would trigger alerts.

---

## 6. BUSINESS & ECONOMICS

### Q6.1: "Banks already have CIBIL. Why would they pay for your product?"

**Answer:**  
- **Thin-file / credit-invisible** — ~400M Indians have no credit history. CIBIL can't score them. We offer an alternative signal.
- **Complementary, not replacement** — We position as "CIBIL + FinBehavior" for a fuller picture. For established customers, CIBIL dominates. For new-to-credit, we fill the gap.
- **Neo-banks and FinTech** — They serve younger, digital-native users who may have sparse traditional data but rich social footprints. We target them first.

---

### Q6.2: "What's your unit economics? How much does each analysis cost?"

**Answer:**  
- **Compute** — ~2 sec/post on CPU. On a $0.50/hr cloud instance, that's ~$0.0003 per post. For 50 posts/user: ~$0.015 per user.
- **Model hosting** — Models load once; marginal cost per request is negligible.
- **Pricing model** — We could charge per analysis (e.g., ₹5–10 per user) or SaaS (e.g., ₹50K/month for 10K analyses). At ₹10/analysis, gross margin is high.

---

## 7. EDGE CASES & LIMITATIONS

### Q7.1: "What if a user posts in Hindi, Hinglish, or regional languages?"

**Answer:**  
Our models (`distilbert-base-uncased-mnli`, `distilbert-sst-2`) are **English-only**. Hindi/Hinglish would be misclassified or get very low confidence.

**Mitigation:**  
- Preprocessing could detect language and route to a multilingual model (e.g., `xlm-roberta`) or Hindi-specific model.
- Our roadmap includes "Multi-Lingual Expansion" for Indian languages.
- For now, we document this as a limitation and assume English/Hinglish-heavy input for the demo.

---

### Q7.2: "Your amount extraction handles '₹50k' and '2L' but what about 'fifty thousand' or '2 lakhs in words'?"

**Answer:**  
We use regex for numeric patterns: `₹`, `Rs.`, `INR`, `k`, `L`, `lakh`, `cr`. We do **not** parse "fifty thousand" or "two lakhs" in words.

**Gap:** Text like "I spent fifty thousand on a phone" would not contribute to the amount factor. This is a known limitation. A future enhancement could use NER (Named Entity Recognition) or a number-word parser for Indian English.

---

### Q7.3: "What if someone deliberately posts fake 'good' content to game the system?"

**Answer:**  
**Adversarial gaming** is a real risk. Mitigations:
1. **Volume** — We analyze multiple posts. Faking 50+ consistent "investment" posts is harder than faking one.
2. **Sentiment consistency** — Fake posts might have unnatural sentiment patterns (e.g., all overly positive).
3. **Cross-validation** — In production, we'd combine with bank transaction data (via Account Aggregator) to validate. Social-only scoring would be a red flag for "too good to be true" profiles.
4. **Temporal patterns** — Sudden shift from "loan stress" to "investment guru" in a week could be flagged.

We don't claim to be game-proof. We claim to raise the bar for gaming compared to having no behavioral signal.

---

## 8. COMPETITIVE & ACADEMIC

### Q8.1: "Companies like Lenddo and Cigniti already do alternative credit scoring. What's novel about your approach?"

**Answer:**  
- **Consent-first, zero-storage** — Many players scrape data. We require explicit consent and don't persist.
- **Explainable, open formula** — Our scoring is transparent. Banks can audit and appeal.
- **Open-source stack** — We use Hugging Face models; no proprietary black box. Reproducible and extensible.
- **Indian context** — Amount extraction (₹, L, cr), DPDP alignment, RBI awareness. We're built for the Indian regulatory landscape.

---

### Q8.2: "Have you published any peer-reviewed validation of your risk score's predictive power?"

**Answer:**  
No. This is a prototype. For academic rigor, we would need:
- A controlled study with a bank partner
- Ground truth: default/non-default over 12–24 months
- ROC-AUC, Gini coefficient, and calibration curves
- Comparison with CIBIL and a random baseline

We present this as a **proof-of-concept** with a clear path to validation, not a validated product.

---

## 9. QUICK-FIRE ROUND

| Question | One-Liner Answer |
|----------|------------------|
| Why 5 categories? | Cover major financial behaviors; extensible (e.g., add "Gambling"). |
| Why confidence threshold 0.5? | Configurable; 0.5 balances precision/recall; banks can tune. |
| Why sentiment boost ±10–15%? | Heuristic; negative + Risk = more concerning; positive + Savings = reassuring. |
| Why amount factor capped at 3x? | Prevents one large risky amount from dominating; avoids outliers. |
| Why Streamlit and not React? | Rapid prototyping; Python-native; easy to demo. Production could be React + FastAPI. |
| Why no database? | Privacy by design; no PII to store; stateless scales horizontally. |

---

## 10. CLOSING DEFENSE

**"Summarize in 30 seconds: Why should we believe this is feasible?"**

> "We've built a working pipeline that takes social media text and produces an explainable risk score. We use proven NLP models, a transparent formula, and consent-first design. We don't claim it's production-ready — we claim it's a validated prototype that demonstrates feasibility. The gaps — language, gaming, scale — are documented and have clear mitigation paths. For 400 million credit-invisible Indians, any additional signal that's ethical and auditable is worth exploring."

---

*Use this document to prepare for viva voce, hackathon Q&A, or investor due diligence. Tailor answers to your audience (technical vs. business vs. regulatory).*
