# 🛡️ ScamShield AI — Suspicious Message Detection System

ScamShield AI is an end-to-end NLP screening application that identifies spam patterns and high-risk social engineering markers (such as credential harvesting, advance-fee recruitment hooks, and fake urgency).

## 🌟 Key Features
- **Dual-Layer Architecture:** Integrates a Scikit-learn NLP pipeline (TF-IDF + Multinomial Naive Bayes) with a deterministic regex rules engine.
- **Transparent Indicator Breakdown:** Explains why specific wording is risky (e.g., OTP solicitations, upfront job fees).
- **Rule-Based Risk Matrix:** Computes Low, Medium, or High risk tiers based on combined model and rule heuristics.
- **Zero Data Persistence:** Messages are analyzed in-memory and are never stored or logged.

## 🛠️ Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas
- **Machine Learning:** Scikit-learn (TF-IDF Vectorization, Multinomial Naive Bayes, Pipelines)
- **Model Serialization:** Joblib
- **Web Interface:** Streamlit
- **Deployment:** Streamlit Community Cloud

## 📊 Dataset & Limitations
Trained on the **SMS Spam Collection** from the UCI Machine Learning Repository.
- *Limitation:* The dataset represents mobile SMS communication patterns and may exhibit lower sensitivity to long-form email spear-phishing or novel multi-stage social engineering scams. It serves as an educational screening tool, not a guarantee of security.

## 🚀 Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<jyothirmayi02>/ScamShield-AI.git
   cd ScamShield-AI