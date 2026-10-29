"""
ScamShield AI - Web Application
Streamlit UI combining Machine Learning and transparent rule-based heuristics.
"""

from pathlib import Path
import re
from typing import Dict, List, Tuple
import joblib
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="ScamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "spam_pipeline.joblib"


@st.cache_resource
def load_pipeline():
    """Caches the trained pipeline in memory for sub-second user queries."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


# ---------------------------------------------------------
# Rule-Based Heuristics Engine
# ---------------------------------------------------------
RULES_CONFIG = [
    {
        "id": "credential_otp",
        "title": "Credential or Security Code Request",
        "regex": r"\b(otp|one[-\s]?time[-\s]?password|pin|password|verification\s*code|cvv|secret\s*code)\b",
        "explanation": "Legitimate institutions and technical support teams never request passwords, OTPs, or CVVs via chat or SMS.",
    },
    {
        "id": "banking_card",
        "title": "Bank or Payment Card Details Requested",
        "regex": r"\b(bank\s*account|credit\s*card|debit\s*card|account\s*number|wire\s*transfer|routing\s*number)\b",
        "explanation": "Direct requests to disclose card or bank credentials over insecure channels are common identity theft vectors.",
    },
    {
        "id": "upfront_fee",
        "title": "Advance / Upfront Fee Demand",
        "regex": r"\b(registration\s*fee|processing\s*fee|application\s*fee|security\s*deposit|advance\s*payment|pay\s*first|clearance\s*charge)\b",
        "explanation": "Authentic job offers and prize notifications never require candidates or winners to pay upfront fees to claim money or start work.",
    },
    {
        "id": "artificial_urgency",
        "title": "Artificial Urgency or Time Pressure",
        "regex": r"\b(urgent|immediately|act\s*now|within\s*\d+\s*(mins?|minutes?|hours?)|expires\s*today|final\s*notice|last\s*chance)\b",
        "explanation": "Scammers trigger panic using fake deadlines to prevent targets from verifying the claims independently.",
    },
    {
        "id": "account_coercion",
        "title": "Account Suspension or Threat of Action",
        "regex": r"\b(suspended|terminated|deactivated|blocked|legal\s*action|arrest\s*warrant|lawsuit|police|frozen)\b",
        "explanation": "Threats of account suspension or police involvement are standard coercive tactics used to demand compliance.",
    },
    {
        "id": "lottery_prize",
        "title": "Unsolicited Prize or Lottery Claim",
        "regex": r"\b(congratulations|you\s*won|lottery|lucky\s*draw|jackpot|claim\s*your\s*prize|free\s*gift|selected\s*winner)\b",
        "explanation": "Unexpected lottery or cash winnings for competitions you did not enter are almost universally scams.",
    },
    {
        "id": "suspicious_links",
        "title": "Embedded Hyperlink or Shortener Detected",
        "regex": r"(https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+|t\.co/\S+)",
        "explanation": "Links—especially shortened ones—are often used to disguise phishing portals or malware downloads.",
    },
]


def detect_warning_signs(text: str) -> List[Dict[str, str]]:
    """Evaluates input text against the rule configurations using regular expressions."""
    matched = []
    for rule in RULES_CONFIG:
        if re.search(rule["regex"], text, flags=re.IGNORECASE):
            matched.append(
                {
                    "title": rule["title"],
                    "explanation": rule["explanation"],
                }
            )
    return matched


# ---------------------------------------------------------
# Risk Assessment Engine
# ---------------------------------------------------------
def compute_risk(
    is_spam_model: bool, warning_count: int, length: int
) -> Tuple[str, str, str]:
    """
    Computes overall risk level by combining model prediction and rule counts.
    Returns: (Risk Label, UI Color Badge, Explanation)
    """
    if length < 10:
        return (
            "Caution (Insufficient Content)",
            "orange",
            "The message is too short for reliable statistical or rule-based analysis.",
        )

    if is_spam_model and warning_count >= 2:
        return (
            "High Risk",
            "red",
            "Both the ML model flagged spam patterns and multiple explicit scam red flags were matched.",
        )
    elif is_spam_model or warning_count >= 2:
        return (
            "Medium Risk",
            "orange",
            "Either the ML model flagged suspicious language OR multiple warning signs were triggered.",
        )
    elif warning_count == 1:
        return (
            "Low-to-Medium Risk",
            "orange",
            "The ML model did not flag spam patterns, but 1 specific warning sign was found. Review carefully.",
        )
    else:
        return (
            "Low Risk",
            "green",
            "No heuristic red flags were detected, and the message aligns with typical legitimate text.",
        )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ How It Works")
    st.markdown(
        """
    **Dual-Layer Screening Architecture:**
    1. **Machine Learning Pipeline:**
       - **TF-IDF Vectorizer:** Extracts term frequency and downweights generic English words.
       - **Multinomial Naive Bayes:** Probabilistic classifier trained on the UCI SMS Spam dataset.
    2. **Deterministic Rules Engine:**
       - Regular expressions flag direct requests for OTPs, upfront recruitment fees, urgency hooks, and links.
    """
    )

    st.markdown("---")
    st.header("🔒 Privacy & Safety")
    st.info(
        "Submitted messages are processed entirely in-memory for the current session. "
        "No messages are logged, stored, or sent to external databases. "
        "Please remove personal identifying information before testing."
    )


# ---------------------------------------------------------
# Main Interface
# ---------------------------------------------------------
st.title("🛡️ ScamShield AI")
st.subheader("Suspicious Message & Scam Screening System")
st.caption("An educational screening tool to help evaluate incoming communications.")

pipeline = load_pipeline()
if pipeline is None:
    st.error(
        "⚠️ Model artifact not found! Run `python train_model.py` in your terminal first to build `spam_pipeline.joblib`."
    )

# Sample Message Selector
st.markdown("##### 🧪 Test with preset samples:")
sample_options = {
    "Select a preset sample...": "",
    "Legitimate: Meeting reminder": "Hi team, let's meet in Conference Room B at 3:00 PM for sprint planning.",
    "Legitimate: Delivery update": "Your delivery package #4920 has arrived at the reception desk.",
    "Scam: Urgent Bank OTP / Block": "URGENT: Your bank account has been locked. Reply with your OTP within 15 minutes to prevent suspension.",
    "Scam: Upfront Fee Job Offer": "Congratulations! You are selected for our remote data entry role at $40/hr. Send $50 registration fee to begin.",
    "Scam: Prize Lottery Claim": "You won $10,000 in our lucky draw! Visit http://bit.ly/cash-prize now to claim your reward before it expires.",
}

chosen_sample = st.selectbox(
    "Choose a preset message or paste your own below:",
    list(sample_options.keys()),
)
default_text = sample_options[chosen_sample]

# User Input Text Area
user_input = st.text_area(
    "Paste message text below:",
    value=default_text,
    height=130,
    placeholder="Paste SMS, WhatsApp message, email snippet, or job offer here...",
)

col_analyze, col_reset = st.columns([1, 5])
with col_analyze:
    analyze_btn = st.button("🔍 Analyze Message", type="primary")
with col_reset:
    if st.button("🔄 Reset"):
        st.rerun()

# ---------------------------------------------------------
# Results Processing
# ---------------------------------------------------------
if analyze_btn:
    cleaned_input = user_input.strip()

    if not cleaned_input:
        st.warning("Please enter or paste a message to analyze.")
    elif pipeline is None:
        st.error("Cannot run analysis because `spam_pipeline.joblib` is missing.")
    else:
        # 1. ML inference
        pred_code = pipeline.predict([cleaned_input])[0]
        is_spam = bool(pred_code == 1)

        # 2. Rule-based evaluation
        warnings = detect_warning_signs(cleaned_input)

        # 3. Combined risk calculation
        risk_level, badge_color, reasoning = compute_risk(
            is_spam, len(warnings), len(cleaned_input)
        )

        st.markdown("---")
        st.subheader("Analysis Findings")

        # Visual indicator badges
        b1, b2, b3 = st.columns(3)
        with b1:
            if is_spam:
                st.error("🤖 ML Model: **Spam-Like Patterns**")
            else:
                st.success("🤖 ML Model: **Not Flagged as Spam**")

        with b2:
            st.metric(label="Warning Signs Found", value=f"{len(warnings)}")

        with b3:
            if badge_color == "red":
                st.error(f"⚠️ Risk Level: **{risk_level}**")
            elif badge_color == "orange":
                st.warning(f"⚠️ Risk Level: **{risk_level}**")
            else:
                st.success(f"✅ Risk Level: **{risk_level}**")

        # Explain the calculation
        with st.expander("ℹ️ How was this risk score determined?", expanded=True):
            st.write(f"**Rationale:** {reasoning}")
            st.caption(
                "Scoring Matrix: High = ML Flag + ≥2 rules | Medium = ML Flag OR ≥2 rules | Low-Medium = 1 rule | Low = Clean ML + 0 rules."
            )

        # Matched Warning Signs
        st.markdown("#### 🚩 Specific Warning Signs Detected")
        if warnings:
            for item in warnings:
                with st.chat_message("assistant"):
                    st.markdown(f"**{item['title']}**")
                    st.caption(item["explanation"])
        else:
            st.info("No explicit warning signs (such as OTP requests, fees, or urgency) were detected.")

        # Practical Next Steps
        st.markdown("#### 💡 Recommended Next Actions")
        if risk_level in ["High Risk", "Medium Risk", "Low-to-Medium Risk"]:
            st.markdown(
                """
            - **Never disclose OTPs, PINs, or passwords:** No verified company will ever ask you to read back a verification code.
            - **Do not pay upfront recruitment fees:** Legitimate companies never ask you to pay for training, onboarding kits, or equipment.
            - **Navigate independently:** Do not click links or call numbers provided in the message. Open your browser and search for the company's official domain.
            """
            )
        else:
            st.markdown(
                """
            - No immediate red flags were detected, but always confirm unexpected requests match previous conversations with the sender.
            - Never bypass security alerts in your browser or phone messaging application.
            """
            )

        # Disclaimer
        st.markdown("---")
        st.caption(
            "**Educational Disclaimer:** ScamShield AI is a pattern-screening demonstration. It does not provide definitive legal or security guarantees. Always verify sensitive matters through official channels."
        )