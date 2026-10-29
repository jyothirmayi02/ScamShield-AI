"""
ScamShield AI - Web Application
Streamlit UI combining Machine Learning and transparent rule-based heuristics.
"""

from pathlib import Path
import re
from typing import Dict, List, Tuple
import joblib
import streamlit as st

st.set_page_config(
    page_title="ScamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "spam_pipeline.joblib"


@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


# ---------------------------------------------------------
# Enhanced Rule-Based Heuristics Engine
# ---------------------------------------------------------
RULES_CONFIG = [
    {
        "id": "credential_otp",
        "title": "Credential or Security Code Request",
        "regex": r"\b(otp|one[-\s]?time[-\s]?password|pin|password|verification\s*code|cvv|secret\s*code)\b",
        "explanation": "Legitimate institutions and support teams never ask you to reveal passwords, OTPs, or CVVs via message.",
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
        "regex": r"\b(urgent|urgently|immediately|act\s*now|within\s*\d+\s*(mins?|minutes?|hours?)|tonight|expires\s*today|final\s*notice|last\s*chance)\b",
        "explanation": "Scammers trigger panic using fake deadlines to prevent targets from verifying claims independently.",
    },
    {
        "id": "service_coercion",
        "title": "Account/Service Disconnection or Legal Coercion",
        "regex": r"\b(suspended|terminated|deactivated|blocked|disconnected|disconnection|power\s*cut|legal\s*action|arrest\s*warrant|lawsuit|police|frozen|freeze)\b",
        "explanation": "Threats of immediate power disconnection, account freezing, or police action are standard coercive scam tactics.",
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
        "explanation": "Shortened or non-official links are frequently used to route victims to phishing portals.",
    },
]


def detect_warning_signs(text: str) -> List[Dict[str, str]]:
    matched = []
    for rule in RULES_CONFIG:
        if re.search(rule["regex"], text, flags=re.IGNORECASE):
            matched.append(
                {"title": rule["title"], "explanation": rule["explanation"]}
            )
    return matched


# ---------------------------------------------------------
# Robust Risk Assessment Matrix
# ---------------------------------------------------------
def compute_risk(
    is_spam_model: bool, warning_count: int, length: int, warnings: list
) -> Tuple[str, str, str]:
    if length < 10:
        return (
            "Caution (Insufficient Content)",
            "orange",
            "The message is too short for reliable statistical or rule-based analysis.",
        )

    # Detect high-severity triggers (links + urgency/disconnection/credential theft)
    has_link = any(
        w["title"] == "Embedded Hyperlink or Shortener Detected"
        for w in warnings
    )
    has_coercion = any(
        w["title"] == "Account/Service Disconnection or Legal Coercion"
        for w in warnings
    )
    has_credential = any(
        w["title"] == "Credential or Security Code Request" for w in warnings
    )
    has_fee = any(
        w["title"] == "Advance / Upfront Fee Demand" for w in warnings
    )

    # 1. High Risk conditions:
    # - Model flagged + at least 1 rule
    # - OR 2+ warning rules matched (even if older ML missed it)
    # - OR dangerous combinations (e.g. Disconnection threat + Link, or OTP + Urgency)
    if (
        (is_spam_model and warning_count >= 1)
        or warning_count >= 2
        or (has_link and has_coercion)
        or (has_credential)
        or (has_fee)
    ):
        return (
            "High Risk",
            "red",
            "Critical warning signs were detected (such as direct credential requests, upfront fees, or coercive threats paired with links).",
        )

    # 2. Medium Risk: Model flagged alone, OR exactly 1 non-critical rule matched
    elif is_spam_model or warning_count == 1:
        return (
            "Medium Risk",
            "orange",
            "The message triggered pattern flags. Verify independently before acting.",
        )

    # 3. Clean
    else:
        return (
            "Low Risk",
            "green",
            "No heuristic red flags were detected, and the message contains no high-risk solicitation indicators.",
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
       - TF-IDF vectorization + Multinomial Naive Bayes trained on the SMS Spam Collection.
    2. **Deterministic Rules Engine:**
       - Scans for advance fees, urgency, disconnection threats, credentials, and suspicious links.
    """
    )
    st.markdown("---")
    st.header("🔒 Privacy & Safety")
    st.info(
        "Submitted messages are processed entirely in-memory for the current request. "
        "No personal text is persisted or logged."
    )


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("🛡️ ScamShield AI")
st.subheader("Suspicious Message & Scam Screening System")
st.caption("An educational screening tool to help evaluate incoming communications.")

pipeline = load_pipeline()
if pipeline is None:
    st.error(
        "⚠️ Model artifact not found! Run `python train_model.py` in your terminal to build `spam_pipeline.joblib`."
    )

# Expanded presets with Category A, B, and C
sample_options = {
    "Select an example...": "",
    "[Category A] WhatsApp Freelance Scam": "Dear candidate, your resume has been shortlisted for a freelance video review role. Earn $300-$500 daily by liking YouTube videos. To activate your employee portal, kindly deposit a refundable security deposit of $35. Act now as seats are limited.",
    "[Category A] Electricity Disconnection Phishing": "URGENT NOTICE: Your electricity connection power will be disconnected tonight at 9:30 PM due to an unpaid pending bill from last month. Immediately contact our officer at http://bit.ly/power-pay to clear charges and avoid account freeze.",
    "[Category A] Tax / Police Coercion Scam": "Federal Tax Authority: A legal action notice and arrest warrant has been filed under your name for tax discrepancy. Please call immediately and share your verification code to speak with an agent.",
    "[Category B] Normal Personal Message": "Hey Mom, I lost my bus pass so I might be running about 20 minutes late for dinner. Don't wait up for me, go ahead and eat!",
    "[Category B] Normal Workplace Update": "Hi Team, please find attached the slide deck for our quarterly project review scheduled for Thursday at 10 AM. Let me know if you have any questions before then.",
    "[Category C] Legitimate Security Alert (No OTP Asked)": "Google Security Alert: A new sign-in was detected on your account from a Windows device. If this was you, you can safely ignore this alert.",
}

chosen_sample = st.selectbox(
    "Choose a preset message or paste your own below:",
    list(sample_options.keys()),
)
default_text = sample_options[chosen_sample]

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
# Processing & Display
# ---------------------------------------------------------
if analyze_btn:
    cleaned_input = user_input.strip()

    if not cleaned_input:
        st.warning("Please enter or paste a message to analyze.")
    elif pipeline is None:
        st.error("Cannot run analysis because `spam_pipeline.joblib` is missing.")
    else:
        pred_code = pipeline.predict([cleaned_input])[0]
        is_spam = bool(pred_code == 1)

        warnings = detect_warning_signs(cleaned_input)
        risk_level, badge_color, reasoning = compute_risk(
            is_spam, len(warnings), len(cleaned_input), warnings
        )

        st.markdown("---")
        st.subheader("Analysis Findings")

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

        with st.expander("ℹ️ How was this risk score determined?", expanded=True):
            st.write(f"**Rationale:** {reasoning}")

        st.markdown("#### 🚩 Specific Warning Signs Detected")
        if warnings:
            for item in warnings:
                with st.chat_message("assistant"):
                    st.markdown(f"**{item['title']}**")
                    st.caption(item["explanation"])
        else:
            st.info("No explicit warning signs were detected.")

        st.markdown("#### 💡 Recommended Next Actions")
        if risk_level in ["High Risk", "Medium Risk"]:
            st.markdown(
                """
            - **Never disclose OTPs, PINs, or passwords:** No verified company asks you to read back verification codes.
            - **Do not pay upfront fees:** Real employers and contest organizers do not charge upfront deposits.
            - **Verify independently:** Do not click links or call numbers sent in the message. Check the organization's official website directly.
            """
            )
        else:
            st.markdown(
                """
            - No immediate red flags were detected. Always confirm unexpected messages through known contact channels.
            """
            )

        st.markdown("---")
        st.caption(
            "**Educational Disclaimer:** ScamShield AI is a pattern-screening demonstration. It does not provide definitive legal or security guarantees."
        )