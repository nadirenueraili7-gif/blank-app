import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="HELOC Decision Support System",
    page_icon="🏦",
    layout="wide"
)

# =========================
# Load Model & Artifacts
# =========================
@st.cache_resource
def load_artifacts():
    """Load model, medians, feature columns, and threshold."""
    artifacts = {}

    if os.path.exists("heloc_model.joblib"):
        artifacts["model"] = joblib.load("heloc_model.joblib")
    else:
        artifacts["model"] = None
        st.warning("⚠️ heloc_model.joblib not found. Using demo mode.")

    if os.path.exists("heloc_medians.joblib"):
        artifacts["medians"] = joblib.load("heloc_medians.joblib")
    else:
        artifacts["medians"] = None

    if os.path.exists("feature_cols.joblib"):
        artifacts["feature_cols"] = joblib.load("feature_cols.joblib")
    else:
        artifacts["feature_cols"] = None

    if os.path.exists("heloc_threshold.joblib"):
        artifacts["threshold"] = joblib.load("heloc_threshold.joblib")
    else:
        artifacts["threshold"] = 0.50  # fallback

    return artifacts

artifacts = load_artifacts()
model = artifacts["model"]
medians = artifacts["medians"]
feature_cols_order = artifacts["feature_cols"]
BEST_THRESHOLD = artifacts["threshold"]

# Feature column list (fallback if feature_cols.joblib missing)
FEATURE_COLUMNS = feature_cols_order or [
    'ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen',
    'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec',
    'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
    'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M',
    'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M',
    'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
    'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
    'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance'
]

# =========================
# Prediction Functions
# =========================
def preprocess_inputs(inputs: dict) -> pd.DataFrame:
    """
    Convert raw inputs dict to a DataFrame matching model expectations.
    Replace sentinel values (-9, -8, -7) with NaN, then fill with training medians.
    """
    X = pd.DataFrame([inputs])[FEATURE_COLUMNS]

    # Replace sentinel values with NaN (same as training pipeline)
    for col in X.columns:
        X[col] = X[col].replace([-7, -8, -9], np.nan)

    # Fill NaN with training-set medians (prevents data leakage)
    if medians is not None:
        X = X.fillna(medians)
    else:
        # Fallback: fill with 0 (demo mode)
        X = X.fillna(0)

    return X


def predict_probability(inputs: dict) -> tuple[float, str]:
    """
    Predict probability.
    The model predicts P(Bad) — class 1 = Bad.
    Decision: if P(Bad) >= threshold → Auto-Reject, else → Escalate for Manual Review.
    Returns (prob_good, decision) for display purposes.
    """
    if model is None:
        return _demo_predict(inputs)

    X = preprocess_inputs(inputs)
    prob_bad = float(model.predict_proba(X)[0, 1])   # class 1 = Bad
    prob_good = 1.0 - prob_bad

    # Use the optimal threshold from training
    if prob_bad >= BEST_THRESHOLD:
        decision = "Auto-Reject"
    else:
        decision = "Escalate for Manual Review"

    return prob_good, decision


def _demo_predict(inputs: dict) -> tuple[float, str]:
    """Demo scoring when model unavailable."""
    score = 0.0
    score += 0.008 * inputs.get("ExternalRiskEstimate", 70)
    score -= 0.003 * inputs.get("NetFractionRevolvingBurden", 25)
    score -= 0.02 * inputs.get("NumInqLast6M", 1)

    msd = inputs.get("MSinceMostRecentDelq", -7)
    if msd == -7 or msd >= 24:
        score += 0.15

    md12 = inputs.get("MaxDelq2PublicRecLast12M", 7)
    if md12 == 7:
        score += 0.25
    elif md12 in [3, 4]:
        score -= 0.10
    elif md12 in [0, 1, 2]:
        score -= 0.30

    prob_good = 1 / (1 + np.exp(-score))
    decision = "Escalate for Manual Review" if prob_good >= 0.50 else "Auto-Reject"
    return prob_good, decision


# =========================
# Explanation Functions
# =========================
def get_top_contributing_features(inputs: dict) -> list[dict]:
    """Generate local explanations based on input values."""
    explanations = []

    ere = inputs.get("ExternalRiskEstimate", 70)
    if ere >= 75:
        explanations.append({
            "feature": "Credit Risk Score",
            "value": f"{ere}/100",
            "impact": "Positive ↗",
            "explanation": "High ExternalRiskEstimate (indicating lower consolidated credit risk) positively influences advancement probability."
        })
    elif ere < 60:
        explanations.append({
            "feature": "Credit Risk Score",
            "value": f"{ere}/100",
            "impact": "Negative ↘",
            "explanation": "Low ExternalRiskEstimate indicates elevated credit risk."
        })

    md12 = inputs.get("MaxDelq2PublicRecLast12M", 7)
    if md12 == 7:
        explanations.append({
            "feature": "Recent Payment History",
            "value": "Current/Never Delinquent",
            "impact": "Positive ↗",
            "explanation": "No recent delinquencies strengthen your application."
        })
    elif md12 in [0, 1, 2, 3]:
        severity_map = {0: "Derogatory", 1: "120+ days", 2: "90 days", 3: "60 days"}
        explanations.append({
            "feature": "Recent Payment History",
            "value": severity_map.get(md12, "Unknown"),
            "impact": "Negative ↘",
            "explanation": f"Recent severe delinquency (MaxDelq severity = {md12}) reduces advancement probability."
        })

    util = inputs.get("NetFractionRevolvingBurden", 25)
    if util <= 30:
        explanations.append({
            "feature": "Credit Utilization",
            "value": f"{int(util)}%",
            "impact": "Positive ↗",
            "explanation": "Low revolving credit utilization strengthens the recommendation."
        })
    elif util >= 60:
        explanations.append({
            "feature": "Credit Utilization",
            "value": f"{int(util)}%",
            "impact": "Negative ↘",
            "explanation": "High credit utilization indicates greater financial strain."
        })

    inq = inputs.get("NumInqLast6M", 1)
    if inq >= 3:
        explanations.append({
            "feature": "Recent Credit Inquiries",
            "value": f"{inq} inquiries",
            "impact": "Negative ↘",
            "explanation": "Multiple recent credit inquiries may signal elevated credit-seeking behavior."
        })
    elif inq == 0:
        explanations.append({
            "feature": "Recent Credit Inquiries",
            "value": "None",
            "impact": "Positive ↗",
            "explanation": "No recent credit inquiries indicate stable credit behavior."
        })

    msd = inputs.get("MSinceMostRecentDelq", -7)
    if msd == -7:
        explanations.append({
            "feature": "Delinquency History",
            "value": "Never delinquent",
            "impact": "Positive ↗",
            "explanation": "No record of delinquency is a strong positive factor."
        })
    elif msd >= 36:
        explanations.append({
            "feature": "Time Since Last Issue",
            "value": f"{msd} months ago",
            "impact": "Positive ↗",
            "explanation": "Sufficient time has passed since the most recent delinquency."
        })
    elif 0 <= msd < 12:
        explanations.append({
            "feature": "Time Since Last Issue",
            "value": f"{msd} months ago",
            "impact": "Negative ↘",
            "explanation": "Recent delinquency reduces advancement probability."
        })

    return explanations[:5]


def get_improvement_suggestions(inputs: dict) -> list[str]:
    """Generate actionable feedback."""
    suggestions = []

    if inputs.get("NetFractionRevolvingBurden", 25) > 35:
        suggestions.append("**Reduce credit card balances** to lower your utilization ratio (target: below 30%).")

    if inputs.get("NumInqLast6M", 1) >= 2:
        suggestions.append("**Avoid new credit applications** for 6-12 months to demonstrate stability.")

    if inputs.get("MaxDelq2PublicRecLast12M", 7) in [0, 1, 2, 3, 4]:
        suggestions.append("**Maintain consistent on-time payments** to improve your payment history.")

    if inputs.get("MSinceOldestTradeOpen", 180) < 24:
        suggestions.append("**Build credit history length** through responsible account management over time.")

    if inputs.get("ExternalRiskEstimate", 70) < 65:
        suggestions.append("**Improve overall credit standing** through on-time payments and responsible credit use.")

    if len(suggestions) == 0:
        suggestions.append("**Continue maintaining your current positive financial profile.**")

    return suggestions[:4]


# =========================
# Default Values (typical good applicant)
# =========================
def get_default_inputs():
    """Return typical values for a decent applicant (raw values, not preprocessed)."""
    return {
        "ExternalRiskEstimate": 70,
        "MSinceOldestTradeOpen": 180,
        "MSinceMostRecentTradeOpen": 5,
        "AverageMInFile": 75,
        "NumSatisfactoryTrades": 18,
        "NumTrades60Ever2DerogPubRec": 0,
        "NumTrades90Ever2DerogPubRec": 0,
        "PercentTradesNeverDelq": 95,
        "MSinceMostRecentDelq": -7,   # -7 = never delinquent (sentinel)
        "MaxDelq2PublicRecLast12M": 7,  # 7 = Current
        "MaxDelqEver": 8,              # 8 = Current
        "NumTotalTrades": 15,
        "NumTradesOpeninLast12M": 1,
        "PercentInstallTrades": 40,
        "MSinceMostRecentInqexcl7days": 3,
        "NumInqLast6M": 1,
        "NumInqLast6Mexcl7days": 1,
        "NetFractionRevolvingBurden": 25,   # Integer scale (0-100+), NOT 0-1
        "NetFractionInstallBurden": 15,     # Integer scale
        "NumRevolvingTradesWBalance": 3,
        "NumInstallTradesWBalance": 2,
        "NumBank2NatlTradesWHighUtilization": 1,
        "PercentTradesWBalance": 60,
    }


# =========================
# UI: Mode Selection
# =========================
st.title("🏦 HELOC Decision Support System")
st.markdown("### Machine Learning-Based Application Screening")

mode = st.sidebar.radio(
    "**Select Mode**",
    options=["👤 Applicant View (Simplified)", "🔧 Internal Testing (Full Controls)"],
    index=0
)

st.sidebar.divider()
st.sidebar.markdown("### About This Tool")
st.sidebar.info(
    "**Applicant View**: Simulates the customer-facing interface with automatic credit bureau data retrieval.\n\n"
    "**Internal Testing**: Allows loan officers to manually adjust all 23 features to test model behavior."
)
if model is not None:
    st.sidebar.success(f"✅ Model loaded")
    st.sidebar.caption(f"Decision threshold: {BEST_THRESHOLD:.3f}")
else:
    st.sidebar.warning("⚠️ Demo mode (no model file)")


# =========================
# MODE 1: Applicant View (Simplified)
# =========================
if mode == "👤 Applicant View (Simplified)":
    st.info(
        "**Welcome!** This tool provides an initial screening for your HELOC application. "
        "In a real system, we would automatically retrieve your credit information with your permission. "
        "For this demo, please provide a few key details below."
    )

    st.divider()
    st.header("📋 Application Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Personal Details")
        applicant_name = st.text_input("Full Name", placeholder="John Doe")
        applicant_ssn = st.text_input("Last 4 digits of SSN", placeholder="1234", max_chars=4)

        st.markdown("### Authorization")
        consent = st.checkbox(
            "I authorize Simon Bank to retrieve my credit bureau information",
            value=False
        )

    with col2:
        st.markdown("### Quick Questions")
        st.caption("*These help us understand your credit profile*")

        credit_self_assessment = st.select_slider(
            "1. How would you rate your overall credit?",
            options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
            value="Good",
            help="Your general assessment of your creditworthiness"
        )

        credit_history_years = st.selectbox(
            "2. How long have you had credit accounts?",
            options=["Less than 2 years", "2-5 years", "5-10 years", "10-15 years", "Over 15 years"],
            index=2,
            help="Approximately when did you open your first credit card, loan, or credit account?"
        )

        recent_late_payments = st.radio(
            "3. Any late payments in the past 12 months?",
            options=["No, always on time", "Yes, 1-2 times", "Yes, 3 or more times"],
            index=0,
            help="This includes credit cards, loans, utilities, etc."
        )

        credit_card_balance = st.selectbox(
            "4. How do you typically manage credit card balances?",
            options=[
                "Pay in full every month",
                "Carry balance under 30% of limit",
                "Carry balance 30-60% of limit",
                "Carry balance over 60% of limit"
            ],
            index=1,
            help="On average, what percentage of your available credit are you using?"
        )

        recent_credit_apps = st.radio(
            "5. New credit applications in the past 6 months?",
            options=["None", "1-2 applications", "3 or more applications"],
            index=0,
            help="This includes credit cards, loans, auto financing, etc."
        )

        total_credit_accounts = st.selectbox(
            "6. Approximately how many credit accounts do you have?",
            options=[
                "1-3 accounts",
                "4-7 accounts",
                "8-15 accounts",
                "16-25 accounts",
                "More than 25 accounts"
            ],
            index=2,
            help="Include all credit cards, loans, mortgages, auto loans, etc."
        )

        worst_ever_delinquency = st.radio(
            "7. Have you ever had serious payment issues?",
            options=[
                "Never had issues",
                "30-60 days late (several years ago)",
                "90+ days late or collections",
                "Bankruptcy or charge-off"
            ],
            index=0,
            help="Your worst payment issue in your entire credit history"
        )

    st.divider()

    if not consent:
        st.warning("⚠️ Please authorize credit bureau access to proceed with screening.")
        st.stop()

    # =========================
    # Simulate Credit Bureau API Call
    # =========================
    with st.spinner("🔄 Retrieving credit bureau data..."):
        import time
        time.sleep(1.5)

    st.success("✅ Credit information retrieved successfully")

    # Map simple inputs to actual feature values (integer scale, matching dataset)
    credit_score_map = {
        "Poor": 45, "Fair": 60, "Good": 70, "Very Good": 80, "Excellent": 90
    }
    credit_history_map = {
        "Less than 2 years": 18, "2-5 years": 42, "5-10 years": 90,
        "10-15 years": 156, "Over 15 years": 240
    }
    delinquency_12m_map = {
        "No, always on time": 7, "Yes, 1-2 times": 4, "Yes, 3 or more times": 2
    }
    utilization_map = {
        "Pay in full every month": 10,
        "Carry balance under 30% of limit": 25,
        "Carry balance 30-60% of limit": 45,
        "Carry balance over 60% of limit": 75
    }
    inquiries_map = {
        "None": 0, "1-2 applications": 1, "3 or more applications": 4
    }
    accounts_map = {
        "1-3 accounts": 3, "4-7 accounts": 6, "8-15 accounts": 12,
        "16-25 accounts": 20, "More than 25 accounts": 30
    }
    worst_ever_map = {
        "Never had issues": 8,
        "30-60 days late (several years ago)": 5,
        "90+ days late or collections": 3,
        "Bankruptcy or charge-off": 2
    }

    # Build inputs using defaults + user adjustments
    inputs = get_default_inputs()

    inputs["ExternalRiskEstimate"] = credit_score_map[credit_self_assessment]

    inputs["MSinceOldestTradeOpen"] = credit_history_map[credit_history_years]
    inputs["AverageMInFile"] = int(credit_history_map[credit_history_years] * 0.6)

    inputs["MaxDelq2PublicRecLast12M"] = delinquency_12m_map[recent_late_payments]
    if recent_late_payments == "No, always on time":
        inputs["MSinceMostRecentDelq"] = -7
        inputs["PercentTradesNeverDelq"] = 100
    elif recent_late_payments == "Yes, 1-2 times":
        inputs["MSinceMostRecentDelq"] = 8
        inputs["PercentTradesNeverDelq"] = 90
    else:
        inputs["MSinceMostRecentDelq"] = 3
        inputs["PercentTradesNeverDelq"] = 75

    inputs["NetFractionRevolvingBurden"] = utilization_map[credit_card_balance]

    inputs["NumInqLast6M"] = inquiries_map[recent_credit_apps]
    inputs["NumInqLast6Mexcl7days"] = inputs["NumInqLast6M"]
    if inquiries_map[recent_credit_apps] > 0:
        inputs["MSinceMostRecentInqexcl7days"] = 2
        inputs["NumTradesOpeninLast12M"] = min(inquiries_map[recent_credit_apps], 3)
    else:
        inputs["MSinceMostRecentInqexcl7days"] = 12
        inputs["NumTradesOpeninLast12M"] = 0

    inputs["NumTotalTrades"] = accounts_map[total_credit_accounts]
    inputs["NumSatisfactoryTrades"] = int(inputs["NumTotalTrades"] * 0.85)
    inputs["NumRevolvingTradesWBalance"] = min(int(inputs["NumTotalTrades"] * 0.3), 8)

    inputs["MaxDelqEver"] = worst_ever_map[worst_ever_delinquency]
    if worst_ever_delinquency in ["90+ days late or collections", "Bankruptcy or charge-off"]:
        inputs["NumTrades60Ever2DerogPubRec"] = 1
        inputs["NumTrades90Ever2DerogPubRec"] = 1
    else:
        inputs["NumTrades60Ever2DerogPubRec"] = 0
        inputs["NumTrades90Ever2DerogPubRec"] = 0

    # Show retrieved data summary
    with st.expander("📊 View Retrieved Credit Information", expanded=False):
        st.markdown("**Summary of information from credit bureaus:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Credit Score", inputs["ExternalRiskEstimate"])
            st.metric("Credit History", f"{inputs['MSinceOldestTradeOpen'] // 12} years")
        with col_b:
            st.metric("Credit Utilization", f"{inputs['NetFractionRevolvingBurden']}%")
            st.metric("Total Accounts", inputs["NumTotalTrades"])
        with col_c:
            st.metric("Recent Inquiries (6M)", inputs["NumInqLast6M"])
            delinq_status = "Current" if inputs["MaxDelq2PublicRecLast12M"] == 7 else "Past Issues"
            st.metric("Payment Status", delinq_status)

    st.divider()

    # Prediction
    st.header("🎯 Screening Results")
    prob_good, decision = predict_probability(inputs)

    if decision == "Escalate for Manual Review":
        st.success(f"### ✅ **{decision}**")
        st.markdown(
            "Great news! Your application has **passed the initial screening** and will be "
            "reviewed by one of our loan officers. They will contact you within 2-3 business days."
        )
        st.balloons()
    else:
        st.error(f"### ❌ **{decision}**")
        st.markdown(
            "Unfortunately, your application did not pass our initial screening at this time. "
            "However, you can take steps to improve your chances and reapply in the future."
        )

    st.progress(prob_good, text=f"Screening Confidence: {prob_good:.0%}")

    st.divider()

    # Explanations
    st.header("📊 Why This Result?")
    contributing_features = get_top_contributing_features(inputs)

    for i, feat in enumerate(contributing_features, 1):
        col_x, col_y = st.columns([4, 1])
        with col_x:
            st.markdown(f"**{i}. {feat['feature']}**: {feat['value']}")
            st.caption(feat['explanation'])
        with col_y:
            if "Positive" in feat['impact']:
                st.success(feat['impact'])
            else:
                st.warning(feat['impact'])

    st.divider()

    # Recommendations
    st.header("💡 Next Steps")
    suggestions = get_improvement_suggestions(inputs)

    if decision == "Auto-Reject":
        st.markdown("**To improve your chances for future applications:**")
    else:
        st.markdown("**While you wait for loan officer review:**")

    for i, suggestion in enumerate(suggestions, 1):
        st.markdown(f"{i}. {suggestion}")

    if decision == "Auto-Reject":
        st.info("💬 **Questions?** Contact our support team at (585) 555-BANK or visit any branch.")


# =========================
# MODE 2: Internal Testing (Full Controls)
# =========================
else:
    st.warning("🔧 **Internal Testing Mode** - For loan officers and model validation only")

    st.markdown(
        "This mode allows you to manually adjust all 23 credit bureau features to test model behavior. "
        "Use preset scenarios or customize values to understand decision boundaries."
    )

    st.divider()

    # Preset scenarios
    st.header("🎲 Quick Scenarios")
    scenario = st.selectbox(
        "Load a preset scenario",
        options=[
            "Default (Typical Good Applicant)",
            "Excellent Applicant",
            "Borderline Applicant",
            "High-Risk Applicant",
            "Custom (Manual)"
        ],
        index=0
    )

    if scenario == "Excellent Applicant":
        inputs = get_default_inputs()
        inputs["ExternalRiskEstimate"] = 90
        inputs["PercentTradesNeverDelq"] = 100
        inputs["NetFractionRevolvingBurden"] = 10
        inputs["NumInqLast6M"] = 0
        inputs["MSinceOldestTradeOpen"] = 300
    elif scenario == "Borderline Applicant":
        inputs = get_default_inputs()
        inputs["ExternalRiskEstimate"] = 65
        inputs["MaxDelq2PublicRecLast12M"] = 4
        inputs["NetFractionRevolvingBurden"] = 45
        inputs["NumInqLast6M"] = 2
    elif scenario == "High-Risk Applicant":
        inputs = get_default_inputs()
        inputs["ExternalRiskEstimate"] = 45
        inputs["MaxDelq2PublicRecLast12M"] = 1
        inputs["MaxDelqEver"] = 3
        inputs["NetFractionRevolvingBurden"] = 80
        inputs["NumInqLast6M"] = 5
        inputs["MSinceMostRecentDelq"] = 3
        inputs["PercentTradesNeverDelq"] = 65
    else:
        inputs = get_default_inputs()

    st.divider()

    # Full feature controls
    st.header("🎛️ Feature Controls")

    tab1, tab2, tab3, tab4 = st.tabs(["Credit Profile", "Payment History", "Utilization & Activity", "Advanced"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            inputs["ExternalRiskEstimate"] = st.slider("Credit Risk Score", 0, 100, inputs["ExternalRiskEstimate"], 1)
            inputs["MSinceOldestTradeOpen"] = st.slider("Credit History (months)", 0, 500, inputs["MSinceOldestTradeOpen"], 6)
            inputs["NumTotalTrades"] = st.slider("Total Credit Accounts", 0, 50, inputs["NumTotalTrades"], 1)
        with col2:
            inputs["NumSatisfactoryTrades"] = st.slider("Satisfactory Trades", 0, 50, inputs["NumSatisfactoryTrades"], 1)
            inputs["PercentTradesNeverDelq"] = st.slider("% Trades Never Delinquent", 0, 100, inputs["PercentTradesNeverDelq"], 5)
            inputs["AverageMInFile"] = st.slider("Average Months in File", 0, 300, inputs["AverageMInFile"], 5)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            inputs["MaxDelq2PublicRecLast12M"] = st.selectbox(
                "Worst Delinquency (12M)",
                options=[(7, "Current"), (4, "30d"), (3, "60d"), (2, "90d"), (1, "120+d"), (0, "Derog")],
                format_func=lambda x: x[1],
                index=next(i for i, (v, _) in enumerate([(7, ""), (4, ""), (3, ""), (2, ""), (1, ""), (0, "")]) if v == inputs["MaxDelq2PublicRecLast12M"])
            )[0]

            inputs["MaxDelqEver"] = st.selectbox(
                "Worst Delinquency (Ever)",
                options=[(8, "Current"), (6, "30d"), (5, "60d"), (4, "90d"), (3, "120+d"), (2, "Derog")],
                format_func=lambda x: x[1],
                index=next(i for i, (v, _) in enumerate([(8, ""), (6, ""), (5, ""), (4, ""), (3, ""), (2, "")]) if v == inputs["MaxDelqEver"])
            )[0]

        with col2:
            inputs["MSinceMostRecentDelq"] = st.slider("Months Since Last Delinq", -7, 120, inputs["MSinceMostRecentDelq"], 1,
                help="-7 = never delinquent")
            inputs["NumTrades60Ever2DerogPubRec"] = st.slider("Trades 60+ Days Delinq", 0, 10, inputs["NumTrades60Ever2DerogPubRec"], 1)
            inputs["NumTrades90Ever2DerogPubRec"] = st.slider("Trades 90+ Days Delinq", 0, 10, inputs["NumTrades90Ever2DerogPubRec"], 1)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            inputs["NetFractionRevolvingBurden"] = st.slider("Revolving Utilization", 0, 200, int(inputs["NetFractionRevolvingBurden"]), 5)
            inputs["NumRevolvingTradesWBalance"] = st.slider("Revolving Accounts w/ Balance", 0, 20, inputs["NumRevolvingTradesWBalance"], 1)
            inputs["NumBank2NatlTradesWHighUtilization"] = st.slider("High Utilization Accounts", 0, 20, inputs["NumBank2NatlTradesWHighUtilization"], 1)

        with col2:
            inputs["NumInqLast6M"] = st.slider("Inquiries (6M)", 0, 10, inputs["NumInqLast6M"], 1)
            inputs["NumTradesOpeninLast12M"] = st.slider("New Accounts (12M)", 0, 10, inputs["NumTradesOpeninLast12M"], 1)
            inputs["PercentTradesWBalance"] = st.slider("% Trades w/ Balance", 0, 100, inputs["PercentTradesWBalance"], 5)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            inputs["MSinceMostRecentTradeOpen"] = st.slider("Months Since Recent Trade", 0, 100, inputs["MSinceMostRecentTradeOpen"], 1)
            inputs["PercentInstallTrades"] = st.slider("% Installment Trades", 0, 100, inputs["PercentInstallTrades"], 5)
            inputs["MSinceMostRecentInqexcl7days"] = st.slider("Months Since Recent Inq", 0, 100, inputs["MSinceMostRecentInqexcl7days"], 1)
            inputs["NumInqLast6Mexcl7days"] = st.slider("Inq Last 6M (excl 7d)", 0, 10, inputs["NumInqLast6Mexcl7days"], 1)
        with col2:
            inputs["NetFractionInstallBurden"] = st.slider("Installment Burden", 0, 200, int(inputs["NetFractionInstallBurden"]), 5)
            inputs["NumInstallTradesWBalance"] = st.slider("Installment Accounts w/ Balance", 0, 20, inputs["NumInstallTradesWBalance"], 1)

    st.divider()

    # Prediction
    st.header("🎯 Model Prediction")
    prob_good, decision = predict_probability(inputs)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Decision", decision)
    with col_b:
        st.metric("Prob(Good)", f"{prob_good:.1%}")
    with col_c:
        st.metric("Prob(Bad)", f"{1 - prob_good:.1%}")

    st.progress(prob_good)

    if decision == "Escalate for Manual Review":
        st.success("✅ Application passes screening")
    else:
        st.error("❌ Application auto-rejected")

    st.divider()

    # Explanations
    st.header("📊 Feature Contributions")
    contributing_features = get_top_contributing_features(inputs)

    for feat in contributing_features:
        col_x, col_y = st.columns([4, 1])
        with col_x:
            st.markdown(f"**{feat['feature']}**: {feat['value']}")
            st.caption(feat['explanation'])
        with col_y:
            if "Positive" in feat['impact']:
                st.success(feat['impact'])
            else:
                st.warning(feat['impact'])

    # Technical details
    with st.expander("🔍 Technical Details", expanded=False):
        st.markdown("### Model Info")
        if model:
            st.success(f"✅ Model: {type(model).__name__}")
            st.caption(f"Decision threshold (P(Bad) ≥ {BEST_THRESHOLD:.3f} → Reject)")
        else:
            st.warning("⚠️ Demo mode — no model loaded")

        st.markdown("### All Feature Values (raw, before preprocessing)")
        st.dataframe(pd.DataFrame([inputs]).T.rename(columns={0: "Value"}), height=600)


st.divider()
st.caption(
    "**Disclaimer**: This prototype demonstrates ML-based screening for educational purposes. "
    "Final lending decisions require human review and regulatory compliance. "
    "© 2026 Simon Bank of Rochester® | CIS432 Team Project"
)
