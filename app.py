import streamlit as st
import pandas as pd
import joblib
import config as config
import matplotlib.pyplot as plt
from src.interest_funcs import calculate_risk_premium, get_break_even_rate
from src.portfolio_funcs import expected_profit_list, portfolio_max_profit_and_threshold, plot_expected_profit
from src.individual_funcs import get_individual_prediction, get_shap_explanation, get_individual_expected_profit
from src.model import CreditFeatureEngineer

# ===============================================================================
# --- Cached Resources ---
# ===============================================================================

@st.cache_resource # run once and keep result in memory
def load_assets():
    # Pipeline file contains: Engineering -> Preprocessing -> XGBoost
    model = joblib.load(config.get_model_path())
    # Test data for portfolio-wide visuals
    test_data = pd.read_csv(config.get_data_path()).iloc[:200] 
    return model, test_data

model_pipeline, sample_df = load_assets()

# ===============================================================================
# --- Page Setup ---
# ===============================================================================

st.set_page_config(page_title="Credit Analytics Pro", layout="wide")
st.title("🏦 Credit Portfolio & Risk Analytics")

selection = st.sidebar.radio("Navigation", ["Portfolio Analysis", "Individual Assessment"])

# ===============================================================================
# --- Portfolio Window ---
# ===============================================================================

if selection == "Portfolio Analysis":
    st.header("Portfolio-Wide Optimization")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ead = st.number_input("Exposure at Default (EAD)", value=100000)
    with col_b:
        lgd = st.slider("Loss Given Default (LGD)", 0.0, 1.0, 0.45)
    with col_c:
        prime_rate = st.number_input("Current Prime Rate", value=0.1025, format="%.4f")

    # PREDICTION: Just pass the raw dataframe to the pipeline!
    pds = model_pipeline.predict_proba(sample_df)[:, 1]
    premiums = calculate_risk_premium(pds)
    total_rates = prime_rate + premiums

    profits, thresholds = expected_profit_list(pds, total_rates, ead, lgd)
    max_p, opt_t = portfolio_max_profit_and_threshold(profits, thresholds)

    # KPIs
    st.markdown("---")
    k1, k2 = st.columns(2)
    k1.metric("Max Portfolio Profit", f"R {max_p:,.2f}")
    k2.metric("Optimal PD Cutoff", f"{opt_t*100:.2f}%")

    fig = plot_expected_profit(thresholds, profits, max_p, opt_t)
    st.pyplot(fig)

# ===============================================================================
# --- Individual Assessment Window ---
# ===============================================================================

elif selection == "Individual Assessment":
    st.header("Client Risk Profiling")

    # UI Global Inputs
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ead = st.number_input("Exposure at Default (EAD)", value=100000)
    with col_b:
        lgd = st.slider("Loss Given Default (LGD)", 0.0, 1.0, 0.45)
    with col_c:
        prime_rate = st.number_input("Current Prime Rate", value=0.1025, format="%.4f")

    # UI Individual Inputs
    with st.expander("Client Demographics & Loan Details", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 100, 30)
            job = st.selectbox("Job", ['unskilled and non-resident', 'unskilled and resident', 'skilled', 'highly skilled'])
        with c2:
            housing = st.selectbox("Housing", ['own', 'rent', 'free'])
            savings = st.selectbox("Savings", ['little', 'moderate', 'quite rich', 'rich'])
            checking = st.selectbox("Checking", ['little', 'moderate', 'rich'])
        with c3:
            amount = st.number_input("Credit Amount", value=5000)
            duration = st.number_input("Duration (months)", value=24)
            purpose = st.selectbox("Purpose", ['car', 'furniture/equipment', 'radio/TV', 'business', 'education'])

    # PREDICTION BRIDGE

    input_data = pd.DataFrame({
        'Age': [age], 'Job': [job], 'Housing': [housing],
        'Saving accounts': [savings], 'Checking account': [checking],
        'Credit amount': [amount], 'Duration': [duration], 'Purpose': [purpose]
    })

    # Results UI
    st.markdown("---")

    # Financial Calculations
    individual_pd = get_individual_prediction(model_pipeline, input_data)
    premium = calculate_risk_premium(individual_pd)
    final_rate = prime_rate + premium
    breakeven_rate = get_break_even_rate(individual_pd, lgd)
    breakeven_premium = breakeven_rate - prime_rate
    individual_expected_profit = get_individual_expected_profit(individual_pd, final_rate, ead, lgd)

    # Layout
    st.subheader("🏦 Credit Decision Summary")
    res1, res2, res3, res4 = st.columns(4)

    # Column 1: Probability of Default
    res1.metric(label="Probability of Default", value=f"{individual_pd*100:.2f}%")

    # Column 2: Suggested Interest Rate (Target)
    res2.metric(label="Suggested Interest Rate", value=f"{final_rate*100:.2f}%")
    res2.markdown(
    f"""
    <div style="line-height: 1.2;">
        <small style="color: #6c757d;">
            Prime: {prime_rate*100:.2f}%<br>
            Premium: {premium*100:.2f}%
        </small>
    </div>
    """, unsafe_allow_html=True)

    # Column 3: Final Decision
    res3.metric(label="Breakeven Rate", value=f"{(breakeven_rate)*100:.2f}%")
    res3.markdown(
    f"""
    <div style="line-height: 1.2;">
        <small style="color: #6c757d;">
            Prime: {prime_rate*100:.2f}%<br>
            Premium: {breakeven_premium*100:.2f}%
        </small>
    </div>
    """, unsafe_allow_html=True)

    # Column 4: Expected Profit
    res4.metric(label="Expected Annual Profit", value=f"R{individual_expected_profit:,.2f}")

    st.divider()

    # Features Driving Risk
    st.subheader("Features Driving Risk")
    fig = get_shap_explanation(model_pipeline, input_data)
    st.pyplot(fig)
    st.caption("The graph is measured in log-odds (f(x)), the starting point is the default log-odds. " \
    "Movement to the right (red bars) indicate a greater chance of default, while movement to the left " \
    "(blue bars) indicate a lower chance of default. The greatest " \
    "effects on the client's probabiltiy of default (PD) are listed first.")
