import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import time

# Internal Modules
import theme
import simulator
import fair_price_net
import trend_net
import advisor

# --- 1. System Initialization & State ---
theme.set_style()

# Initialize Session State for "Memory" (The Ledger)
if 'ledger' not in st.session_state:
    st.session_state.ledger = []

@st.cache_resource(show_spinner="Booting Market Simulation & AI Models...")
def initialize_system():
    """
    Cold-start initialization.
    Generates the market world and trains the brains (Neural Nets) ONCE.
    """
    # 1. Create the World
    df = simulator.get_simulation_data()
    
    # 2. Train the Brains
    fair_price_net.train_and_validate()
    trend_net.train_trend_model()
    
    return df

market_data = initialize_system()

# --- 2. Sidebar: Context & News ---
with st.sidebar:
    st.header("Market Intelligence")
    st.markdown("Live updates from the simulation engine.")
    
    # Dynamic "News" based on the simulation data
    # We look at the latest factor changes to generate "Headlines"
    products = market_data['good'].unique()
    for p in products:
        last_row = market_data[market_data['good'] == p].iloc[-1]
        prev_row = market_data[market_data['good'] == p].iloc[-2]
        
        # Determine movement
        price_delta = last_row['market_price'] - prev_row['market_price']
        icon = "🟢" if price_delta > 0 else "🔴"
        
        st.markdown(f"**{p}**")
        st.caption(f"{icon} Trading at ${last_row['market_price']:.2f}")
        
        # Simulated "Reason"
        if last_row['supply_factor'] < 0.95:
            st.info("⚠️ Supply Shortage detected.")
        elif last_row['demand_factor'] > 1.05:
            st.success("🔥 High Demand period.")
        
        st.divider()

# --- 3. Main Interface ---
st.title("PriceWise ⚖️")
st.markdown("### Decision Engine for Small Sellers")
st.caption("AI-Powered Equilibrium & Trend Analysis")

# --- 4. The Workspace ---
with st.container():
    c1, c2, c3 = st.columns([2, 1, 1])
    
    with c1:
        product_selector = st.selectbox(
            "Item to Sell", 
            options=['Organic Wheat (Crop)', 'Handmade Table (Product)', 'Studio Apt (Rent)']
        )
        # Get defaults
        current_market = market_data[market_data['good'] == product_selector].iloc[-1]

    with c2:
        user_cost = st.number_input("Unit Cost ($)", value=float(current_market['cost_factor']), step=10.0)

    with c3:
        user_price = st.number_input("Asking Price ($)", value=float(current_market['market_price']), step=10.0)

# --- 5. Execution ---
if st.button("Run Analysis", type="primary"):
    
    with st.spinner("Consulting Neural Networks..."):
        time.sleep(0.5) # UX Pacing
        
        # A. Fair Price Net
        fair_price_est = fair_price_net.predict_fair_price(
            cost=user_cost,
            demand=current_market['demand_factor'],
            supply=current_market['supply_factor'],
            seasonality=current_market['seasonality']
        )
        
        # B. Trend Net
        history = market_data[market_data['good'] == product_selector].tail(6)
        trend_input = history.tail(3)
        trend_pred = trend_net.predict_trend(trend_input)
        
        # C. Business Logic
        verdict = advisor.compare_to_fair(user_price, fair_price_est)
        advice = advisor.sell_or_wait(trend_pred)
        
        # --- Save to Ledger ---
        st.session_state.ledger.insert(0, {
            "Time": time.strftime("%H:%M"),
            "Product": product_selector,
            "My Price": f"${user_price}",
            "Fair Price": f"${fair_price_est:.2f}",
            "Verdict": verdict['status'],
            "Action": advice['action']
        })

    # --- 6. The Dashboard (Results) ---
    
    # Verdict Banner
    color_map = {"Fair": "verdict-fair", "Overpriced": "verdict-over", "Underpriced": "verdict-under", "Perfectly Aligned": "verdict-fair", "Premium Pricing": "verdict-over", "Value Pricing": "verdict-under"}
    css = color_map.get(verdict['status'], "verdict-fair")
    
    st.markdown(f"""
        <div class="verdict-box {css}">
            <h3 style="margin:0">Verdict: {verdict['status']}</h3>
            <p style="margin:5px 0 0 0">{verdict['reason']}</p>
            <p style="margin:5px 0 0 0; font-weight:600">💡 Tip: {verdict['strategic_tip']}</p>
        </div>
    """, unsafe_allow_html=True)

    # Split View: Logic vs Momentum
    tab1, tab2 = st.tabs(["💰 Price Logic", "📈 Market Momentum"])
    
    with tab1:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Fair Value", f"${fair_price_est:.2f}", delta=f"{verdict['gap_percent']}% Gap", delta_color="inverse")
            st.metric("Est. Margin", f"{((user_price - user_cost)/user_price)*100:.1f}%")
        
        with col_b:
            # Range Chart
            range_df = pd.DataFrame({
                'Price': [fair_price_est*0.85, fair_price_est, fair_price_est*1.15, user_price],
                'Label': ['Market Low', 'Fair Value', 'Market High', 'Your Price'],
                'Color': ['Range', 'Range', 'Range', 'You']
            })
            c = alt.Chart(range_df).mark_bar(cornerRadius=5).encode(
                x=alt.X('Price', scale=alt.Scale(zero=False)),
                y=alt.Y('Label', sort=['Your Price', 'Market High', 'Fair Value', 'Market Low']),
                color=alt.Color('Color', scale=alt.Scale(domain=['Range', 'You'], range=['#CFD8DC', '#2E7D32']), legend=None)
            ).properties(height=200)
            st.altair_chart(c, use_container_width=True)

    with tab2:
        col_c, col_d = st.columns([1, 2])
        with col_c:
            st.markdown(f"### {advice['action']}")
            st.caption(f"Confidence: {trend_pred['confidence']}%")
            st.info(advice['risk_note'])
        
        with col_d:
            # Trend Chart
            line = alt.Chart(history).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('date', format='%b %d', title=None),
                y=alt.Y('market_price', scale=alt.Scale(zero=False), title="Price ($)"),
                tooltip=['date', 'market_price']
            ).properties(height=200, title="Last 6 Months Trend")
            st.altair_chart(line, use_container_width=True)

# --- 7. The Ledger (Memory) ---
if st.session_state.ledger:
    st.divider()
    st.subheader("📝 Session Log")
    st.dataframe(
        pd.DataFrame(st.session_state.ledger),
        use_container_width=True,
        hide_index=True
    )
