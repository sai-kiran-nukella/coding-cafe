import streamlit as st

def set_style():
    """
    Applies the 'Slate & Soil' visual identity.
    - Typography: Inter (Clean, legible, modern sans-serif)
    - Palette: 
        * Action Green (#2E7D32): Growth, Go, Profit.
        * Trust Navy (#102A43): Stability, Text.
        * Paper White (#F9FAFB): Background, Readability.
    """
    
    # 1. Page Configuration (Must be the first Streamlit command)
    st.set_page_config(
        page_title="PriceWise | Strategic Pricing Tool",
        page_icon="⚖️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # 2. Inject Custom CSS
    st.markdown("""
        <style>
        /* --- IMPORT FONTS --- */
        @import url('https://googleapis.com');

        /* --- GLOBAL RESET --- */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #102A43;
        }
        
        /* Background Color */
        .stApp {
            background-color: #F9FAFB;
        }

        /* --- TYPOGRAPHY --- */
        h1 {
            font-weight: 700;
            color: #102A43;
            font-size: 2.2rem !important;
            letter-spacing: -0.02em;
        }
        h2, h3 {
            font-weight: 600;
            color: #243B53;
            letter-spacing: -0.01em;
        }
        
        /* Subheaders / Labels */
        div[class*="stMarkdown"] p {
            font-size: 1.05rem;
            line-height: 1.6;
        }

        /* --- UI COMPONENTS --- */

        /* Primary Action Button (The 'Analyze' Button) */
        .stButton > button {
            background-color: #2E7D32; /* Soil Green */
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.6rem 1rem;
            font-weight: 600;
            font-size: 1rem;
            width: 100%;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(46, 125, 50, 0.2);
        }
        .stButton > button:hover {
            background-color: #1B5E20;
            color: white;
            box-shadow: 0 4px 8px rgba(46, 125, 50, 0.3);
            transform: translateY(-1px);
        }
        .stButton > button:active {
            transform: translateY(0px);
        }

        /* Input Fields (Number Input, Selectbox) */
        div[data-baseweb="input"] {
            border-radius: 6px;
            border: 1px solid #CFD8DC;
            background-color: #FFFFFF;
        }
        div[data-baseweb="select"] {
            border-radius: 6px;
        }

        /* Metrics Containers */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #627D98;
            font-weight: 500;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem;
            color: #102A43;
            font-weight: 700;
        }

        /* --- CUSTOM VERDICT BOXES --- */
        .verdict-box {
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 24px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.03);
            border: 1px solid transparent;
        }
        
        /* Fair: Gentle Green */
        .verdict-fair {
            background-color: #F1F8E9;
            border-left: 6px solid #2E7D32;
            border-color: #C5E1A5;
            color: #1B5E20;
        }
        
        /* Overpriced: Alert Red */
        .verdict-over {
            background-color: #FFEBEE;
            border-left: 6px solid #C62828;
            border-color: #FFCDD2;
            color: #B71C1C;
        }
        
        /* Underpriced/Warning: Warm Amber */
        .verdict-under {
            background-color: #FFFDE7;
            border-left: 6px solid #F9A825;
            border-color: #FFF59D;
            color: #F57F17;
        }

        /* --- ADVICE & CONTEXT CARDS --- */
        
        /* Strategy Card (Blue/Neutral) */
        .advice-card {
            background-color: #F0F4F8; /* Light slate */
            padding: 18px;
            border-radius: 8px;
            border: 1px solid #D9E2EC;
            color: #334E68;
            line-height: 1.5;
        }
        
        /* Real World Context (Yellow/Note) */
        .real-world-box {
            background-color: #FFFFF0; /* Ivory */
            padding: 16px;
            border-radius: 8px;
            border-left: 4px solid #F0D000;
            border: 1px solid #F5F5DC;
            color: #746000;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* --- UTILITIES --- */
        hr {
            margin: 2rem 0;
            border-color: #E0E0E0;
        }
        
        </style>
    """, unsafe_allow_html=True)
