"""Session-local dashboard appearance; no global Streamlit config mutation."""

import streamlit as st


def apply_dashboard_style(dark):
    palette = (
        ("#141923", "#1E2531", "#273140", "#EDF0F7", "#AFB9CB", "#354052")
        if dark else
        ("#F3F5F9", "#FFFFFF", "#F0F2F7", "#202736", "#626D80", "#E4E8F0")
    )
    background, panel, field, text, muted, border = palette
    st.html(f"""<style>
    .stApp {{ background: {background}; color: {text}; }}
    [data-testid="stHeader"] {{ background: {background}; }}
    [data-testid="stMain"] {{ color: {text}; }}
    [data-testid="stMain"] :is(h1,h2,h3,p,label,[data-testid="stMetricValue"],
      [data-testid="stMetricLabel"], [data-testid="stWidgetLabel"]) {{ color: {text}; }}
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {{ color: {muted}; }}
    [data-testid="stMain"] h1 {{ font-size: 1.85rem; letter-spacing: -.035em; }}
    [data-testid="stMain"] h3 {{ font-size: 1.12rem; }}
    .st-key-input_panel, .st-key-segmentation_panel, .st-key-classification_panel {{
      background: {panel}; border: 1px solid {border}; border-radius: 14px;
      padding: 22px; box-shadow: 0 4px 18px #09122b06;
    }}
    [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] {{
      background: {field}; border: 1px dashed {border}; border-radius: 10px;
    }}
    [data-testid="stMain"] button {{ background: {field}; color: {text}; border-color: {border}; }}
    [data-testid="stMain"] button p {{ color: inherit; }}
    [data-testid="stMain"] button[kind="primary"] {{
      background: linear-gradient(110deg,#BD287F,#6543D5); color: white; border: none;
      box-shadow: 0 3px 12px #7543c524;
    }}
    [data-testid="stMain"] button:disabled {{ opacity: .45; }}
    [data-testid="stMain"] details {{ background: {panel}; border-color: {border}; color: {text}; }}
    [data-testid="stMain"] summary {{ color: {text}; }}
    [data-testid="stMain"] [data-testid="stMetricValue"] {{ font-size: 2rem; }}
    [data-testid="stSidebar"] {{ background: #1B222C; color: #E9EDF5; }}
    @media(max-width:640px) {{
      .st-key-input_panel, .st-key-segmentation_panel, .st-key-classification_panel {{ padding: 14px; }}
    }}
    </style>""")
