import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import plotly.graph_objects as go
import plotly.express as px
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import google.generativeai as genai
# from google.colab import userdata
import PIL.Image
import os
# import gdown
# from dotenv import load_dotenv
# load_dotenv()
import requests
import json
import time
from skimage import measure
from skimage.draw import polygon
from streamlit_option_menu import option_menu
# from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import tempfile

# Configure page 
st.set_page_config(
    page_title="NeuroLens Brain Tumor Analysis", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure API
try:
    # Get API key from Streamlit secrets
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except Exception as e:
    st.warning("Google API key not configured. AI explanations will have limited functionality.")
    # Don't expose the full error details in production

# Create directories
output_dir = 'saliency_maps'
reports_dir = 'reports'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# Initialize session state for tracking user history
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Analysis"

# Removing Lottie animation functions
# Define static image URLs for fallbacks - use more reliable image URLs (only as last resort)
brain_image_fallback = "https://cdn-icons-png.flaticon.com/512/4616/4616734.png"
analysis_image_fallback = "https://cdn-icons-png.flaticon.com/512/2329/2329087.png"

# Custom CSS for better styling
st.markdown(
    """
    <style>
    /* Set the entire page background */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"], .block-container {
        background-color: #0E1117; /* Dark blue-gray */
        color: #FFFFFF; /* White text for readability */
    }
    
    /* Hide the footer */
    footer {visibility: hidden;}

    /* Clean sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem 1rem;
    }
    
    /* Add a subtle gradient background to sidebar */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0E1117, #171923);
    }
    
    /* SUPER AGGRESSIVE RADIO BUTTON STYLING */
    /* Force label text to be visible */
    div[data-testid="stRadio"] > label {
        color: #FFFFFF !important; 
        font-size: 18px !important;
        font-weight: 700 !important;
        margin-bottom: 15px !important;
        display: block !important;
        text-shadow: 0 0 8px rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Force radio container to stand out */
    div[data-testid="stRadio"] {
        background-color: transparent !important;
        padding: 15px !important;
        border: 2px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 10px !important;
        margin-bottom: 20px !important;
    }
    
    /* Make radio button container visible */
    div[role="radiogroup"] > div {
        background-color: rgba(30, 30, 30, 0.8) !important;
        border-radius: 8px !important;
        padding: 5px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        margin-bottom: 10px !important;
    }
    
    /* Make individual radio buttons HUGE and bright */
    div[role="radiogroup"] input[type="radio"] {
        width: 28px !important;
        height: 28px !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: inline-block !important;
        margin-right: 10px !important;
        cursor: pointer !important;
        accent-color: #FFFFFF !important;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.8) !important;
        border: 3px solid #FFFFFF !important;
    }
    
    /* Make sure the text in radio buttons is extremely visible */
    div[role="radiogroup"] label {
        padding: 10px 15px !important;
        margin: 5px !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        border-radius: 8px !important;
        background-color: rgba(50, 50, 50, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Super bright text for radio options */
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span {
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Extremely obvious selected state */
    div[role="radiogroup"] label[data-checked="true"] {
        background-color: rgba(0, 119, 255, 0.5) !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 0 20px rgba(0, 119, 255, 0.6) !important;
    }
    
    /* Extremely visible selected text */
    div[role="radiogroup"] label[data-checked="true"] p,
    div[role="radiogroup"] label[data-checked="true"] span {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 16px !important;
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.8) !important;
    }
    
    /* Even more aggressive styling */
    body div[data-testid="stRadio"] > div > div > div[role="radiogroup"] label {
        background-color: rgba(50, 50, 50, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
    }
    
    body div[data-testid="stRadio"] > div > div > div[role="radiogroup"] label[data-checked="true"] {
        background-color: rgba(0, 119, 255, 0.5) !important;
        border: 2px solid #FFFFFF !important;
    }

    /* Set text color for the file uploader label (e.g., "Choose an image") */
    div[data-testid="stFileUploader"] > label {
        color: #FFFFFF !important; /* White color for the file uploader label */
    }

    /* Target the image caption */
    div.stImage div {
        color: #FFFFFF !important; /* Set caption text color to white */
    }

    /* Card styling */
    .card {
        border-radius: 10px;
        background-color: #1F2937;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Info box styling */
    .info-box {
        background-color: rgba(49, 51, 63, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #2196F3;
    }

    /* Warning box styling */
    .warning-box {
        background-color: rgba(49, 51, 63, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #FFC107;
    }

    /* Success box styling */
    .success-box {
        background-color: rgba(49, 51, 63, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #FFFFFF;
    }

    /* Custom header with gradient */
    .gradient-header {
        background: linear-gradient(90deg, #8A2387, #E94057, #F27121);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        font-weight: bold;
    }

    /* Enhanced Button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
        background-color: #1E293B;
        color: white;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.4);
        background-color: #2C3E50;
    }
    
    /* Specific button types */
    /* Primary action buttons (like Download, Start Analysis) */
    .stButton button[kind="primary"], 
    [data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #1E293B, #2C3E50);
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        border: 1px solid rgba(33, 150, 243, 0.3);
    }
    
    /* Select sample buttons */
    [data-testid="baseButton-secondary"] {
        background-color: #1E293B;
        border: 1px solid rgba(108, 117, 125, 0.3);
        color: #E0E0E0;
    }
    
    [data-testid="baseButton-secondary"]:hover {
        background-color: #2C3E50;
        box-shadow: 0 5px 15px rgba(255, 255, 255, 0.1);
    }
    
    /* Download buttons */
    .stDownloadButton button {
        background: linear-gradient(90deg, #0F2027, #203A43, #2C5364);
        color: white;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(17, 153, 142, 0.4);
        background: linear-gradient(90deg, #203A43, #2C5364, #0F2027);
    }
    
    /* Select sample buttons customization */
    .sample-select-btn button {
        background: linear-gradient(90deg, #1E293B, #2C3E50);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        margin-top: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .sample-select-btn button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(123, 31, 162, 0.4);
        background: linear-gradient(90deg, #2C3E50, #3E5771);
    }
    
    /* Model selection buttons */
    div[role="radiogroup"] label div {
        background-color: transparent;
        border-radius: 0;
        padding: 0;
        transition: none;
        border: none;
    }
    
    div[role="radiogroup"] label:hover div {
        background-color: transparent;
        box-shadow: none;
    }
    
    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        font-weight: normal;
        color: inherit !important;
    }

    /* Active tab indicator */
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #2196F3 !important;
        color: white !important;
    }
    
    /* Unselected tab styling */
    div[data-baseweb="tab-list"] button {
        background-color: #1E293B !important;
        color: #E0E0E0 !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 0.5rem 1rem !important;
        margin-right: 4px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Tab hover effect */
    div[data-baseweb="tab-list"] button:hover {
        background-color: #2C3E50 !important;
        color: white !important;
    }
    
    /* Tab panel styling */
    div[data-baseweb="tab-panel"] {
        background-color: #1F2937 !important;
        border-radius: 0 6px 6px 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 16px !important;
    }
    
    /* Style the entire tabs container */
    [data-testid="stTabs"] {
        background-color: transparent !important;
        margin-bottom: 1rem !important;
    }
    
    /* Style for the tab content area  */
    [data-testid="stTabs"] > div:nth-child(2) {
        background-color: #1F2937 !important;
        border-radius: 0 8px 8px 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 16px !important;
    }
    
    /* Custom hero section */
    .hero-container {
        display: flex;
        align-items: center;
        margin-bottom: 3rem;
        background: linear-gradient(90deg, rgba(10, 10, 20, 0.7), rgba(10, 10, 20, 0.8));
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Custom metrics container */
    .metrics-container {
        display: flex;
        justify-content: space-between;
        padding: 1rem;
        border-radius: 10px;
        background-color: #1F2937;
        margin-bottom: 2rem;
    }
    
    .metric-item {
        text-align: center;
        padding: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4CAF50;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #B0BEC5;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #1F2937 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
    }
    
    /* File uploader drag area */
    [data-testid="stFileUploader"] > section {
        background-color: #1E293B !important;
        border-radius: 6px !important;
        border: 1px dashed rgba(255, 255, 255, 0.3) !important;
        color: #E0E0E0 !important;
    }
    
    /* File uploader button */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, #1E293B, #2C3E50) !important;
        color: white !important;
        border-radius: 6px !important;
        border: 1px solid rgba(33, 150, 243, 0.3) !important;
        font-weight: 500 !important;
    }
    
    /* File uploader button hover */
    [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(90deg, #2C3E50, #3E5771) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.2) !important;
    }
    
    /* File uploader text color */
    [data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] p {
        color: #B0BEC5 !important;
    }
    
    /* File uploader uploaded file */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background-color: #1E293B !important;
        color: #E0E0E0 !important;
    }
    
    /* Streamlit inputs (text, number, etc.) */
    .stTextInput input, .stNumberInput input, .stDateInput input, .stTimeInput input {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 6px !important;
    }
    
    /* Streamlit inputs focus state */
    .stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus, .stTimeInput input:focus {
        border: 1px solid rgba(33, 150, 243, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.2) !important;
    }
    
    /* Streamlit select boxes */
    .stSelectbox > div, .stMultiSelect > div {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 6px !important;
    }
    
    /* Streamlit select box options */
    .stSelectbox [data-baseweb="select"] ul {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
    }
    
    /* Streamlit select box option hover */
    .stSelectbox [data-baseweb="select"] ul li:hover {
        background-color: #2C3E50 !important;
    }
    
    /* Metric containers */
    [data-testid="stMetric"] {
        background-color: #1F2937 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Success/info/error/warning message styling */
    .element-container .stAlert {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
    }
    
    /* Success message */
    .element-container [data-baseweb="notification"] {
        background-color: #1E293B !important;
        border-left: 4px solid #4CAF50 !important;
    }
    
    /* Error message */
    .element-container .stException, .element-container [data-testid="stDecoration"] {
        background-color: #1E293B !important;
        border-left: 4px solid #FF5252 !important;
        color: #FFFFFF !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 500 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        background-color: #1F2937 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        padding: 1rem !important;
    }
    
    /* Data display (dataframes, tables) */
    .stDataFrame, div[data-testid="stTable"] {
        background-color: #1E293B !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Table headers */
    .stDataFrame th, div[data-testid="stTable"] th {
        background-color: #2C3E50 !important;
        color: #FFFFFF !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Table cells */
    .stDataFrame td, div[data-testid="stTable"] td {
        background-color: #1E293B !important;
        color: #E0E0E0 !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Alternating rows */
    .stDataFrame tr:nth-child(even) td, div[data-testid="stTable"] tr:nth-child(even) td {
        background-color: #212b3b !important;
    }
    
    /* Plotly chart background */
    .js-plotly-plot .plotly {
        background-color: #1F2937 !important;
    }
    
    /* General containers */
    [data-testid="stVerticalBlock"] > div {
        background-color: transparent !important;
    }
    
    /* Card containers */
    .css-1r6slb0, .css-12w0qpk {
        background-color: #1F2937 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    /* Code blocks */
    pre, code {
        background-color: #1E293B !important;
        color: #E0E0E0 !important;
        border-radius: 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation - completely rewritten for better appearance
with st.sidebar:
    st.markdown("""
    <h1 style='
        text-align: center; 
        font-size: 1.8rem; 
        margin-bottom: 25px; 
        font-weight: 600;
        background: linear-gradient(90deg, #1565C0, #0D47A1, #1565C0); 
        -webkit-background-clip: text; 
        background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        justify-content: center;
    '>
        <span style="margin-right: 8px;">🧠</span>
        <span>NeuroLens</span>
    </h1>
    """, unsafe_allow_html=True)
    
    # Add custom CSS just for the navigation menu
    st.markdown("""
    <style>
    /* Option menu container */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }
    
    /* Remove background from all sidebar elements */
    section[data-testid="stSidebarUserContent"] * {
        background-color: transparent !important;
    }
    
    /* Add hover effect for nav links */
    .nav-link:hover {
        transform: translateY(-2px) !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        background-color: #263952 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add some space
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    
    # Simplified option menu with cleaner styling
    selected = option_menu(
        menu_title=None,
        options=["Home", "Analysis", "About"],
        icons=["house", "activity", "info-circle"],
        menu_icon="cast",
        default_index=["Home", "Analysis", "About"].index(st.session_state.current_page),
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"font-size": "18px", "color": "#90CAF9"}, 
            "nav-link": {
                "font-size": "15px", 
                "text-align": "left", 
                "margin": "10px 0", 
                "padding": "12px 15px",
                "border-radius": "7px",
                "background-color": "#1E293B",
                "color": "#E0E0E0",
                "font-weight": "400",
                "border": "1px solid rgba(255, 255, 255, 0.05)",
                "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                "transition": "all 0.3s ease"
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, #1565C0, #0D47A1)",
                "color": "#FFFFFF",
                "font-weight": "500",
                "border": "none",
                "box-shadow": "0 2px 6px rgba(21, 101, 192, 0.4)"
            },
        }
    )
    
    # Only update the page if a new page is selected
    # This is the key change - don't update state if it's already that page
    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        # Force a rerun to properly update the page content
        st.rerun()
    
    # Add copyright info with cleaner styling
    st.markdown("<div style='position: fixed; bottom: 20px; left: 0; right: 0; text-align: center;'><hr style='margin: 20px 15px; border-color: rgba(255,255,255,0.05);'></div>", unsafe_allow_html=True)
    st.markdown("<div style='position: fixed; bottom: 0; left: 0; right: 0; text-align: center; padding: 10px; color: #78909C; font-size: 12px;'>© 2023 NeuroLens</div>", unsafe_allow_html=True)

def generate_explanation(img_path, model_prediction, confidence):
    try:
        # Check if the file exists first
        if not os.path.exists(img_path):
            # Check if the directory exists
            directory = os.path.dirname(img_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                st.warning(f"Created missing directory: {directory}")
            
            st.warning(f"Saliency map file not found at {img_path}. Using fallback explanation.")
            return f"""
            Based on the model analysis, this appears to be a {model_prediction.lower()} with {confidence*100:.1f}% confidence.
            
            Note: Detailed AI-powered explanation is not available because the saliency map image could not be found.
            """

        # The original prompt is unchanged
        prompt = f"""You are an expert neurologist specializing in brain tumor diagnosis through MRI scans. You have been given a saliency map generated by a deep learning model that was trained to classify brain tumors into one of four categories: glioma, meningioma, pituitary tumor, or no tumor. The saliency map identifies which areas of the MRI scan the model is focusing on to make its prediction.

                  The model has predicted the scan to belong to the class '{model_prediction}' with a confidence of {confidence * 100}%. Your task is to explain the saliency map and the reasoning behind the model's prediction.

                  In your explanation:
                  1. Identify and describe the brain regions highlighted in the saliency map, particularly those marked in light cyan. Explain how these regions are relevant to the type of tumor the model predicts, focusing on the typical locations of such tumors in the brain.
                  2. Provide a brief, scientifically-backed rationale for why the model might have made this prediction based on the highlighted regions. Refer to general knowledge of brain tumor characteristics (e.g., gliomas often appear in specific brain regions, meningiomas are typically on the outer layers of the brain, etc.).
                  3. Your response should avoid stating basic facts like 'The saliency map highlights areas in light cyan,' and instead focus directly on the clinical reasoning behind the model's focus on these regions.
                  4. Limit your explanation to no more than 4 sentences.

                  Please carefully consider the model's prediction and the anatomical and clinical implications of the highlighted regions. Let's think step by step about this. Verify step by step."""

        img = PIL.Image.open(img_path)

        # Check if Google API is configured
        try:
            model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite")
            response = model.generate_content([prompt, img])
            return response.text
        except Exception as e:
            st.warning(f"Could not generate AI explanation. Error: {str(e)}")
            return f"""
            Based on the highlighted regions in the saliency map, this appears to be a {model_prediction.lower()} with {confidence*100:.1f}% confidence. 
            
            Note: Detailed AI-powered explanation is not available due to API connection issues. Please ensure your Google API key is properly configured.
            """
    except PIL.UnidentifiedImageError:
        st.warning(f"Could not open saliency map image at {img_path}. The file may be corrupted or in an unsupported format.")
        return f"Based on the model analysis, this appears to be a {model_prediction.lower()} with {confidence*100:.1f}% confidence. Detailed explanation unavailable due to image processing error."
    except Exception as e:
        st.error(f"Error in explanation generation: {str(e)}")
        return "Could not generate explanation due to an error."

def generate_saliency_map(model, img_array, class_index, img_size, file_path, file_name):
  with tf.GradientTape() as tape:
    img_tensor = tf.convert_to_tensor(img_array)
    tape.watch(img_tensor)
    predictions = model(img_tensor)
    target_class = predictions[:, class_index]

  gradients = tape.gradient(target_class, img_tensor)
  gradients = tf.math.abs(gradients)
  gradients = tf.reduce_max(gradients, axis=-1)
  gradients = gradients.numpy().squeeze()

  # Resize gradients to match original image size
  gradients = cv2.resize(gradients, img_size)

  # Create a circular mask for the brain area
  center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
  radius = min(center[0], center[1]) - 10
  y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
  mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

  # Apply mask to gradients
  gradients = gradients * mask

  # Normalize only the brain area
  brain_gradients = gradients[mask]
  if brain_gradients.max() > brain_gradients.min():
    brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
  gradients[mask] = brain_gradients

  # Apply a higher threshold
  threshold = np.percentile(gradients[mask], 80)
  gradients[gradients < threshold] = 0

  # Apply more aggressive smoothing
  gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

  # Create a heatmap overlay with enhanced contrast
  heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

  # Resize heatmap to match original image size
  heatmap = cv2.resize(heatmap, img_size)

  # Superimpose the heatmap on original image with increased opacity
  original_img = image.img_to_array(img)
  superimposed_img = heatmap * 0.7 + original_img * 0.3
  superimposed_img = superimposed_img.astype(np.uint8)

  img_path = os.path.join(output_dir, file_name)
  with open(file_path, "rb") as f_in:
      with open(img_path, "wb") as f_out:
          f_out.write(f_in.read())

  saliency_map_path = f'saliency_maps/{file_name}'

  # Save the saliency map
  cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

  return superimposed_img

def load_xception_model(model_path):
    try:
        img_shape=(299,299,3)
        base_model = tf.keras.applications.Xception(
            include_top=False, 
            weights="imagenet", 
            input_shape=img_shape, 
            pooling='max'
        )

        model = Sequential([
            base_model,
            Flatten(),
            Dropout(rate=0.3),
            Dense(128, activation='relu'),
            Dropout(rate=0.25),
            Dense(4, activation='softmax')
        ])

        model.build((None,) + img_shape)


        model.load_weights(model_path)

        # Compile after loading weights
        model.compile(
            optimizer=Adamax(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )

        return model
    except Exception as e:
        st.error(f"Error in load_xception_model: {str(e)}")
        return None

# New function for tumor segmentation 
def segment_tumor(img, prediction_class):
    # Basic segmentation using threshold - in a real app you'd use a dedicated segmentation model
    # This is a simplified version for demonstration
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold - different for each tumor type
    if prediction_class == 'Glioma':
        thresh = 0.65
    elif prediction_class == 'Meningioma':
        thresh = 0.60
    elif prediction_class == 'Pituitary':
        thresh = 0.55
    else:  # No tumor
        return np.zeros_like(img_gray), 0, "N/A"
    
    # Apply threshold
    _, binary = cv2.threshold(img_gray, int(255 * thresh), 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours = measure.find_contours(binary, 0.5)
    
    # Create mask
    mask = np.zeros_like(img_gray)
    for contour in contours:
        contour = np.flip(contour, axis=1)
        rr, cc = polygon(contour[:, 0], contour[:, 1], mask.shape)
        mask[rr, cc] = 1
        
    # Calculate area (simplified)
    tumor_area = np.sum(mask) / (mask.shape[0] * mask.shape[1])
    tumor_area_mm = tumor_area * 100  # Simulated conversion to mm²
    
    # Estimate diameter (assuming circular shape)
    tumor_diameter = 2 * np.sqrt(tumor_area_mm / np.pi)
    
    return mask, tumor_area_mm, f"{tumor_diameter:.2f} mm"

# New function for generating PDF report
def generate_report(original_img, saliency_map, prediction, confidence, explanation):
    """Generate a clean and readable HTML report for the brain tumor analysis.
    
    Args:
        original_img: The original MRI image
        saliency_map: The generated saliency map highlighting areas of interest
        prediction: The model's prediction (tumor type)
        confidence: The confidence score of the prediction
        explanation: The AI-generated explanation of the findings
    
    Returns:
        HTML string representing the formatted report
    """
    # Create a BytesIO buffer for storing the images
    buffer_original = io.BytesIO()
    buffer_saliency = io.BytesIO()
    
    # Convert arrays to PIL Images if needed
    if isinstance(original_img, np.ndarray):
        original_img_pil = PIL.Image.fromarray(original_img.astype('uint8'))
    else:
        original_img_pil = original_img
        
    if isinstance(saliency_map, np.ndarray):
        saliency_map_pil = PIL.Image.fromarray(saliency_map.astype('uint8'))
    else:
        saliency_map_pil = saliency_map
    
    # Save images to buffers with high quality
    original_img_pil.save(buffer_original, format='PNG', quality=95)
    saliency_map_pil.save(buffer_saliency, format='PNG', quality=95)
    
    # Convert to base64 for embedding in HTML
    buffer_original.seek(0)
    buffer_saliency.seek(0)
    img_str_original = base64.b64encode(buffer_original.read()).decode()
    img_str_saliency = base64.b64encode(buffer_saliency.read()).decode()
    
    # Determine text color based on prediction
    if prediction == "No Tumor":
        prediction_color = "#4CAF50"  # Green
    else:
        prediction_color = "#FF5722"  # Orange/Red
    
    # Format confidence as percentage
    confidence_formatted = f"{confidence:.2%}"
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create an HTML string with modern styling
    html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NeuroLens Brain Tumor Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
            }}
            .report-container {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 30px;
                margin-bottom: 30px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            .header h1 {{
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 28px;
            }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 14px;
                margin-top: 10px;
            }}
            .images-container {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }}
            .image-box {{
                width: 48%;
                margin-bottom: 20px;
                border-radius: 6px;
                overflow: hidden;
                box-shadow: 0 1px 5px rgba(0,0,0,0.1);
            }}
            .image-box h3 {{
                background-color: #2c3e50;
                color: white;
                margin: 0;
                padding: 10px 15px;
                font-size: 16px;
            }}
            .image-box img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            .results-container {{
                background-color: #f8f9fa;
                border-radius: 6px;
                padding: 25px;
                margin-bottom: 30px;
            }}
            .prediction {{
                font-size: 24px;
                color: {prediction_color};
                margin-bottom: 10px;
                font-weight: bold;
            }}
            .confidence {{
                font-size: 18px;
                color: #34495e;
                margin-bottom: 20px;
            }}
            .explanation-container {{
                background-color: #fff;
                border-left: 4px solid #3498db;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 0 6px 6px 0;
            }}
            .explanation-container h3 {{
                color: #2c3e50;
                margin-top: 0;
            }}
            .explanation-text {{
                font-size: 16px;
                line-height: 1.7;
                color: #34495e;
            }}
            .disclaimer {{
                font-size: 12px;
                color: #7f8c8d;
                font-style: italic;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 6px;
                margin-top: 30px;
            }}
            @media (max-width: 768px) {{
                .image-box {{
                    width: 100%;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <div class="header">
                <h1>NeuroLens Brain Tumor Analysis Report</h1>
                <div class="timestamp">Generated on: {timestamp}</div>
            </div>
            
            <div class="images-container">
                <div class="image-box">
                    <h3>Original MRI Scan</h3>
                    <img src="data:image/png;base64,{img_str_original}" alt="Original MRI Scan">
                </div>
                <div class="image-box">
                    <h3>Saliency Map Analysis</h3>
                    <img src="data:image/png;base64,{img_str_saliency}" alt="Saliency Map">
                </div>
            </div>
            
            <div class="results-container">
                <h2>Analysis Results</h2>
                <div class="prediction">Diagnosis: {prediction}</div>
                <div class="confidence">Confidence: {confidence_formatted}</div>
            </div>
            
            <div class="explanation-container">
                <h3>Expert Analysis</h3>
                <div class="explanation-text">
                    {explanation}
                </div>
            </div>
            
            <div class="disclaimer">
                <strong>Disclaimer:</strong> This analysis is generated by an AI model and should be reviewed by a healthcare professional. 
                NeuroLens is designed as a research and assistive tool and is not intended to replace professional medical diagnosis or advice.
            </div>
        </div>
    </body>
    </html>
    '''
    
    # Return the HTML string
    return html

# Add a utility function to handle different types of file inputs and resize images if needed
def process_image_input(file_input, max_dimension=800):
    """Process various image input types and optionally resize large images.
    
    Args:
        file_input: Can be a path string, BytesIO object, or a file-like object (e.g., from st.file_uploader)
        max_dimension: Maximum width or height for displayed images
    
    Returns:
        A tuple of (processed_file_path, original_file_name)
    """
    # Create a temporary file to store the image data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    
    # Get the original filename if available
    if hasattr(file_input, 'name'):
        original_filename = os.path.basename(file_input.name)
    else:
        original_filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    
    # Process the input based on its type
    if isinstance(file_input, str):
        # It's a file path
        if os.path.isfile(file_input):
            with open(file_input, 'rb') as f:
                temp_file.write(f.read())
    elif hasattr(file_input, 'read'):
        # It's a file-like object (UploadedFile or BufferedReader)
        if hasattr(file_input, 'getbuffer'):
            # UploadedFile from streamlit
            temp_file.write(file_input.getbuffer())
        else:
            # BufferedReader or similar
            temp_file.write(file_input.read())
            # Seek back to start for potential reuse
            if hasattr(file_input, 'seek'):
                file_input.seek(0)
    else:
        # Unsupported type
        temp_file.close()
        os.unlink(temp_file.name)
        raise TypeError(f"Unsupported file input type: {type(file_input)}")
    
    temp_file.close()
    
    # Check if image needs resizing
    try:
        img = PIL.Image.open(temp_file.name)
        w, h = img.size
        
        # Only resize if image is too large
        if max(w, h) > max_dimension:
            # Calculate new dimensions while maintaining aspect ratio
            if w > h:
                new_w = max_dimension
                new_h = int(h * (max_dimension / w))
            else:
                new_h = max_dimension
                new_w = int(w * (max_dimension / h))
            
            # Resize and save back to temp file
            img = img.resize((new_w, new_h), PIL.Image.LANCZOS)
            img.save(temp_file.name)
    except Exception as e:
        print(f"Warning: Could not resize image: {e}")
    
    return temp_file.name, original_filename

# Function to display sample images in grid - defined at global scope so it can be used in multiple pages
def display_sample_grid(samples, category_name, max_display_size=300):
    """Display a grid of sample images with selection buttons.
    
    Args:
        samples: List of sample file names/paths
        category_name: Name of the tumor category for display
        max_display_size: Maximum dimension for image display
        
    Returns:
        Selected sample path or None if no selection made
    """
    if not samples:
        st.info(f"No {category_name} samples found. You can add sample images in the About page.")
        return None
            
    st.write(f"**Available {category_name} samples:**")
    
    # Create a grid of images (3 columns)
    cols = st.columns(min(3, len(samples)))
    selected_sample = None
    
    for i, sample_file in enumerate(samples):
        with cols[i % 3]:
            sample_path = os.path.join("sample_images", sample_file)
            
            # Check if the sample path is a directory
            if os.path.isdir(sample_path):
                # Get all image files from this directory
                dir_images = [f for f in os.listdir(sample_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if dir_images:
                    # Use the first image from the directory
                    sample_path = os.path.join(sample_path, dir_images[0])
                else:
                    # Skip if no images in directory
                    continue
            
            # Check if the path exists and is a file before displaying
            if os.path.isfile(sample_path):
                # Resize image for display
                try:
                    img = PIL.Image.open(sample_path)
                    w, h = img.size
                    
                    # Only resize if image is too large
                    if max(w, h) > max_display_size:
                        # Calculate new dimensions while maintaining aspect ratio
                        if w > h:
                            new_w = max_display_size
                            new_h = int(h * (max_display_size / w))
                        else:
                            new_h = max_display_size
                            new_w = int(w * (max_display_size / h))
                        
                        # Use st.image with width parameter instead of resizing the actual image
                        st.image(sample_path, caption=f"Sample {i+1}", width=new_w)
                    else:
                        st.image(sample_path, caption=f"Sample {i+1}", use_column_width=True)
                except Exception as e:
                    st.warning(f"Error displaying image: {str(e)}")
                    st.image(sample_path, caption=f"Sample {i+1}", use_column_width=True)
                
                if st.button(f"Select Sample {i+1}", key=f"{category_name}_{i}", use_container_width=True):
                    selected_sample = sample_path
    
    return selected_sample

# Main app logic based on navigation
if st.session_state.current_page == "Home":
    # Create hero section
    st.markdown("<h1 class='gradient-header' style='text-align: center; font-size: 3rem;'>Welcome to NeuroLens</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.5rem; margin-bottom: 2rem;'>Advanced Brain Tumor Analysis Using Deep Learning</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🧠 What is NeuroLens?</h3>
            <p>NeuroLens is a state-of-the-art deep learning application designed to assist medical professionals in the analysis of brain MRI scans for tumor detection and classification.</p>
            <p>Using advanced convolutional neural networks and explainable AI, NeuroLens provides detailed insights into brain tumor classification with visual explanations of the model's focus areas.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Important Disclaimer</h3>
            <p>NeuroLens is a research tool and not a replacement for professional medical diagnosis. All results should be interpreted by qualified healthcare professionals.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Static brain MRI image instead of Lottie animation
        st.markdown("""
        <div style="background: linear-gradient(145deg, rgba(33, 150, 243, 0.1), rgba(156, 39, 176, 0.1)); 
                    border-radius: 12px; 
                    padding: 20px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    height: 280px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                    text-align: center;">
            <img src="https://cdn-icons-png.flaticon.com/512/4616/4616734.png" 
                 style="max-width: 150px; margin-bottom: 20px;" alt="Brain MRI">
            <h3 style="color: #90CAF9; margin-bottom: 10px;">Advanced MRI Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("<h2 style='margin-top: 2rem;'>Key Features</h2>", unsafe_allow_html=True)
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div class="card">
            <h3 style="color: #4CAF50;">🔍 Tumor Classification</h3>
            <p>Classify brain tumors into multiple categories: Glioma, Meningioma, Pituitary, or No Tumor with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with feature_col2:
        st.markdown("""
        <div class="card">
            <h3 style="color: #2196F3;">🔥 Saliency Maps</h3>
            <p>Visualize which areas of the brain scan the model is focusing on to make its predictions, enhancing explainability.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with feature_col3:
        st.markdown("""
        <div class="card">
            <h3 style="color: #FFC107;">📊 Detailed Analysis</h3>
            <p>Get comprehensive probability scores for each class and expert-like explanations of the findings using AI.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("<div style='text-align: center; margin-top: 2rem;'>", unsafe_allow_html=True)
    
    # Remove the Start Analysis button and styling
    # Instead, add an informational message
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(33, 150, 243, 0.1), rgba(33, 150, 243, 0.2));
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #2196F3;
                margin: 20px 0;">
        <h3 style="color: #2196F3; margin-top: 0;">Ready to Analyze Brain MRI Scans?</h3>
        <p>You're currently viewing the Home page. Navigate to the <b>Analysis</b> tab in the sidebar to upload and analyze MRI scans.</p>
        <p>The Analysis tab is the main workspace where you can classify brain tumors using our advanced deep learning models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sample results section
    st.markdown("<h2 style='margin-top: 2rem;'>How It Works</h2>", unsafe_allow_html=True)
    
    workflow_col1, workflow_col2, workflow_col3, workflow_col4 = st.columns(4)
    
    with workflow_col1:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h3 style="color: #9C27B0;">1</h3>
            <p>Upload your brain MRI scan</p>
        </div>
        """, unsafe_allow_html=True)
        
    with workflow_col2:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h3 style="color: #9C27B0;">2</h3>
            <p>Select a deep learning model</p>
        </div>
        """, unsafe_allow_html=True)
        
    with workflow_col3:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h3 style="color: #9C27B0;">3</h3>
            <p>Run the analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
    with workflow_col4:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h3 style="color: #9C27B0;">4</h3>
            <p>Review detailed results and explanations</p>
        </div>
        """, unsafe_allow_html=True)
        
elif st.session_state.current_page == "Analysis":
    # Analysis page with reorganized flow and simplified model selection

    # Only keep one title
    st.title("Brain Tumor Classification")
    
    # Create a model selection header
    st.subheader("Select Model for Analysis")
    
    # Initialize model_choice from session state or set default to "Xception"
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "Xception"
    
    # The absolute most basic radio implementation possible, with a container to make it stand out
    with st.container():
        st.markdown("""
        <style>
        div[data-testid="stRadio"] > div {
            background-color: #ffffff !important;
            padding: 15px !important;
            border-radius: 8px !important;
            margin-bottom: 20px !important;
        }
        div[data-testid="stRadio"] label {
            color: #000000 !important;
            font-size: 18px !important;
            font-weight: bold !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] > label {
            padding: 10px !important;
            margin: 5px !important;
            background-color: #f0f0f0 !important;
            border-radius: 5px !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
            background-color: #d0d0d0 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Add back the radio button for model selection
        model_choice = st.radio(
            "",  # No label since we already have a subheader
            ["Xception", "Custom CNN"],
            horizontal=True  # Horizontal layout for better visibility
        )
    
    # Update the session state
    st.session_state.model_choice = model_choice
    
    # Display selected model in a visible card
    st.markdown(f"""
    <div style="
        background-color: #28a745; 
        color: white; 
        padding: 10px; 
        border-radius: 5px; 
        text-align: center; 
        margin-bottom: 20px;
        font-weight: bold;">
        Currently using: {model_choice} model
    </div>
    """, unsafe_allow_html=True)
    
    # Load the selected model immediately (before image upload)
    # This prevents having to reload after selecting a model
    if model_choice == "Xception":
        try:
            model = load_xception_model('xception_model.weights.h5')
            if model is None:
                raise Exception("Failed to load Xception model")
            img_size = (299,299)
        except Exception as e:
            st.error(f"Error loading Xception model: {str(e)}")
            st.stop()
    else:  # Custom CNN
        try:
            model = load_model('cnn_model.h5', compile=False)
            if model is None:
                raise Exception("Failed to load CNN model")
            model.compile(
                optimizer=Adamax(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', Precision(), Recall()]
            )
            img_size = (224,224)
        except Exception as e:
            st.error(f"Error loading CNN model: {str(e)}")
            st.stop()
    
    # SECOND - Now show tabs for upload or sample selection
    upload_tab, sample_tab = st.tabs(["Upload Your Image", "Try Sample Images"])
    
    with upload_tab:
        st.write("Upload an image of a brain MRI scan to classify.")
        uploaded_file = st.file_uploader("Choose an image....", type=["jpg", "jpeg", "png"])
    
    with sample_tab:
        st.write("Select a sample brain MRI scan to analyze.")
        
        # Create directory for sample images if it doesn't exist
        sample_dir = "sample_images"
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get currently available samples
        sample_files = []
        sample_dirs = []
        if os.path.exists(sample_dir):
            # Get all items in the directory
            dir_items = os.listdir(sample_dir)
            
            # Separate files and directories
            sample_files = [f for f in dir_items if os.path.isfile(os.path.join(sample_dir, f))]
            sample_dirs = [d for d in dir_items if os.path.isdir(os.path.join(sample_dir, d))]
            
            # For each subdirectory, check for image files inside
            for d in sample_dirs:
                subdir_path = os.path.join(sample_dir, d)
                subdir_files = [f for f in os.listdir(subdir_path) 
                                 if os.path.isfile(os.path.join(subdir_path, f)) 
                                 and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                # Add the subdirectory to the sample count if it contains images
                if subdir_files:
                    if d.lower() not in [f.lower() for f in sample_files]:
                        sample_files.append(d)

        # Categorize existing samples
        glioma_samples = [f for f in sample_files if "glioma" in f.lower()]
        meningioma_samples = [f for f in sample_files if "meningioma" in f.lower()]
        no_tumor_samples = [f for f in sample_files if "no_tumor" in f.lower() or "normal" in f.lower()]
        pituitary_samples = [f for f in sample_files if "pituitary" in f.lower()]
        
        # Create a tab for each tumor type
        tumor_tabs = st.tabs(["Glioma", "Meningioma", "No Tumor", "Pituitary"])
        
        # Display sample grids in each tab and handle selection
        with tumor_tabs[0]:
            selected_glioma = display_sample_grid(glioma_samples, "Glioma")
            if selected_glioma:
                uploaded_file, uploaded_file_name = process_image_input(selected_glioma)
                st.success(f"Selected Glioma sample for analysis")
        
        with tumor_tabs[1]:
            selected_meningioma = display_sample_grid(meningioma_samples, "Meningioma")
            if selected_meningioma:
                uploaded_file, uploaded_file_name = process_image_input(selected_meningioma)
                st.success(f"Selected Meningioma sample for analysis")
        
        with tumor_tabs[2]:
            selected_no_tumor = display_sample_grid(no_tumor_samples, "No Tumor")
            if selected_no_tumor:
                uploaded_file, uploaded_file_name = process_image_input(selected_no_tumor)
                st.success(f"Selected No Tumor sample for analysis")
        
        with tumor_tabs[3]:
            selected_pituitary = display_sample_grid(pituitary_samples, "Pituitary")
            if selected_pituitary:
                uploaded_file, uploaded_file_name = process_image_input(selected_pituitary)
                st.success(f"Selected Pituitary sample for analysis")
        
        # If no samples found in any category, show a message
        if not (glioma_samples or meningioma_samples or no_tumor_samples or pituitary_samples):
            st.warning("""
            No sample images found. To use this feature:
            
            1. Go to the About page
            2. Expand the "🖼️ Manage Sample Images" section 
            3. Upload sample images for each tumor type
            """)
            st.info("You can continue using the 'Upload Your Image' tab to upload your own images.")
            uploaded_file = None
    
    # THIRD - Process the image if one is selected
    if uploaded_file is not None:
        # If uploaded_file is a string (from sample selection), it's already processed
        # If it's from the file uploader, process it
        if not isinstance(uploaded_file, str):
            uploaded_file, uploaded_file_name = process_image_input(uploaded_file)
        
        # Store current image info in session state
        st.session_state.current_image_path = uploaded_file
        st.session_state.current_image_name = uploaded_file_name
        
        # Add a divider before analysis
        st.markdown("<hr style='margin: 20px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.subheader(f"Analyzing with {model_choice} Model")
        
        with st.spinner(f"Processing image with {model_choice} model..."):
            # Process the image with the selected model
            labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            img = image.load_img(st.session_state.current_image_path, target_size=img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            class_index = np.argmax(prediction[0])
            result = labels[class_index]

            # Generate saliency map
            saliency_map = generate_saliency_map(model, img_array, class_index, img_size, 
                                               st.session_state.current_image_path, 
                                               st.session_state.current_image_name)

            # Display images
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.current_image_path, caption="Uploaded Image", use_container_width=True)
            with col2:
                st.image(saliency_map, caption="Saliency Map", use_container_width=True)

            # Show results
            st.write("## Classification Results")
            st.markdown(
                f"""
                <div style="background-color: #000000; color: #ffffff; padding: 30px; border-radius: 15px;">
                  <div style="display: flex; justify-content: space-between; align-items: center:">
                    <div style="flex: 1; text-align: center;">
                      <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Prediction</h3>
                      <p style="font-size: 36px; font-weight: 800; color: #FF0000; margin: 0;">
                        {result}
                      </p>
                    </div>
                    <div style="width: 2px; height: 80px; background-color: #ffffff; margin: 0 20px;"></div>
                    <div style="flex: 1; text-align: center;">
                      <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Confidence</h3>
                      <p style="font-size: 36px; font-weight: 800; color: #2196F3; margin: 0;">
                        {prediction[0][class_index]:.4%}
                      </p>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Probability breakdown
            st.write("### Probability Breakdown")
            st.write("Detailed probability scores for each tumor class:")
            
            probabilities = prediction[0]
            sorted_indices = np.argsort(probabilities)[::-1]
            sorted_labels = [labels[i] for i in sorted_indices]
            sorted_probabilities = probabilities[sorted_indices]

            # Bar chart
            fig = go.Figure(go.Bar(
                x=sorted_probabilities,
                y=sorted_labels,
                orientation='h',
                marker_color=['red' if label == result else 'blue' for label in sorted_labels]
            ))

            fig.update_layout(
                title=f'Probability Analysis ({model_choice} Model)',
                xaxis=dict(
                  title='Probability',
                  title_font=dict(color='#FFFFFF'),
                  tickfont=dict(color='#FFFFFF')
                ),
                yaxis=dict(
                  title='Class',
                  title_font=dict(color='#FFFFFF'),
                  tickfont=dict(color='#FFFFFF'),
                  autorange='reversed'
                ),
                height=400,
                # Remove fixed width to allow responsive sizing
                plot_bgcolor='#1c1c1c',
                paper_bgcolor='#1c1c1c',
                font=dict(color='#FFFFFF'),
                title_font=dict(color='#FFFFFF'),
                margin=dict(l=50, r=50, t=80, b=50)  # Add proper margins
            )

            # Add value labels
            for i, prob in enumerate(sorted_probabilities):
              fig.add_annotation(
                  x=prob,
                  y=i,
                  text=f'{prob:.4f}',
                  showarrow=False,
                  xanchor='left',
                  xshift=5
              )

            # Use the full width of the container
            st.plotly_chart(fig, use_container_width=True)

            # Model explanation
            saliency_map_path = f'saliency_maps/{st.session_state.current_image_name}'
            explanation = generate_explanation(saliency_map_path, result, prediction[0][class_index])

            st.write("## Explanation:")
            st.write(explanation)

            # Generate and prepare download report
            report_html = generate_report(image.img_to_array(img), saliency_map, result, prediction[0][class_index], explanation)
            report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"neurolens_report_{result.lower()}_{report_time}.html"
            
            report_path = os.path.join(reports_dir, report_filename)
            with open(report_path, "w") as f:
                f.write(report_html)
            
            st.write("## Download Analysis Report")
            st.write("You can download a detailed report of this analysis for offline viewing or sharing.")
            
            # Download button styling
            st.markdown("""
            <style>
            div.stDownloadButton > button {
                background: linear-gradient(90deg, #11998e, #38ef7d);
                color: white;
                font-weight: 600;
                padding: 0.7rem 1.5rem;
                border-radius: 8px;
                border: none;
                transition: all 0.3s ease;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            }
            div.stDownloadButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(17, 153, 142, 0.4);
            }
            </style>
            """, unsafe_allow_html=True)
            
            with open(report_path, "r") as f:
                st.download_button(
                    label="Download Analysis Report",
                    data=f,
                    file_name=report_filename,
                    mime="text/html",
                    use_container_width=True
                )
            
            # Update history
            if len(st.session_state.history) >= 10:
                st.session_state.history.pop(0)
            
            st.session_state.history.append({
                "filename": st.session_state.current_image_name,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "prediction": result,
                "confidence": prediction[0][class_index],
                "model": model_choice,
                "saliency_map_path": f'saliency_maps/{st.session_state.current_image_name}',
                "report_path": report_path
            })

elif st.session_state.current_page == "About":
    st.title("About NeuroLens")
    
    st.markdown("""
    <div class="info-box">
        <h3>Project Overview</h3>
        <p>NeuroLens is an advanced brain MRI analysis tool that leverages deep learning to assist medical professionals in detecting and classifying brain tumors.</p>
        <p>Developed as part of medical imaging research, this application showcases how AI can be used to enhance diagnostic capabilities in neurology.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Technical Details</h3>
        <p><strong>Models:</strong> The application uses two types of models:</p>
        <ul>
            <li><strong>Transfer Learning with Xception:</strong> A pre-trained Xception architecture fine-tuned on brain MRI data.</li>
            <li><strong>Custom CNN:</strong> A custom-built convolutional neural network specifically designed for brain tumor classification.</li>
        </ul>
        <p><strong>Explainability:</strong> Gradient-based saliency maps highlight regions that contribute most significantly to model decisions.</p>
        <p><strong>AI Explanations:</strong> Powered by Google's Gemini model, producing human-like explanations of findings.</p>
        <p><strong>User Interface:</strong> Built with Streamlit for a seamless, interactive experience.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Dataset</h3>
        <p>Models were trained on a dataset containing labeled MRI scans of:</p>
        <ul>
            <li>Glioma tumors</li>
            <li>Meningioma tumors</li>
            <li>Pituitary tumors</li>
            <li>No tumor (healthy) scans</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Remove the entire "Manage Sample Images" expander section
    
    st.markdown("### References & Acknowledgements")
    
    # Remove the column structure and Research Papers section
    st.markdown("""
    <div class="card">
        <h4>Technologies Used</h4>
        <ul>
            <li>TensorFlow & Keras for deep learning models</li>
            <li>Streamlit for the web interface</li>
            <li>Plotly for interactive visualizations</li>
            <li>OpenCV for image processing</li>
            <li>Google's Gemini for explanations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box" style="margin-top: 20px;">
        <h4>Disclaimer</h4>
        <p>NeuroLens is a demonstration tool and is not FDA-approved for clinical use. All predictions should be verified by qualified healthcare professionals. This application is intended for research and educational purposes only.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Contact & Feedback")
    
    st.markdown("""
    <div class="card">
        <p>For questions, feedback, or collaboration opportunities, please contact:</p>
        <p><strong>Email:</strong> neurolens.project@example.com</p>
        <p><strong>GitHub:</strong> <a href="https://huggingface.co/spaces/Pranavch/neurolens-brain-tumor" style="color: #2196F3;">Pranavch/neurolens-brain-tumor</a></p>
    </div>
    """, unsafe_allow_html=True)

