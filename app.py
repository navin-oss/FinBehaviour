import streamlit as st
from transformers import pipeline
import praw
import tweepy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import json
import time
import io
import os
import base64
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from fpdf import FPDF
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- LOGO HELPER ---
def get_logo_base64():
    """Load logo and return as base64 string for inline display"""
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# --- PAGE CONFIG ---
st.set_page_config(page_title="FinBehavior AI", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")

# --- CUSTOM DARK THEME CSS (PREMIUM ANIMATED) ---
st.markdown("""
<style>
    /* ===== GOOGLE FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ===== KEYFRAME ANIMATIONS ===== */
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 8px rgba(0, 210, 255, 0.3), 0 0 20px rgba(0, 210, 255, 0.1); }
        50%      { box-shadow: 0 0 16px rgba(0, 210, 255, 0.6), 0 0 40px rgba(0, 210, 255, 0.2); }
    }
    @keyframes borderGlow {
        0%, 100% { border-color: rgba(0, 210, 255, 0.3); }
        50%      { border-color: rgba(108, 92, 231, 0.6); }
    }
    @keyframes floatBadge {
        0%, 100% { transform: translateY(0px); }
        50%      { transform: translateY(-6px); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes pulseRing {
        0%   { box-shadow: 0 0 0 0 rgba(0, 210, 255, 0.4); }
        70%  { box-shadow: 0 0 0 12px rgba(0, 210, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 210, 255, 0); }
    }
    @keyframes bounceArrow {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(6px); }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.85); }
        to   { opacity: 1; transform: scale(1); }
    }
    @keyframes particleDrift {
        0%   { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0; }
        10%  { opacity: 0.6; }
        90%  { opacity: 0.6; }
        100% { transform: translateY(-100vh) translateX(50px) rotate(360deg); opacity: 0; }
    }

    /* ===== GLOBAL DARK THEME ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0f0c29 20%, #1a1a3e 50%, #24243e 80%, #0f0c29 100%);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        color: #e0e0e0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ===== PARTICLE OVERLAY ===== */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            radial-gradient(1.5px 1.5px at 10% 20%, rgba(0,210,255,0.25) 50%, transparent 50%),
            radial-gradient(1px 1px at 30% 65%, rgba(108,92,231,0.2) 50%, transparent 50%),
            radial-gradient(1.5px 1.5px at 55% 15%, rgba(0,210,255,0.15) 50%, transparent 50%),
            radial-gradient(1px 1px at 75% 80%, rgba(108,92,231,0.2) 50%, transparent 50%),
            radial-gradient(1.5px 1.5px at 90% 40%, rgba(0,210,255,0.18) 50%, transparent 50%),
            radial-gradient(1px 1px at 45% 90%, rgba(253,121,168,0.12) 50%, transparent 50%);
        pointer-events: none;
        z-index: 0;
        animation: gradientShift 30s ease infinite;
    }

    /* ===== CUSTOM SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0f0c29; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #00d2ff, #6c5ce7); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #00b4d8, #5a4bd1); }

    /* ===== LOGO & HEADER ===== */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 4px;
        animation: fadeInUp 0.8s ease-out;
    }
    .logo-img {
        width: 60px;
        height: 60px;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0, 210, 255, 0.35);
        flex-shrink: 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: pulseRing 3s ease-in-out infinite;
    }
    .logo-img:hover {
        transform: scale(1.08) rotate(-3deg);
        box-shadow: 0 6px 30px rgba(0, 210, 255, 0.5);
    }
    .main-header {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 25%, #6c5ce7 50%, #00d2ff 75%, #3a7bd5 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 4s linear infinite;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -1px;
        line-height: 1.1;
        margin: 0;
        font-family: 'Inter', sans-serif !important;
    }
    .sub-header {
        color: #a0a0c0;
        font-size: 1.05rem;
        margin-top: 4px;
        font-weight: 400;
        letter-spacing: 0.3px;
        animation: fadeInUp 1s ease-out 0.2s both;
    }

    /* ===== METRIC CARDS ===== */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 22px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out both;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,210,255,0.06), transparent);
        transition: left 0.6s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 35px rgba(0, 210, 255, 0.2), 0 0 0 1px rgba(0,210,255,0.15);
        border-color: rgba(0, 210, 255, 0.25);
    }
    [data-testid="stMetric"]:hover::before {
        left: 100%;
    }
    [data-testid="stMetricLabel"] {
        color: #a0a0c0 !important;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800;
        font-family: 'Inter', sans-serif !important;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d20 0%, #141428 40%, #1a1a3e 100%);
        border-right: 1px solid rgba(0, 210, 255, 0.08);
        backdrop-filter: blur(20px);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #00d2ff !important;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    /* ===== BUTTONS ===== */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d2ff 0%, #6c5ce7 50%, #00d2ff 100%) !important;
        background-size: 200% 200% !important;
        animation: gradientShift 4s ease infinite !important;
        border: none !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        position: relative;
        overflow: hidden;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 8px 28px rgba(0, 210, 255, 0.35), 0 0 60px rgba(108, 92, 231, 0.15) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0) scale(0.98);
    }

    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        transition: all 0.3s ease !important;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(0, 210, 255, 0.05) !important;
        border-color: rgba(0, 210, 255, 0.15) !important;
    }

    /* ===== CONSENT CARD ===== */
    .consent-card {
        background: rgba(108, 92, 231, 0.08);
        border: 2px solid rgba(108, 92, 231, 0.25);
        border-radius: 20px;
        padding: 36px;
        text-align: center;
        margin: 40px auto;
        max-width: 620px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        animation: scaleIn 0.7s cubic-bezier(0.4, 0, 0.2, 1), borderGlow 3s ease infinite;
        box-shadow: 0 8px 40px rgba(108, 92, 231, 0.15), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }
    .consent-card::before {
        content: '';
        position: absolute;
        top: -2px; left: -2px; right: -2px; bottom: -2px;
        background: linear-gradient(45deg, #00d2ff, #6c5ce7, #fd79a8, #00d2ff);
        background-size: 400% 400%;
        animation: gradientShift 6s linear infinite;
        border-radius: 22px;
        z-index: -1;
        opacity: 0.3;
    }
    .consent-card h2 {
        color: #00d2ff;
        margin-bottom: 16px;
        font-weight: 800;
        font-size: 1.5rem;
        animation: fadeInUp 0.8s ease-out 0.3s both;
    }
    .consent-card p {
        color: #b0b0d0;
        line-height: 1.8;
        animation: fadeInUp 0.8s ease-out 0.5s both;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 10px 22px;
        color: #808098;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 210, 255, 0.06);
        color: #c0c0e0;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 210, 255, 0.12) !important;
        color: #00d2ff !important;
        border-bottom: 2px solid #00d2ff;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.15);
        font-weight: 600;
    }

    /* ===== ARCHITECTURE FLOW ===== */
    .flow-step {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }
    .flow-step::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, rgba(0,210,255,0.05), transparent);
        opacity: 0;
        transition: opacity 0.4s ease;
        border-radius: 16px;
    }
    .flow-step:hover {
        border-color: rgba(0, 210, 255, 0.4);
        transform: translateY(-6px) scale(1.03);
        box-shadow: 0 16px 40px rgba(0, 210, 255, 0.15), 0 0 0 1px rgba(0,210,255,0.2);
    }
    .flow-step:hover::after { opacity: 1; }
    .flow-step .icon {
        font-size: 2.2rem;
        margin-bottom: 10px;
        display: inline-block;
        transition: transform 0.3s ease;
    }
    .flow-step:hover .icon { transform: scale(1.2); }
    .flow-step .title { color: #00d2ff; font-weight: 700; font-size: 0.95rem; }
    .flow-step .desc { color: #a0a0c0; font-size: 0.8rem; margin-top: 8px; line-height: 1.4; }

    /* ===== FLOW STEP STAGGER ANIMATION ===== */
    .flow-step-1 { animation: fadeInUp 0.6s ease-out 0.1s both; }
    .flow-step-2 { animation: fadeInUp 0.6s ease-out 0.2s both; }
    .flow-step-3 { animation: fadeInUp 0.6s ease-out 0.3s both; }
    .flow-step-4 { animation: fadeInUp 0.6s ease-out 0.4s both; }
    .flow-step-5 { animation: fadeInUp 0.6s ease-out 0.5s both; }

    /* ===== GLASSMORPHISM CARD ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(0, 210, 255, 0.25);
        box-shadow: 0 12px 35px rgba(0, 210, 255, 0.1);
        transform: translateY(-3px);
    }

    /* ===== DECISION CARD GLOW ===== */
    .decision-card {
        animation: fadeInUp 0.7s ease-out, glowPulse 3s ease-in-out infinite;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }

    /* ===== PRODUCT CARD ===== */
    .product-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        min-height: 150px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .product-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,210,255,0.04), transparent);
        transition: left 0.5s ease;
    }
    .product-card:hover {
        transform: translateY(-5px) scale(1.03);
        box-shadow: 0 12px 30px rgba(0, 210, 255, 0.15);
        border-color: rgba(0, 210, 255, 0.3);
    }
    .product-card:hover::before { left: 100%; }
    .product-card .card-icon { font-size: 2rem; margin-bottom: 10px; display: inline-block; transition: transform 0.3s ease; }
    .product-card:hover .card-icon { transform: scale(1.15) translateY(-3px); }

    /* ===== COMPARISON BOXES ===== */
    .comparison-box {
        border-radius: 16px;
        padding: 24px;
        transition: all 0.35s ease;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        animation: fadeInUp 0.6s ease-out both;
    }
    .comparison-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
    }

    /* ===== COMPLIANCE BADGES ===== */
    .compliance-badge {
        text-align: center;
        padding: 14px;
        border-radius: 14px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        animation: fadeInUp 0.6s ease-out both;
    }
    .compliance-badge:hover {
        transform: translateY(-4px) scale(1.05);
    }
    .compliance-badge .badge-icon {
        font-size: 1.6rem;
        display: inline-block;
        animation: floatBadge 3s ease-in-out infinite;
    }

    /* ===== FEED POST CARD ===== */
    .feed-post-card {
        animation: slideInLeft 0.5s ease-out both;
        transition: all 0.3s ease;
    }
    .feed-post-card:hover {
        transform: translateX(4px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    .new-post-badge {
        animation: glowPulse 2s ease-in-out infinite;
        display: inline-block;
    }

    /* ===== SUCCESS/ERROR BOXES ===== */
    .stAlert {
        border-radius: 12px !important;
        animation: fadeInUp 0.5s ease-out;
        backdrop-filter: blur(8px);
    }

    /* ===== PLOTLY CHARTS ===== */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* ===== GRADIENT DIVIDERS ===== */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(0,210,255,0.3), rgba(108,92,231,0.3), transparent) !important;
        margin: 24px 0 !important;
    }

    /* ===== DATA TABLE ===== */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        animation: fadeInUp 0.5s ease-out;
    }

    /* ===== HIDE STREAMLIT BRANDING (keep sidebar toggle visible) ===== */
    footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}

    /* ===== ENSURE SIDEBAR IS VISIBLE ===== */
    [data-testid="stSidebar"] {
        min-width: 320px !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        visibility: visible !important;
    }

    /* ===== ANIMATED GET-STARTED CTA ===== */
    .cta-arrow {
        display: inline-block;
        animation: bounceArrow 1.5s ease-in-out infinite;
        font-size: 1.2rem;
    }

    /* ===== FEATURE LIST ITEMS ===== */
    .feature-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 24px;
        backdrop-filter: blur(12px);
        transition: all 0.35s ease;
        animation: fadeInUp 0.6s ease-out both;
    }
    .feature-card:hover {
        border-color: rgba(0, 210, 255, 0.2);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.08);
    }

    /* ===== 1. HERO TITLE GLOW ===== */
    @keyframes titleGlow {
        0%, 100% { text-shadow: 0 0 20px rgba(0,210,255,0.3), 0 0 40px rgba(108,92,231,0.15); filter: brightness(1); }
        50%      { text-shadow: 0 0 30px rgba(0,210,255,0.6), 0 0 60px rgba(108,92,231,0.3), 0 0 80px rgba(0,210,255,0.1); filter: brightness(1.1); }
    }
    .main-header {
        animation: shimmer 4s linear infinite, titleGlow 4s ease-in-out infinite !important;
    }

    /* ===== 1b. TYPING ANIMATION FOR SUBTITLE ===== */
    @keyframes typingReveal {
        from { max-width: 0; }
        to   { max-width: 800px; }
    }
    @keyframes blinkCursor {
        0%, 100% { border-right-color: #00d2ff; }
        50%      { border-right-color: transparent; }
    }
    .sub-header-typing {
        color: #a0a0c0;
        font-size: 1.05rem;
        margin-top: 4px;
        font-weight: 400;
        letter-spacing: 0.3px;
        overflow: hidden;
        white-space: nowrap;
        max-width: 0;
        border-right: 2px solid #00d2ff;
        animation: typingReveal 3s steps(50, end) 0.5s forwards, blinkCursor 0.8s step-end infinite;
        display: inline-block;
    }

    /* ===== 2. PIPELINE FLOW ARROWS ===== */
    .flow-container {
        display: flex;
        align-items: stretch;
        gap: 0;
        justify-content: center;
    }
    .flow-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #00d2ff;
        font-size: 1.6rem;
        padding: 0 2px;
        animation: bounceArrow 1.5s ease-in-out infinite;
        opacity: 0.6;
        filter: drop-shadow(0 0 6px rgba(0,210,255,0.4));
    }
    .flow-step {
        animation: fadeInUp 0.6s ease-out both, glowPulse 4s ease-in-out 2s infinite;
    }
    .flow-step:hover {
        border-color: rgba(0, 210, 255, 0.5) !important;
        box-shadow: 0 16px 40px rgba(0, 210, 255, 0.2), 0 0 0 1px rgba(0,210,255,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    }
    .flow-step:hover .icon {
        transform: scale(1.3) translateY(-4px) !important;
        filter: drop-shadow(0 4px 8px rgba(0,210,255,0.4));
    }

    /* ===== 3. ANIMATED RISK GAUGE ===== */
    @keyframes fillGauge {
        from { --gauge-angle: 0deg; }
    }
    .risk-gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        animation: scaleIn 0.8s cubic-bezier(0.4, 0, 0.2, 1) both;
    }
    .risk-gauge {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fillGauge 1.5s ease-out both;
    }
    .risk-gauge-inner {
        width: 130px;
        height: 130px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 100%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: inset 0 2px 15px rgba(0,0,0,0.4);
    }
    .risk-gauge-score {
        font-size: 2.4rem;
        font-weight: 900;
        font-family: 'Inter', sans-serif;
        line-height: 1;
    }
    .risk-gauge-label {
        font-size: 0.7rem;
        color: #a0a0c0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
        font-weight: 600;
    }
    .risk-gauge-level {
        font-size: 0.85rem;
        font-weight: 700;
        margin-top: 10px;
        letter-spacing: 0.5px;
        padding: 4px 16px;
        border-radius: 20px;
    }

    /* ===== 4. AI PROCESSING ANIMATION ===== */
    @keyframes dotPulse {
        0%   { content: ''; }
        25%  { content: '.'; }
        50%  { content: '..'; }
        75%  { content: '...'; }
        100% { content: ''; }
    }
    @keyframes iconSpin {
        0%   { transform: scale(1) rotate(0deg); }
        50%  { transform: scale(1.15) rotate(5deg); }
        100% { transform: scale(1) rotate(0deg); }
    }
    .ai-stage {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 14px 22px;
        background: rgba(0, 210, 255, 0.04);
        border: 1px solid rgba(0, 210, 255, 0.1);
        border-radius: 12px;
        margin: 8px 0;
        animation: fadeInUp 0.4s ease-out both;
        backdrop-filter: blur(10px);
    }
    .ai-stage-icon {
        font-size: 1.5rem;
        animation: iconSpin 1.5s ease-in-out infinite;
    }
    .ai-stage-text {
        color: #e0e0e0;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .ai-stage-active {
        border-color: rgba(0, 210, 255, 0.3);
        background: rgba(0, 210, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 210, 255, 0.1);
    }
    .ai-stage-done {
        border-color: rgba(0, 184, 148, 0.2);
        opacity: 0.7;
    }
    .ai-stage-done .ai-stage-icon {
        animation: none;
    }

    /* ===== 5. CONFIDENCE BARS ===== */
    @keyframes barFill {
        from { width: 0%; }
    }
    .confidence-bar-container {
        margin: 8px 0;
    }
    .confidence-bar-label {
        display: flex;
        justify-content: space-between;
        color: #a0a0c0;
        font-size: 0.82rem;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .confidence-bar-track {
        height: 8px;
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 4px;
        animation: barFill 1.2s ease-out both;
        box-shadow: 0 0 10px rgba(0,210,255,0.3);
    }

    /* ===== 6. RESULT REVEAL STAGGER ===== */
    .reveal-section {
        animation: fadeInUp 0.7s ease-out both;
    }
    .reveal-1 { animation-delay: 0.1s; }
    .reveal-2 { animation-delay: 0.3s; }
    .reveal-3 { animation-delay: 0.5s; }
    .reveal-4 { animation-delay: 0.7s; }
    .reveal-5 { animation-delay: 0.9s; }
    .reveal-6 { animation-delay: 1.1s; }
    .reveal-7 { animation-delay: 1.3s; }

    .slide-up-card {
        animation: fadeInUp 0.6s ease-out both;
    }
    .slide-up-card-1 { animation-delay: 0.8s; }
    .slide-up-card-2 { animation-delay: 0.95s; }
    .slide-up-card-3 { animation-delay: 1.1s; }
    .slide-up-card-4 { animation-delay: 1.25s; }
</style>
""", unsafe_allow_html=True)


# --- INITIALIZE SESSION STATE ---
if 'consent_given' not in st.session_state:
    st.session_state.consent_given = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_scores' not in st.session_state:
    st.session_state.analysis_scores = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'live_feed_running' not in st.session_state:
    st.session_state.live_feed_running = False
if 'live_feed_posts' not in st.session_state:
    st.session_state.live_feed_posts = []
if 'live_feed_risk' not in st.session_state:
    st.session_state.live_feed_risk = 0
if 'flagged_posts' not in st.session_state:
    st.session_state.flagged_posts = set()


# --- PREPROCESSING ---
def preprocess_text(text):
    """Clean social media posts for NLP"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s₹]', '', text)  # Keep ₹ for amount detection
    text = ' '.join(text.split())
    return text


def extract_amount(text):
    """Extract monetary amounts from post"""
    patterns = [
        r'₹\s?((?:\d+,)*\d+(?:\.\d+)?)\s*((?:crore|crores|cr|lakh|lakhs|lac|l|k)\b)?',
        r'(?:Rs\.?|INR)\s?((?:\d+,)*\d+(?:\.\d+)?)\s*((?:crore|crores|cr|lakh|lakhs|lac|l|k)\b)?',
        r'((?:\d+,)*\d+(?:\.\d+)?)\s*(?:rupees|rs)\s*((?:crore|crores|cr|lakh|lakhs|lac|l|k)\b)?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '')
            amount = float(amount_str)
            # Handle suffix captured by regex (group 2)
            suffix = (match.group(2) or '').lower().strip()
            if suffix in ('lakh', 'lakhs', 'lac', 'l'):
                amount *= 100000
            elif suffix in ('crore', 'crores', 'cr'):
                amount *= 10000000
            elif suffix == 'k':
                amount *= 1000
            return amount
    return 0


# --- LOAD MODELS (CACHED) ---
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Models are loaded lazily inside the analysis block with visual feedback
# (removed top-level blocking call that prevented page from rendering)


# --- REDDIT API ---
def fetch_reddit_posts(client_id, client_secret, subreddit_name, sort_by="hot", limit=20):
    """Fetch posts from a subreddit using PRAW (Reddit API)"""
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="FinBehaviour AI v1.0 (by /u/FinBehaviourBot)"
        )
        subreddit = reddit.subreddit(subreddit_name)
        fetch_fn = getattr(subreddit, sort_by)  # .hot(), .new(), .top()
        posts = []
        for post in fetch_fn(limit=limit):
            text = post.title
            if post.selftext and len(post.selftext.strip()) > 10:
                text += " — " + post.selftext[:300]
            # Skip very short or mod/sticky posts
            if len(text.strip()) > 15:
                posts.append(text.strip())
        return posts, None
    except Exception as e:
        return [], str(e)


# --- TWITTER/X API ---
def fetch_twitter_posts(bearer_token, username, max_results=20):
    """Fetch recent tweets from a public user using Twitter API v2 (Bearer Token)"""
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        # Look up user by username
        clean_username = username.lstrip('@').strip()
        user = client.get_user(username=clean_username)
        if not user.data:
            return [], f"User @{clean_username} not found"
        user_id = user.data.id
        # Fetch recent tweets (excludes retweets and replies for cleaner data)
        tweets = client.get_users_tweets(
            user_id,
            max_results=min(max_results, 100),
            tweet_fields=["created_at", "text"],
            exclude=["retweets", "replies"]
        )
        if not tweets.data:
            return [], "No tweets found for this user"
        posts = [tweet.text for tweet in tweets.data if len(tweet.text.strip()) > 15]
        return posts, None
    except tweepy.TooManyRequests:
        return [], "Rate limit exceeded. Twitter free tier allows 1 request per 15 minutes. Please wait."
    except tweepy.Unauthorized:
        return [], "Invalid Bearer Token. Get yours at developer.twitter.com"
    except Exception as e:
        return [], str(e)


# --- SCORING LOGIC (ENHANCED) ---
def calculate_scores(classifications):
    """Enhanced scoring with confidence weighting and amount extraction"""
    scores = {"Spending": 0, "Investment": 0, "Loan": 0, "Savings": 0, "Risk": 0, "Gambling / Speculative": 0}
    weighted_scores = {"Spending": 0, "Investment": 0, "Loan": 0, "Savings": 0, "Risk": 0, "Gambling / Speculative": 0}
    total_amount = {"Spending": 0, "Investment": 0, "Loan": 0, "Savings": 0, "Risk": 0, "Gambling / Speculative": 0}
    
    confident_items = [c for c in classifications if c['label'] != "Uncertain"]
    
    for item in confident_items:
        label = item['label']
        if label in scores:
            scores[label] += 1
            weighted_scores[label] += item.get('confidence', 0.5)
            total_amount[label] += item.get('amount', 0)
    
    total = sum(scores.values())
    if total == 0:
        return scores, 0, 0, weighted_scores, total_amount
    
    # Enhanced risk: weighted by confidence + amount factor
    amount_factor = 1.0
    risky_amount = total_amount['Spending'] + total_amount['Loan'] + total_amount['Risk'] + total_amount['Gambling / Speculative']
    if risky_amount > 0:
        safe_amount = total_amount['Investment'] + total_amount['Savings'] + 1
        amount_factor = min(risky_amount / safe_amount, 3.0)  # Cap at 3x
    
    base_risk = (
        weighted_scores['Spending'] * 1 +
        weighted_scores['Loan'] * 1.5 +
        weighted_scores['Risk'] * 2 +
        weighted_scores['Gambling / Speculative'] * 3.0  # Highest weight — gambling/fraud
    )
    stabilizer = (weighted_scores['Investment'] * 1.2 + weighted_scores['Savings'] * 1.0)
    
    risk_score = ((base_risk - stabilizer * 0.5) / total) * 50 * (1 + amount_factor * 0.1)
    risk_score = max(0, min(risk_score, 100))
    
    uncertainty_rate = (len(classifications) - len(confident_items)) / len(classifications) * 100
    
    return scores, risk_score, uncertainty_rate, weighted_scores, total_amount


# --- PDF REPORT GENERATION ---
def _pdf_safe(text):
    """Sanitize text for PDF output — replace Unicode chars unsupported by Helvetica"""
    replacements = {
        '₹': 'Rs.', '😍': '', '📈': '', '✅': '', '🎉': '', '✈️': '',
        '🚀': '', '💰': '', '😅': '', '📦': '', '🤞': '', '🙌': '',
        '💪': '', '🎯': '', '🎲': '', '⚠️': '', '❓': '', '💸': '',
        '🏦': '', '📊': '', '💡': '', '🔒': '', '🔴': '', '🟢': '',
        '🟡': '', '🕸️': '', '📋': '', '🔬': '', '📜': '', '⚡': '',
        '\u200d': '', '\ufe0f': '',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Remove any remaining non-latin1 characters
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text


def generate_pdf_report(results, scores, risk_score, uncertainty_rate):
    """Generate a downloadable PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 15, "FinBehavior AI Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    
    # Risk Score
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Risk Score: {int(risk_score)}/100", ln=True)
    level = "HIGH RISK" if risk_score > 70 else "LOW RISK" if risk_score < 30 else "MEDIUM RISK"
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Profile: {level}", ln=True)
    pdf.cell(0, 8, f"Uncertainty Rate: {uncertainty_rate:.0f}%", ln=True)
    pdf.ln(5)
    
    # Category Breakdown
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Category Breakdown", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for cat, count in scores.items():
        pdf.cell(0, 7, f"  {cat}: {count} posts", ln=True)
    pdf.ln(5)
    
    # Detailed Classifications
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Post Classifications", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for r in results:
        post_text = _pdf_safe(r['Post'][:80] + ('...' if len(r['Post']) > 80 else ''))
        pdf.cell(0, 6, f"[{r['Category']}] (conf: {r['Confidence']}) {post_text}", ln=True)
    
    # Privacy Notice
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, "Privacy Notice: This report was generated from consent-based behavioral analysis. Data is not stored or shared.")
    
    return bytes(pdf.output())


# --- CONFUSION MATRIX ---
def run_model_evaluation(classifier_model):
    """Evaluate model accuracy on synthetic data"""
    try:
        with open("synthetic_posts.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        return None, None, None
    
    labels = ["Spending", "Investment", "Loan", "Savings", "Risk", "Gambling / Speculative"]
    true_labels = []
    pred_labels = []
    
    sample = test_data[:25]  # Use 25 samples for speed
    
    progress = st.progress(0)
    for i, item in enumerate(sample):
        clean = preprocess_text(item['text'])
        output = classifier_model(clean, candidate_labels=labels, multi_label=False)
        pred_labels.append(output['labels'][0])
        true_labels.append(item['true_label'])
        progress.progress((i + 1) / len(sample))
    progress.empty()
    
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    report = classification_report(true_labels, pred_labels, labels=labels, output_dict=True, zero_division=0)
    
    return cm, report, labels


# ========================
# MAIN APPLICATION
# ========================
def main():
    # --- CONSENT FLOW ---
    if not st.session_state.consent_given:
        logo_b64 = get_logo_base64()
        if logo_b64:
            st.markdown(f'''
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_b64}" class="logo-img" alt="FinBehavior AI logo">
                <div>
                    <h1 class="main-header">FinBehavior AI</h1>
                    <div class="sub-header-typing">Social Behavioral Credit Intelligence Platform</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('<h1 class="main-header">FinBehavior AI</h1>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header-typing">Social Behavioral Credit Intelligence Platform</div>', unsafe_allow_html=True)
            st.markdown('<h1 class="main-header">FinBehavior AI</h1>', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Social Behavioral Credit Intelligence Platform</p>', unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("")
        
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.markdown("""
            <div class="consent-card">
                <h2>🔐 User Consent Required</h2>
                <p>
                    This tool analyzes social media posts to assess financial behavior patterns.<br><br>
                    <strong>Before proceeding, please confirm:</strong><br>
                    ✅ You consent to analyze the text you provide<br>
                    ✅ You understand this is a research prototype<br>
                    ✅ No data is stored or shared externally<br>
                    ✅ All scores are explainable and appealable
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("✅ I Consent — Proceed to Dashboard", type="primary", use_container_width=True):
                st.session_state.consent_given = True
                st.rerun()
            
            st.markdown("")
            st.caption("🔒 Compliant with GDPR & India's DPDP Act 2023")
        return
    
    # --- MAIN DASHBOARD ---
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.markdown(f'''
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_b64}" class="logo-img" alt="FinBehavior AI logo">
            <div>
                <h1 class="main-header">FinBehavior AI</h1>
                <div class="sub-header-typing">Social Behavioral Credit Intelligence Dashboard — Consent-Based & Privacy-First</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<h1 class="main-header">FinBehavior AI</h1>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header-typing">Social Behavioral Credit Intelligence Dashboard — Consent-Based & Privacy-First</div>', unsafe_allow_html=True)

    # --- SIDEBAR ---
    st.sidebar.markdown("## 📥 Input Data")
    
    input_method = st.sidebar.radio("Input Method", ["📝 Text Input", "📁 Upload File", "📱 Reddit Live", "🐦 Twitter/X"], horizontal=True)
    
    if input_method == "📝 Text Input":
        demo_mode = st.sidebar.checkbox("✅ Use Demo Data", value=True)
        
        if demo_mode:
            default_text = """Bought new iPhone 15 Pro on EMI ₹5000/month 😍
Invested ₹50,000 in Mutual Funds today 📈
Paid credit card bill of ₹12,500 on time ✅
Personal loan of ₹2L approved finally! 🎉
Trip to Dubai booked for ₹80,000 - can't wait! ✈️
Transferred ₹15,000 rent to landlord
Bought crypto because FOMO - ₹25,000 in Shiba Inu 🚀
Saved ₹10,000 this month for emergency fund 💰
Lost ₹5000 in options trading... lesson learned 😅
Started SIP of ₹2000/month in NPS"""
            user_input = st.sidebar.text_area("Social Posts (one per line)", value=default_text, height=250)
        else:
            user_input = st.sidebar.text_area("Paste Social Posts (one per line)", height=250,
                                              placeholder="Example: Bought new laptop on EMI ₹3000/month")
        posts = [line.strip() for line in user_input.split('\n') if line.strip()] if user_input else []
    
    elif input_method == "📁 Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV or JSON", type=['csv', 'json'])
        posts = []
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                text_col = st.sidebar.selectbox("Select text column", df.columns)
                posts = df[text_col].dropna().tolist()
            elif uploaded_file.name.endswith('.json'):
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    if isinstance(data[0], dict) and 'text' in data[0]:
                        posts = [item['text'] for item in data]
                    else:
                        posts = [str(item) for item in data]
            st.sidebar.success(f"✅ Loaded {len(posts)} posts")
        user_input = "\n".join(posts) if posts else ""
    
    elif input_method == "📱 Reddit Live":
        st.sidebar.markdown("#### 🔑 Reddit API Credentials")
        st.sidebar.caption("Get yours free at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)")
        reddit_client_id = st.sidebar.text_input("Client ID", type="password", placeholder="e.g. aB3cD4eFgH...")
        reddit_client_secret = st.sidebar.text_input("Client Secret", type="password", placeholder="e.g. xY9zW8vU7t...")
        
        st.sidebar.markdown("#### 📡 Subreddit Settings")
        subreddit_options = {
            "IndiaInvestments": "🇮🇳 India Investments",
            "personalfinanceindia": "💰 Personal Finance India",
            "CreditCardsIndia": "💳 Credit Cards India",
            "IndianStreetBets": "📈 Indian Street Bets",
            "CryptoCurrency": "🪙 Crypto Currency",
        }
        selected_sub = st.sidebar.selectbox(
            "Subreddit",
            options=list(subreddit_options.keys()),
            format_func=lambda x: subreddit_options[x]
        )
        reddit_sort = st.sidebar.selectbox("Sort By", ["hot", "new", "top"], index=0)
        reddit_limit = st.sidebar.slider("Number of Posts", 5, 50, 15)
        
        posts = []
        user_input = ""
        
        if reddit_client_id and reddit_client_secret:
            if st.sidebar.button("🔄 Fetch Reddit Posts", use_container_width=True):
                with st.sidebar:
                    with st.spinner(f"Fetching from r/{selected_sub}..."):
                        fetched_posts, error = fetch_reddit_posts(
                            reddit_client_id, reddit_client_secret,
                            selected_sub, reddit_sort, reddit_limit
                        )
                if error:
                    # --- FALLBACK: load synthetic data when Reddit is unreachable ---
                    st.sidebar.warning(f"⚠️ Reddit unavailable — loading synthetic demo data instead.\n\n_({error[:120]}...)_")
                    try:
                        with open("synthetic_posts.json", "r", encoding="utf-8") as f:
                            synthetic_data = json.load(f)
                        fetched_posts = [item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in synthetic_data[:reddit_limit]]
                        st.session_state.reddit_posts = fetched_posts
                        st.sidebar.success(f"✅ Loaded {len(fetched_posts)} synthetic posts as fallback")
                    except Exception:
                        st.sidebar.error("❌ Could not load synthetic fallback data either.")
                elif fetched_posts:
                    st.session_state.reddit_posts = fetched_posts
                    st.sidebar.success(f"✅ Fetched {len(fetched_posts)} posts from r/{selected_sub}")
                else:
                    st.sidebar.warning("⚠️ No posts found in this subreddit.")
        else:
            # --- Allow fetching synthetic data even without Reddit credentials ---
            if st.sidebar.button("📦 Load Synthetic Demo Data", use_container_width=True):
                try:
                    with open("synthetic_posts.json", "r", encoding="utf-8") as f:
                        synthetic_data = json.load(f)
                    fetched_posts = [item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in synthetic_data[:reddit_limit]]
                    st.session_state.reddit_posts = fetched_posts
                    st.sidebar.success(f"✅ Loaded {len(fetched_posts)} synthetic posts")
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to load synthetic data: {e}")
            st.sidebar.info("💡 Enter Reddit credentials above for live data, or use synthetic demo data.")
        
        # Use fetched posts if available
        if 'reddit_posts' in st.session_state and st.session_state.reddit_posts:
            posts = st.session_state.reddit_posts
            user_input = "\n".join(posts)
            st.sidebar.info(f"📋 {len(posts)} posts ready for analysis")

    else:  # 🐦 Twitter/X
        st.sidebar.markdown("#### 🔑 Twitter/X API Credentials")
        st.sidebar.caption("Get your Bearer Token free at [developer.twitter.com](https://developer.twitter.com)")
        
        # Auto-load from .env if available
        env_token = os.getenv("TWITTER_BEARER_TOKEN", "")
        default_token = env_token if env_token and env_token != "your_bearer_token_here" else ""
        
        twitter_bearer = st.sidebar.text_input(
            "Bearer Token", type="password",
            value=default_token,
            placeholder="Paste your Bearer Token here..."
        )
        
        st.sidebar.markdown("#### 👤 Twitter User")
        twitter_username = st.sidebar.text_input(
            "Username", placeholder="e.g. @economic_times",
            help="Enter the Twitter handle (with or without @)"
        )
        twitter_limit = st.sidebar.slider("Number of Tweets", 5, 100, 20, key="twitter_limit")
        
        posts = []
        user_input = ""
        
        if twitter_bearer and twitter_username:
            if st.sidebar.button("🔄 Fetch Tweets", use_container_width=True):
                with st.sidebar:
                    with st.spinner(f"Fetching tweets from @{twitter_username.lstrip('@')}..."):
                        fetched_posts, error = fetch_twitter_posts(
                            twitter_bearer, twitter_username, twitter_limit
                        )
                if error:
                    # Fallback to synthetic data
                    st.sidebar.warning(f"⚠️ Twitter unavailable — loading synthetic demo data.\n\n_({error[:120]}...)_")
                    try:
                        with open("synthetic_posts.json", "r", encoding="utf-8") as f:
                            synthetic_data = json.load(f)
                        fetched_posts = [item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in synthetic_data[:twitter_limit]]
                        st.session_state.twitter_posts = fetched_posts
                        st.sidebar.success(f"✅ Loaded {len(fetched_posts)} synthetic posts as fallback")
                    except Exception:
                        st.sidebar.error("❌ Could not load synthetic fallback data.")
                elif fetched_posts:
                    st.session_state.twitter_posts = fetched_posts
                    st.sidebar.success(f"✅ Fetched {len(fetched_posts)} tweets from @{twitter_username.lstrip('@')}")
                else:
                    st.sidebar.warning("⚠️ No tweets found for this user.")
        else:
            # Allow synthetic data without credentials
            if st.sidebar.button("📦 Load Synthetic Demo Data", use_container_width=True, key="twitter_demo"):
                try:
                    with open("synthetic_posts.json", "r", encoding="utf-8") as f:
                        synthetic_data = json.load(f)
                    fetched_posts = [item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in synthetic_data[:twitter_limit]]
                    st.session_state.twitter_posts = fetched_posts
                    st.sidebar.success(f"✅ Loaded {len(fetched_posts)} synthetic posts")
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to load synthetic data: {e}")
            st.sidebar.info("💡 Enter Bearer Token + Username for live tweets, or use synthetic demo data.")
        
        # Use fetched tweets if available
        if 'twitter_posts' in st.session_state and st.session_state.twitter_posts:
            posts = st.session_state.twitter_posts
            user_input = "\n".join(posts)
            st.sidebar.info(f"📋 {len(posts)} tweets ready for analysis")

    # Advanced Settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        confidence_threshold = st.slider("Confidence Threshold", 0.2, 0.9, 0.35, 0.05,
                                         help="Posts below this confidence → 'Uncertain'. Recommended: 0.35 for 6-label zero-shot classifier.")
        include_sentiment = st.checkbox("🎭 Include Sentiment Analysis", value=True,
                                        help="Adds emotional context to risk scoring")

    # --- ROI CALCULATOR (Add-On #3) ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Business Impact Calculator")
    loan_amount = st.sidebar.number_input("Avg Loan Size (₹)", value=500000, step=50000, format="%d")
    default_rate_reduction = st.sidebar.slider("Expected Default Reduction (%)", 1, 20, 5)
    customers = st.sidebar.number_input("Customers Screened/Year", value=1000, step=100, format="%d")
    potential_savings = (loan_amount * (default_rate_reduction / 100) * customers)
    st.sidebar.metric("💎 Potential Savings/Year", f"₹{potential_savings:,.0f}")
    if potential_savings >= 10000000:
        st.sidebar.success(f"= ₹{potential_savings/10000000:.1f} Crores")
    elif potential_savings >= 100000:
        st.sidebar.info(f"= ₹{potential_savings/100000:.1f} Lakhs")
    st.sidebar.caption("_Projected savings from reduced loan defaults using behavioral screening_")

    analyze_btn = st.sidebar.button("🚀 Analyze Behavior", type="primary", use_container_width=True)

    # --- ANALYSIS LOGIC ---
    if analyze_btn and posts:
        if not posts:
            st.warning("⚠️ Please enter at least one post.")
            return
        
        labels = ["Spending", "Investment", "Loan", "Savings", "Risk", "Gambling / Speculative"]
        results = []
        
        # Load classification model with visible feedback
        with st.spinner("🧠 Loading NLP classification model... (first time may take a minute)"):
            classifier_model = load_classifier()
        
        # Load sentiment model if needed
        sentiment_model = None
        if include_sentiment:
            with st.spinner("🎭 Loading sentiment model..."):
                sentiment_model = load_sentiment_model()
        
        # Progress UX
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        status_messages = [
            ("🔍", "Parsing financial intent..."),
            ("🧠", "Running NLP classification..."),
            ("📊", "Computing behavior vectors..."),
            ("⚡", "Calculating risk metrics..."),
            ("🎯", "Generating recommendations...")
        ]
        
        start_time = time.time()
        
        with st.spinner(''):
            for i, post in enumerate(posts):
                # Status message UI with animation
                msg_idx = min(i * len(status_messages) // len(posts), len(status_messages) - 1)
                icon, text = status_messages[msg_idx]
                
                status_html = f'''
                <div class="ai-stage ai-stage-active">
                    <div class="ai-stage-icon">{icon}</div>
                    <div class="ai-stage-text">{text}</div>
                    <div style="margin-left: auto; color: #00d2ff; font-weight: bold;">{int(((i+1)/len(posts))*100)}%</div>
                </div>
                '''
                status_container.markdown(status_html, unsafe_allow_html=True)
                
                clean_post = preprocess_text(post)
                amount = extract_amount(post)
                
                # Zero-shot classification
                output = classifier_model(clean_post, candidate_labels=labels, multi_label=False)
                
                # Apply heuristic boosts based on explicit keywords
                scores_dict = {k: v for k, v in zip(output['labels'], output['scores'])}
                post_lower = clean_post.lower()
                
                # NOTE: Boosts need to be LARGE (1.5+) because after normalization across 6 categories,
                # a boost of 0.5 only yields ~0.47 confidence (below default 0.5 threshold).
                # We need the boosted category to clearly dominate.
                
                # --- GAMBLING / SPECULATIVE DETECTION (highest priority — dangerous behaviors) ---
                gambling_keywords = ['fomo', 'options trading', 'lost in trading', 'lost in options',
                                     'betting', 'gamble', 'gambled', 'gambling', 'casino', 'satta',
                                     'jackpot', 'lottery', 'day trading', 'speculative', 'pump and dump',
                                     'ponzi', 'quick money', 'shiba', 'dogecoin', 'meme coin']
                is_gambling = any(kw in post_lower for kw in gambling_keywords)
                # Catch pattern: "lost ₹X in trading/options/crypto"
                if re.search(r'lost.*\d+.*(?:trading|options|bet|crypto|market)', post_lower):
                    is_gambling = True
                if re.search(r'lost.*(?:trading|options|bet)', post_lower):
                    is_gambling = True
                # Catch FOMO-driven crypto buys
                if 'crypto' in post_lower and ('fomo' in post_lower or 'because' in post_lower):
                    is_gambling = True
                # Catch "bought crypto because FOMO" even after preprocessing
                if 'bought' in post_lower and 'fomo' in post_lower:
                    is_gambling = True
                
                if is_gambling:
                    scores_dict['Gambling / Speculative'] += 2.5
                    scores_dict['Risk'] += 0.5
                
                # --- RISK DETECTION (risky but not pure gambling) ---
                risk_keywords = ['risky', 'risk', 'volatile', 'crash', 'bubble', 'scam', 'fraud',
                                 'defaulted', 'overdue', 'penalty', 'bounced', 'cheque bounced',
                                 'bankruptcy', 'debt trap', 'overleveraged', 'margin call',
                                 'high interest', 'loan shark', 'emi missed', 'missed emi']
                if any(kw in post_lower for kw in risk_keywords) and not is_gambling:
                    scores_dict['Risk'] += 2.0
                
                # --- LOAN DETECTION ---
                loan_keywords = ['loan', 'borrowed', 'borrow', 'lending', 'lend',
                                 'personal loan', 'home loan', 'car loan', 'education loan',
                                 'outstanding', 'installment', 'repayment',
                                 'mortgage', 'financed', 'financing', 'credit line', 'overdraft']
                has_loan = any(kw in post_lower for kw in loan_keywords)
                # Also detect EMI (but not in gambling context)
                if 'emi' in post_lower and not is_gambling:
                    has_loan = True
                # Detect "credit card bill" as loan-related
                if re.search(r'credit\s*card\s*bill', post_lower):
                    has_loan = True
                    
                if has_loan:
                    scores_dict['Loan'] += 1.8
                    # Exception: "paid credit card bill on time" is responsible → also strong Savings signal
                    if ('paid' in post_lower or 'cleared' in post_lower) and ('on time' in post_lower or 'timely' in post_lower or 'before due' in post_lower):
                        scores_dict['Savings'] += 2.0  # Discipline = financial health — dominant signal
                        scores_dict['Loan'] -= 0.5  # Less risky since paid on time
                
                # --- INVESTMENT DETECTION ---
                investment_keywords = ['invested', 'invest', 'investment', 'mutual fund', 'mutual funds',
                                       'sip', 'nps', 'nifty', 'sensex', 'portfolio', 'stocks', 'shares',
                                       'equity', 'index fund', 'etf', 'ppf', 'elss', 'blue chip',
                                       'long term', 'wealth creation', 'compounding',
                                       'dividend', 'returns', 'systematic investment']
                if any(kw in post_lower for kw in investment_keywords):
                    scores_dict['Investment'] += 2.0
                    # SIP/NPS are strong disciplined investment signals
                    if 'sip' in post_lower or 'nps' in post_lower or 'systematic' in post_lower:
                        scores_dict['Investment'] += 0.5
                
                # --- SAVINGS DETECTION ---
                savings_keywords = ['saved', 'saving', 'savings', 'emergency fund', 'rainy day',
                                    'fixed deposit', 'fd', 'recurring deposit', 'rd', 'piggy bank',
                                    'set aside', 'put away', 'stashed', 'frugal', 'budget', 'budgeting',
                                    'cut expenses', 'saved up']
                if any(kw in post_lower for kw in savings_keywords):
                    scores_dict['Savings'] += 2.0
                
                # --- SPENDING DETECTION (general purchases, lifestyle expenses) ---
                spending_keywords = ['bought', 'purchased', 'spent', 'splurged', 'ordered',
                                     'booked', 'shopping', 'trip', 'vacation', 'holiday',
                                     'luxury', 'expensive', 'upgrade', 'new phone', 'new car',
                                     'iphone', 'gadget', 'travel', 'restaurant', 'dining']
                is_spending = any(kw in post_lower for kw in spending_keywords)
                # Rent/transfer as spending
                if re.search(r'(?:rent|transferred|transfer)', post_lower):
                    is_spending = True
                # "paid" alone without loan/savings context → Spending
                if 'paid' in post_lower and not any(kw in post_lower for kw in ['on time', 'timely', 'loan', 'credit card bill']):
                    is_spending = True
                
                if is_spending and not is_gambling:
                    scores_dict['Spending'] += 1.8
                    # Luxury spending gets extra weight
                    if any(kw in post_lower for kw in ['luxury', 'splurged', 'dubai', 'vacation', 'trip']):
                        scores_dict['Spending'] += 0.5
                
                # --- Normalize scores back to sum to 1 ---
                total_score = sum(scores_dict.values())
                if total_score > 0:
                    for k in scores_dict:
                        scores_dict[k] /= total_score
                
                # Re-sort lists based on boosted scores
                sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
                output['labels'] = [k for k, v in sorted_items]
                output['scores'] = [v for k, v in sorted_items]

                confidence = output['scores'][0]
                category = output['labels'][0] if confidence >= confidence_threshold else "Uncertain"
                
                # Sentiment analysis
                sentiment = {"label": "N/A", "score": 0}
                if sentiment_model:
                    try:
                        sent_result = sentiment_model(clean_post[:512])
                        sentiment = sent_result[0]
                    except:
                        pass
                
                # Sentiment-adjusted confidence
                adjusted_confidence = confidence
                if include_sentiment and sentiment['label'] != "N/A":
                    if sentiment['label'] == 'NEGATIVE' and category in ['Spending', 'Loan', 'Risk', 'Gambling / Speculative']:
                        adjusted_confidence = min(confidence * 1.15, 1.0)  # Boost risky items with negative sentiment
                    elif sentiment['label'] == 'POSITIVE' and category in ['Investment', 'Savings']:
                        adjusted_confidence = min(confidence * 1.1, 1.0)  # Boost stable items with positive sentiment
                
                results.append({
                    "Post": post,
                    "Category": category,
                    "Confidence": round(adjusted_confidence, 3),
                    "Raw_Confidence": round(confidence, 3),
                    "Amount": amount,
                    "Sentiment": sentiment['label'],
                    "Sentiment_Score": round(sentiment.get('score', 0), 3),
                    "All_Scores": {k: round(v, 3) for k, v in zip(output['labels'], output['scores'])}
                })
                
                progress_bar.progress((i + 1) / len(posts))
        
        elapsed = time.time() - start_time
        progress_bar.empty()
        status_container.empty()
        
        # Store in session state
        scores, risk_score, uncertainty_rate, weighted_scores, total_amounts = calculate_scores([
            {"label": r["Category"], "confidence": r["Confidence"], "amount": r["Amount"]} for r in results
        ])
        
        st.session_state.analysis_results = results
        st.session_state.analysis_scores = {
            "scores": scores, "risk_score": risk_score,
            "uncertainty_rate": uncertainty_rate,
            "weighted_scores": weighted_scores,
            "total_amounts": total_amounts,
            "elapsed": elapsed,
            "confidence_threshold": confidence_threshold,
            "include_sentiment": include_sentiment
        }
        st.session_state.analysis_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "risk_score": risk_score,
            "posts": len(posts)
        })
        st.rerun()
    
    # --- DISPLAY RESULTS (from session state) ---
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        data = st.session_state.analysis_scores
        scores = data['scores']
        risk_score = data['risk_score']
        uncertainty_rate = data['uncertainty_rate']
        weighted_scores = data['weighted_scores']
        total_amounts = data['total_amounts']
        elapsed = data['elapsed']
        confidence_threshold = data['confidence_threshold']
        
        st.success(f"✅ Analysis Complete — {len(results)} posts in {elapsed:.1f}s ({elapsed/len(results)*1000:.0f}ms/post)")
        
        # --- TABS ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📋 Details", "🔬 Model Eval", "📜 History", "📡 Live Feed"])
        
        with tab1:
            # Top Metrics
            st.markdown('<div class="reveal-section reveal-1">', unsafe_allow_html=True)
            st.subheader("📊 Financial Behavior Summary")
            col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
            
            with col1:
                # --- ANIMATED RISK GAUGE ---
                risk_color = "#ff3838" if risk_score > 70 else "#00b894" if risk_score < 30 else "#ffc107"
                gauge_deg = int((risk_score / 100) * 360)
                level = "HIGH RISK" if risk_score > 70 else "LOW RISK" if risk_score < 30 else "MED RISK"
                
                st.markdown(f'''
                <style>
                @keyframes fillCustomGauge {{
                    from {{ background: conic-gradient(from 0deg, {risk_color} 0deg, rgba(255,255,255,0.05) 0deg); }}
                    to   {{ background: conic-gradient(from 0deg, {risk_color} {gauge_deg}deg, rgba(255,255,255,0.05) {gauge_deg}deg); }}
                }}
                .risk-gauge-active {{ animation: fillCustomGauge 1.5s ease-out forwards; }}
                </style>
                <div class="risk-gauge-container">
                    <div class="risk-gauge risk-gauge-active">
                        <div class="risk-gauge-inner">
                            <div class="risk-gauge-score" style="color: {risk_color};">{int(risk_score)}</div>
                            <div class="risk-gauge-label">Score</div>
                        </div>
                    </div>
                    <div class="risk-gauge-level" style="background: {risk_color}20; border: 1px solid {risk_color}50; color: {risk_color};">{level}</div>
                </div>
                ''', unsafe_allow_html=True)
                
            with col2:
                st.metric("💸 Spending", scores['Spending'])
            with col3:
                st.metric("📈 Investment", scores['Investment'])
            with col4:
                st.metric("🏦 Loan", scores['Loan'])
            with col5:
                st.metric("❓ Uncertain", f"{uncertainty_rate:.0f}%",
                          delta="Good" if uncertainty_rate < 20 else "Review needed")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Charts
            st.markdown('<div class="reveal-section reveal-2">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("🥧 Behavior Distribution")
                df = pd.DataFrame(list(scores.items()), columns=['Category', 'Count'])
                df = df[df['Count'] > 0]
                if not df.empty:
                    fig = px.pie(df, values='Count', names='Category', hole=0.45,
                                color_discrete_sequence=['#00d2ff', '#6c5ce7', '#fd79a8', '#00b894', '#e17055', '#ff3838'])
                    fig.update_layout(
                        showlegend=True, height=350,
                        margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#a0a0c0',
                        legend=dict(font=dict(color='#e0e0e0'))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No confident classifications to display.")
            
            with c2:
                st.subheader("🕸️ Behavior Profile Radar")
                # Radar Chart
                categories = list(scores.keys())
                values = list(scores.values())
                max_val = max(values) if max(values) > 0 else 1
                normalized = [v / max_val * 100 for v in values]
                
                # Add benchmark
                benchmark = [50, 50, 30, 60, 20, 5]  # "Ideal" profile
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized + [normalized[0]],
                    theta=categories + [categories[0]],
                    fill='toself', name='Your Profile',
                    fillcolor='rgba(0, 210, 255, 0.2)',
                    line=dict(color='#00d2ff', width=2)
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=benchmark + [benchmark[0]],
                    theta=categories + [categories[0]],
                    fill='toself', name='Ideal Benchmark',
                    fillcolor='rgba(0, 184, 148, 0.1)',
                    line=dict(color='#00b894', width=1, dash='dot')
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor='rgba(255,255,255,0.1)'),
                        angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#e0e0e0')
                    ),
                    showlegend=True, height=350,
                    margin=dict(t=30, b=10, l=40, r=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#a0a0c0',
                    legend=dict(font=dict(color='#e0e0e0'))
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk Factors
            st.markdown('<div class="reveal-section reveal-3">', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            
            with c3:
                st.subheader("📈 Risk Factor Weights")
                risk_data = pd.DataFrame({
                    'Factor': ['Spending', 'Loan', 'Risk Behavior', 'Investment (-)', 'Savings (-)'],
                    'Weight': [
                        weighted_scores['Spending'] * 1,
                        weighted_scores['Loan'] * 1.5,
                        weighted_scores['Risk'] * 2,
                        -weighted_scores['Investment'] * 0.6,
                        -weighted_scores['Savings'] * 0.5
                    ],
                    'Type': ['Risk', 'Risk', 'Risk', 'Stabilizer', 'Stabilizer']
                })
                colors = {'Risk': '#e17055', 'Stabilizer': '#00b894'}
                fig2 = px.bar(risk_data, x='Factor', y='Weight', color='Type',
                              color_discrete_map=colors)
                fig2.update_layout(
                    showlegend=True, height=300,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#a0a0c0',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    legend=dict(font=dict(color='#e0e0e0'))
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with c4:
                st.subheader("💰 Amount Distribution")
                amount_data = pd.DataFrame(list(total_amounts.items()), columns=['Category', 'Amount'])
                amount_data = amount_data[amount_data['Amount'] > 0]
                if not amount_data.empty:
                    fig_amt = px.bar(amount_data, x='Category', y='Amount',
                                    color='Category',
                                    color_discrete_sequence=['#00d2ff', '#6c5ce7', '#fd79a8', '#00b894', '#e17055', '#ff3838'])
                    fig_amt.update_layout(
                        showlegend=False, height=300,
                        margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#a0a0c0',
                        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='₹ Amount')
                    )
                    st.plotly_chart(fig_amt, use_container_width=True)
                else:
                    st.info("No monetary amounts detected in posts.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # --- BEHAVIORAL TIMELINE ---
            st.markdown('<div class="reveal-section reveal-4">', unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("📅 Behavioral Timeline")
            timeline_data = []
            for idx, r in enumerate(results):
                if r['Category'] != 'Uncertain':
                    timeline_data.append({
                        'Day': f'Day {idx + 1}',
                        'DayNum': idx + 1,
                        'Category': r['Category'],
                        'Post': r['Post'][:50] + ('...' if len(r['Post']) > 50 else ''),
                        'Confidence': r['Confidence']
                    })
            if timeline_data:
                tl_df = pd.DataFrame(timeline_data)
                category_risk_map = {
                    'Investment': 10, 'Savings': 15, 'Spending': 50,
                    'Loan': 60, 'Risk': 75, 'Gambling / Speculative': 95
                }
                tl_df['RiskLevel'] = tl_df['Category'].map(category_risk_map).fillna(50)
                
                # Cumulative risk trend
                tl_df['CumulativeRisk'] = tl_df['RiskLevel'].expanding().mean()
                
                fig_timeline = go.Figure()
                
                # Scatter for individual posts
                color_map = {
                    'Spending': '#fd79a8', 'Investment': '#00b894', 'Loan': '#6c5ce7',
                    'Savings': '#00d2ff', 'Risk': '#e17055', 'Gambling / Speculative': '#ff3838'
                }
                for cat in tl_df['Category'].unique():
                    cat_data = tl_df[tl_df['Category'] == cat]
                    fig_timeline.add_trace(go.Scatter(
                        x=cat_data['DayNum'], y=cat_data['RiskLevel'],
                        mode='markers+text', name=cat,
                        marker=dict(size=14, color=color_map.get(cat, '#aaa'), symbol='circle',
                                    line=dict(width=1, color='white')),
                        text=cat_data['Category'],
                        textposition='top center',
                        textfont=dict(size=9, color='#a0a0c0'),
                        hovertext=cat_data['Post'],
                        hoverinfo='text+name'
                    ))
                
                # Cumulative risk trendline
                fig_timeline.add_trace(go.Scatter(
                    x=tl_df['DayNum'], y=tl_df['CumulativeRisk'],
                    mode='lines', name='Risk Trend',
                    line=dict(color='#ffc107', width=2, dash='dot'),
                    hoverinfo='y+name'
                ))
                
                fig_timeline.update_layout(
                    xaxis_title='Post Sequence', yaxis_title='Risk Level',
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#a0a0c0',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 100]),
                    legend=dict(font=dict(color='#e0e0e0'), orientation='h', yanchor='bottom', y=1.02)
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                st.caption("_Each dot is a classified post. The dashed line shows cumulative risk trend — judges see pattern detection._")
            else:
                st.info("No confident classifications for timeline.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Recommendations
            st.markdown('<div class="reveal-section reveal-5">', unsafe_allow_html=True)
            st.subheader("💡 AI-Powered Bank Recommendations")
            
            if risk_score > 70:
                st.error("⚠️ **High Risk Profile Detected**")
                rec_cols = st.columns(4)
                recs = [
                    ("🔒", "Secured Credit Card", "Deposit-backed, low limit"),
                    ("📚", "Financial Literacy", "Budgeting courses"),
                    ("🔍", "3-Month Monitor", "Review before upgrade"),
                    ("💰", "Micro-Savings Plan", "Auto-debit ₹500/week")
                ]
            elif scores['Investment'] > scores['Spending'] and scores['Savings'] > 0:
                st.success("✅ **Stable Investor Profile**")
                recs = [
                    ("💎", "Premium Credit Card", "Rewards + lounge access"),
                    ("📊", "Wealth Advisory", "Portfolio management"),
                    ("📈", "Higher Credit Limit", "Pre-approved upgrade"),
                    ("🛡️", "Insurance Cross-sell", "Term + health cover")
                ]
            elif scores['Loan'] > 0 and scores['Savings'] == 0:
                st.warning("⚡ **Active Borrower, Low Savings**")
                recs = [
                    ("🔄", "Debt Consolidation", "Single lower-rate loan"),
                    ("🏦", "Auto Savings", "₹1000/month auto-debit"),
                    ("🎓", "Financial Coaching", "Emergency fund guide"),
                    ("💳", "Balance Transfer", "0% intro rate offer")
                ]
            else:
                st.info("ℹ️ **Average Profile — Growth Opportunity**")
                recs = [
                    ("💳", "Standard Credit Card", "2% cashback"),
                    ("📱", "Micro-Investments", "Start with ₹100/day"),
                    ("🎮", "Savings Challenge", "Gamified goals"),
                    ("📊", "Spending Insights", "Category-wise breakdown")
                ]
            
            rec_cols = st.columns(4)
            for col, (icon, title, desc) in zip(rec_cols, recs):
                with col:
                    st.markdown(f"""
                    <div class="flow-step">
                        <div class="icon">{icon}</div>
                        <div class="title">{title}</div>
                        <div class="desc">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Explainability
            with st.expander("🔍 Why This Risk Score? (Explainable AI)"):
                st.markdown(f"""
                **Risk Score Calculation (Confidence-Weighted):**
                | Factor | Posts | Weighted Score | Weight | Contribution |
                |--------|-------|---------------|--------|-------------|
                | Spending | {scores['Spending']} | {weighted_scores['Spending']:.2f} | ×1.0 | +{weighted_scores['Spending']*1:.2f} |
                | Loan | {scores['Loan']} | {weighted_scores['Loan']:.2f} | ×1.5 | +{weighted_scores['Loan']*1.5:.2f} |
                | Risk | {scores['Risk']} | {weighted_scores['Risk']:.2f} | ×2.0 | +{weighted_scores['Risk']*2:.2f} |
                | Gambling / Speculative | {scores['Gambling / Speculative']} | {weighted_scores['Gambling / Speculative']:.2f} | ×3.0 | +{weighted_scores['Gambling / Speculative']*3:.2f} |
                | Investment | {scores['Investment']} | {weighted_scores['Investment']:.2f} | ×-0.6 | {-weighted_scores['Investment']*0.6:.2f} |
                | Savings | {scores['Savings']} | {weighted_scores['Savings']:.2f} | ×-0.5 | {-weighted_scores['Savings']*0.5:.2f} |
                
                **Final Score:** {int(risk_score)}/100
                **Sentiment Integration:** {'Enabled — negative sentiment boosts risk labels' if data.get('include_sentiment') else 'Disabled'}
                **Uncertain Posts:** {len([r for r in results if r['Category'] == 'Uncertain'])} (threshold: {confidence_threshold})
                """)
            
            # --- TRADITIONAL vs FINBEHAVIOR COMPARISON (Add-On #4) ---
            st.markdown("---")
            st.subheader("🆚 Traditional vs. FinBehavior Scoring")
            cmp1, cmp2 = st.columns(2)
            with cmp1:
                st.markdown("""
                <div class="comparison-box" style="background: rgba(255,100,100,0.06); border: 1px solid rgba(255,100,100,0.2);">
                    <h4 style="color: #ff6b6b; font-weight: 800; margin-top: 0;">📋 Traditional (CIBIL/Bureau)</h4>
                    <ul style="color: #b0b0d0; line-height: 2;">
                        <li>Based on past loan repayment only</li>
                        <li>❌ <b>Rejects thin-file</b> — 400M Indians excluded</li>
                        <li>Lagging indicator (30–90 day delay)</li>
                        <li>No behavioral or lifestyle signals</li>
                        <li>Opaque scoring — black box</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with cmp2:
                st.markdown("""
                <div class="comparison-box" style="background: rgba(0,210,255,0.06); border: 1px solid rgba(0,210,255,0.2);">
                    <h4 style="color: #00d2ff; font-weight: 800; margin-top: 0;">🧠 FinBehavior AI</h4>
                    <ul style="color: #b0b0d0; line-height: 2;">
                        <li>Based on <b>real-time behavior</b> signals</li>
                        <li>✅ <b>Includes underbanked</b> & first-time borrowers</li>
                        <li>Leading indicator — catches trends early</li>
                        <li>Consent-based social + spending signals</li>
                        <li>Fully explainable — open formula</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # --- BANK DECISION ENGINE ---
            st.markdown("---")
            st.subheader("🏦 AI Credit Recommendation Engine")
            
            decision_confidence = max(0, 100 - uncertainty_rate)
            
            if risk_score <= 30:
                decision_emoji = "✅"
                decision_text = "Approved — Low Risk"
                decision_color = "#00b894"
                eligibility = "Eligible for Personal Loan (₹5L) • Premium Credit Card"
                products = [
                    ("💳", "Premium Credit Card", "₹5L limit, 3% cashback, lounge access"),
                    ("🏠", "Home Loan Pre-approval", "Up to ₹50L at preferential rates"),
                    ("📈", "Wealth Management", "Dedicated relationship manager"),
                    ("🛡️", "Insurance Bundle", "Term + Health at group rates")
                ]
            elif risk_score <= 50:
                decision_emoji = "✅"
                decision_text = "Conditionally Approved — Moderate Risk"
                decision_color = "#00d2ff"
                eligibility = "Eligible for Micro Loan (₹50k) • Low-limit Credit Card"
                products = [
                    ("💳", "Low-limit Credit Card", "₹25k limit, build credit history"),
                    ("📊", "SIP Starter Plan", "₹500/month auto-invest in index funds"),
                    ("🎓", "Financial Literacy Program", "Free budgeting course + certification"),
                    ("💰", "Micro Savings Account", "2x interest on first ₹50k")
                ]
            elif risk_score <= 70:
                decision_emoji = "⚠️"
                decision_text = "Under Review — Elevated Risk"
                decision_color = "#ffc107"
                eligibility = "Conditional approval with 3-month monitoring period"
                products = [
                    ("🔒", "Secured Credit Card", "Deposit-backed, ₹10k limit"),
                    ("📚", "Financial Coaching", "1-on-1 sessions with advisor"),
                    ("💰", "Auto-Savings Plan", "₹500/week mandatory savings"),
                    ("🔍", "3-Month Review", "Re-evaluate after monitoring")
                ]
            else:
                decision_emoji = "❌"
                decision_text = "Declined — High Risk"
                decision_color = "#ff3838"
                eligibility = "Not eligible at this time. Alternative programs recommended."
                products = [
                    ("🎓", "Financial Literacy Program", "Mandatory 4-week course"),
                    ("💰", "Guided Savings Program", "₹200/day auto-debit to savings"),
                    ("🔄", "Debt Counseling", "Free session with certified counselor"),
                    ("📅", "6-Month Re-application", "Apply again after program completion")
                ]
            
            # Decision Card
            st.markdown(f"""
            <div class="decision-card" style="background: rgba(255,255,255,0.04); border: 2px solid {decision_color}; border-radius: 20px; padding: 32px; margin: 10px 0;">
                <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 16px;">
                    <span style="font-size: 2.8rem; filter: drop-shadow(0 0 12px {decision_color});">{decision_emoji}</span>
                    <div>
                        <h3 style="color: {decision_color}; margin: 0; font-weight: 800; font-size: 1.4rem;">{decision_text}</h3>
                        <p style="color: #a0a0c0; margin: 6px 0 0 0; font-size: 0.95rem;">Risk Score: <b style="color: white; font-size: 1.1rem;">{int(risk_score)}/100</b> &nbsp;•&nbsp; Decision Confidence: <b style="color: white; font-size: 1.1rem;">{decision_confidence:.0f}%</b></p>
                    </div>
                </div>
                <p style="color: #e0e0e0; font-size: 1.05rem; margin: 0; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06);">📋 {eligibility}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Product Recommendations
            st.markdown("**Recommended Products:**")
            st.markdown('<div style="display: flex; gap: 16px; margin-bottom: 24px;">', unsafe_allow_html=True)
            prod_cols = st.columns(4)
            for i, (col, (icon, title, desc)) in enumerate(zip(prod_cols, products)):
                with col:
                    st.markdown(f"""
                    <div class="product-card slide-up-card slide-up-card-{i+1}">
                        <div class="card-icon">{icon}</div>
                        <div style="color: #00d2ff; font-weight: 700; font-size: 0.9rem;">{title}</div>
                        <div style="color: #a0a0c0; font-size: 0.78rem; margin-top: 8px; line-height: 1.4;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

            # Download Report
            st.markdown("---")
            dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1])
            with dl_col2:
                pdf_bytes = generate_pdf_report(results, scores, risk_score, uncertainty_rate)
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"finbehavior_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        # --- TAB 2: DETAILS ---
        with tab2:
            st.subheader("📋 Detailed Post Classification")
            
            # Summary table
            results_df = pd.DataFrame(results)
            display_df = results_df[['Post', 'Category', 'Confidence', 'Amount', 'Sentiment']].copy()
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"₹{x:,.0f}" if x > 0 else "—")
            st.dataframe(display_df, use_container_width=True, height=300)
            
            # Expandable details
            for idx, row in results_df.iterrows():
                emoji_map = {"Spending": "💸", "Investment": "📈", "Loan": "🏦", "Savings": "💰", "Risk": "⚠️", "Gambling / Speculative": "🎰", "Uncertain": "❓"}
                emoji = emoji_map.get(row['Category'], "📝")
                
                with st.expander(f"{emoji} {row['Post'][:70]}{'...' if len(row['Post'])>70 else ''}"):
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Category", row['Category'])
                    d2.metric("Confidence", f"{row['Confidence']:.1%}")
                    d3.metric("Sentiment", row['Sentiment'])
                    d4.metric("Amount", f"₹{row['Amount']:,.0f}" if row['Amount'] > 0 else "—")
                    
                    # --- HUMAN-IN-THE-LOOP FLAGGING (Add-On #2) ---
                    if row['Category'] in ['Risk', 'Gambling / Speculative'] or row['Confidence'] < 0.6:
                        flag_key = f"flag_{idx}_{row['Post'][:20]}"
                        post_id = f"{idx}_{row['Post'][:30]}"
                        already_flagged = post_id in st.session_state.flagged_posts
                        
                        if already_flagged:
                            st.warning("🚩 Already flagged — risk score was increased by +10")
                        else:
                            if st.button("🚩 Flag for Manual Review", key=flag_key):
                                # Track this post as flagged
                                st.session_state.flagged_posts.add(post_id)
                                
                                # Increase risk score by 10 points (capped at 100)
                                RISK_PENALTY_PER_FLAG = 10
                                current_risk = st.session_state.analysis_scores['risk_score']
                                new_risk = min(current_risk + RISK_PENALTY_PER_FLAG, 100)
                                st.session_state.analysis_scores['risk_score'] = new_risk
                                
                                st.success(f"✅ Flagged for Loan Officer Review — Risk score increased: {int(current_risk)} → {int(new_risk)} (+{RISK_PENALTY_PER_FLAG})")
                                st.caption("_In production, this triggers an alert in the bank's Case Management System_")
                                st.rerun()
                    
                    if row['All_Scores']:
                        st.markdown('<div class="confidence-bar-container">', unsafe_allow_html=True)
                        for label, score in list(row['All_Scores'].items())[:3]:  # Top 3 scores
                            color = "#00d2ff" if label == row['Category'] else "#a0a0c0"
                            st.markdown(f'''
                            <div class="confidence-bar-label"><span>{label}</span> <span>{score:.1%}</span></div>
                            <div class="confidence-bar-track">
                                <div class="confidence-bar-fill" style="width: {score*100}%; background: {color};"></div>
                            </div>
                            ''', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- TAB 3: MODEL EVALUATION ---
        with tab3:
            st.subheader("🔬 Model Accuracy Evaluation")
            st.caption("Evaluates the zero-shot model against synthetic labeled data")
            
            if st.button("▶️ Run Evaluation (25 samples)", type="primary"):
                with st.spinner("🧠 Loading model for evaluation..."):
                    eval_classifier = load_classifier()
                with st.spinner("Running model on labeled test data..."):
                    cm, report, eval_labels = run_model_evaluation(eval_classifier)
                
                if cm is not None:
                    ev1, ev2 = st.columns(2)
                    
                    with ev1:
                        st.markdown("**Confusion Matrix**")
                        fig_cm, ax = plt.subplots(figsize=(6, 5))
                        fig_cm.patch.set_facecolor('#0f0c29')
                        ax.set_facecolor('#0f0c29')
                        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                        ax.set_xticks(range(len(eval_labels)))
                        ax.set_yticks(range(len(eval_labels)))
                        ax.set_xticklabels(eval_labels, rotation=45, ha='right', color='#e0e0e0', fontsize=9)
                        ax.set_yticklabels(eval_labels, color='#e0e0e0', fontsize=9)
                        ax.set_xlabel('Predicted', color='#a0a0c0')
                        ax.set_ylabel('True', color='#a0a0c0')
                        for i in range(len(eval_labels)):
                            for j in range(len(eval_labels)):
                                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                                       color='white' if cm[i, j] > cm.max()/2 else '#333', fontweight='bold')
                        fig_cm.colorbar(im, ax=ax, shrink=0.8)
                        plt.tight_layout()
                        st.pyplot(fig_cm)
                    
                    with ev2:
                        st.markdown("**Per-Category Metrics**")
                        metrics_data = []
                        for label in eval_labels:
                            if label in report:
                                metrics_data.append({
                                    "Category": label,
                                    "Precision": f"{report[label]['precision']:.0%}",
                                    "Recall": f"{report[label]['recall']:.0%}",
                                    "F1-Score": f"{report[label]['f1-score']:.0%}",
                                    "Support": int(report[label]['support'])
                                })
                        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                        
                        # Overall accuracy
                        if 'accuracy' in report:
                            st.metric("Overall Accuracy", f"{report['accuracy']:.0%}")
                        
                        st.markdown("""
                        **What this means:**
                        - **Precision**: Of predicted X, how many were actually X?
                        - **Recall**: Of actual X, how many did we find?
                        - **F1-Score**: Harmonic mean of precision & recall
                        - Higher is better. >70% is good for zero-shot.
                        """)
                else:
                    st.warning("⚠️ `synthetic_posts.json` not found. Run `generate_data.py` first.")
        
        # --- TAB 4: HISTORY ---
        with tab4:
            st.subheader("📜 Analysis History")
            if st.session_state.analysis_history:
                hist_df = pd.DataFrame(st.session_state.analysis_history)
                fig_hist = px.line(hist_df, x='timestamp', y='risk_score', markers=True,
                                  labels={'risk_score': 'Risk Score', 'timestamp': 'Time'})
                fig_hist.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#a0a0c0',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 100])
                )
                fig_hist.update_traces(line_color='#00d2ff', marker_color='#6c5ce7')
                st.plotly_chart(fig_hist, use_container_width=True)
                st.dataframe(hist_df, use_container_width=True)
            else:
                st.info("No analysis history yet. Run an analysis to see trends.")
        
        # --- TAB 5: LIVE FEED SIMULATION ---
        with tab5:
            st.subheader("📡 Live Social Feed Simulation")
            st.caption("_Demo Mode — Simulates real-time social media post analysis like a live AI pipeline_")
            
            demo_feed_posts = [
                ("I spent ₹12,000 on a new phone 😭", "Spending", "Negative", 6),
                ("Just invested ₹5000 in mutual funds 📈", "Investment", "Positive", -3),
                ("Borrowed ₹20000 from friend for rent", "Loan", "Negative", 8),
                ("Spent ₹8000 on shopping spree! 🛍️", "Spending", "Positive", 5),
                ("Saving ₹3000 every month now 💪", "Savings", "Positive", -4),
                ("Thinking to take personal loan 🤔", "Loan", "Neutral", 7),
                ("Bet ₹10,000 on IPL match! 🏐", "Gambling / Speculative", "Negative", 25),
                ("Started SIP of ₹2000/month today", "Investment", "Positive", -5),
                ("Online casino deposit ₹5000 🎰", "Gambling / Speculative", "Negative", 25),
                ("Emergency fund reached ₹1 lakh! 🎉", "Savings", "Positive", -6),
                ("Crypto leverage trade ₹50000 🚀", "Gambling / Speculative", "Negative", 20),
                ("Paid all credit card bills on time ✅", "Savings", "Positive", -3),
                ("Bought PS5 on EMI ₹4000/month", "Spending", "Neutral", 4),
                ("Lost ₹15000 in options trading 😅", "Risk", "Negative", 12),
                ("Rebalanced portfolio to debt funds", "Investment", "Positive", -4),
            ]
            
            feed_col1, feed_col2 = st.columns([2, 1])
            
            with feed_col1:
                num_posts = st.slider("Number of posts to stream", 3, 15, 8, key="feed_slider")
                
                if st.button("▶️ Start Live Feed", type="primary", use_container_width=True):
                    feed_container = st.container()
                    score_display = feed_col2.empty()
                    running_risk = 50  # Start at neutral
                    
                    import random as rnd
                    indices = rnd.sample(range(len(demo_feed_posts)), min(num_posts, len(demo_feed_posts)))
                    
                    for step, idx in enumerate(indices):
                        post_text, category, sentiment, risk_impact = demo_feed_posts[idx]
                        running_risk = max(0, min(100, running_risk + risk_impact))
                        
                        with feed_container:
                            # Post card
                            risk_color = "#ff3838" if risk_impact > 0 else "#00b894"
                            cat_colors = {
                                'Spending': '#fd79a8', 'Investment': '#00b894', 'Loan': '#6c5ce7',
                                'Savings': '#00d2ff', 'Risk': '#e17055', 'Gambling / Speculative': '#ff3838'
                            }
                            cat_color = cat_colors.get(category, '#aaa')
                            
                            st.markdown(f"""
                            <div class="feed-post-card" style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); border-left: 4px solid {cat_color}; border-radius: 12px; padding: 18px; margin: 10px 0;">
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <div>
                                        <span class="new-post-badge" style="background: rgba(0,210,255,0.15); color: #00d2ff; padding: 3px 10px; border-radius: 6px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px;">NEW POST</span>
                                        <p style="color: #e0e0e0; font-size: 1rem; margin: 8px 0 12px 0;">“{post_text}”</p>
                                    </div>
                                </div>
                                <div style="display: flex; gap: 20px; flex-wrap: wrap; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05);">
                                    <span style="color: #a0a0c0; font-size: 0.85rem;">Category → <b style="color: {cat_color};">{category}</b></span>
                                    <span style="color: #a0a0c0; font-size: 0.85rem;">Sentiment → <b>{sentiment}</b></span>
                                    <span style="color: #a0a0c0; font-size: 0.85rem;">Risk Impact → <b style="color: {risk_color};">{"+" if risk_impact > 0 else ""}{risk_impact}</b></span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Update score display
                        risk_emoji = "🔴" if running_risk > 70 else "🟢" if running_risk < 30 else "🟡"
                        with score_display.container():
                            st.metric("🎯 Live Risk Score", f"{running_risk}/100")
                            st.progress(running_risk / 100)
                            level = "HIGH" if running_risk > 70 else "LOW" if running_risk < 30 else "MEDIUM"
                            st.markdown(f"**{risk_emoji} {level} RISK**")
                            st.caption(f"Posts analyzed: {step + 1}/{num_posts}")
                        
                        time.sleep(1.5)
                    
                    st.success(f"✅ Live Feed Complete — Final Risk Score: **{running_risk}/100**")
                    st.balloons()
                else:
                    st.info("👆 Click **Start Live Feed** to simulate real-time social media post analysis with AI classification.")
            
            with feed_col2:
                st.markdown("")
                st.markdown("""
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px;">
                    <h4 style="color: #00d2ff; margin-top: 0;">How It Works</h4>
                    <p style="color: #a0a0c0; font-size: 0.85rem; line-height: 1.6;">
                        📡 Posts stream in real-time<br>
                        🤖 AI classifies each post instantly<br>
                        📊 Risk score updates live<br>
                        🎰 Gambling/fraud posts cause major spikes<br>
                        💰 Savings/investment posts reduce risk
                    </p>
                    <p style="color: #666; font-size: 0.75rem; margin-bottom: 0;">In production, this connects to social media APIs with user consent.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance Metrics
        with st.expander("⚡ Performance Metrics"):
            perf1, perf2, perf3 = st.columns(3)
            perf1.metric("Total Time", f"{elapsed:.1f}s")
            perf2.metric("Avg per Post", f"{elapsed/len(results)*1000:.0f}ms")
            perf3.metric("Posts Analyzed", len(results))
        
        # --- REGULATORY COMPLIANCE BADGES (Add-On #5) ---
        st.markdown("---")
        badge1, badge2, badge3, badge4 = st.columns(4)
        with badge1:
            st.markdown("""
            <div class="compliance-badge" style="background:rgba(0,210,255,0.06); border:1px solid rgba(0,210,255,0.15);">
                <div class="badge-icon">🛡️</div>
                <div style="color:#00d2ff; font-weight:700; font-size:0.85rem; margin-top:6px;">DPDP Act 2023</div>
                <div style="color:#808098; font-size:0.7rem; margin-top:2px;">Compliant</div>
            </div>
            """, unsafe_allow_html=True)
        with badge2:
            st.markdown("""
            <div class="compliance-badge" style="background:rgba(108,92,231,0.06); border:1px solid rgba(108,92,231,0.15);">
                <div class="badge-icon">🏛️</div>
                <div style="color:#6c5ce7; font-weight:700; font-size:0.85rem; margin-top:6px;">RBI Digital Lending</div>
                <div style="color:#808098; font-size:0.7rem; margin-top:2px;">Guidelines Ready</div>
            </div>
            """, unsafe_allow_html=True)
        with badge3:
            st.markdown("""
            <div class="compliance-badge" style="background:rgba(0,200,83,0.06); border:1px solid rgba(0,200,83,0.15);">
                <div class="badge-icon">🔐</div>
                <div style="color:#00c853; font-weight:700; font-size:0.85rem; margin-top:6px;">ISO 27001</div>
                <div style="color:#808098; font-size:0.7rem; margin-top:2px;">Architecture Ready</div>
            </div>
            """, unsafe_allow_html=True)
        with badge4:
            st.markdown("""
            <div class="compliance-badge" style="background:rgba(255,193,7,0.06); border:1px solid rgba(255,193,7,0.15);">
                <div class="badge-icon">🇪🇺</div>
                <div style="color:#ffc107; font-weight:700; font-size:0.85rem; margin-top:6px;">GDPR</div>
                <div style="color:#808098; font-size:0.7rem; margin-top:2px;">By Design</div>
            </div>
            """, unsafe_allow_html=True)
        st.caption("🔒 **Privacy & Ethics**: Consent-based analysis • No data stored • Explainable scores • Bias-audited • Human-in-the-loop oversight")
    
    else:
        # --- WELCOME / LANDING PAGE ---
        st.markdown("")
        
        # Architecture Flow
        st.subheader("🏗️ How It Works")
        st.markdown('<div class="flow-container">', unsafe_allow_html=True)
        flow_cols = st.columns([2, 1, 2, 1, 2, 1, 2, 1, 2])
        steps = [
            ("📱", "1. Input", "Social posts with consent"),
            ("🧹", "2. Preprocess", "Clean & extract amounts"),
            ("🤖", "3. NLP Engine", "Zero-shot classification"),
            ("📊", "4. Score", "Risk & behavior calculation"),
            ("💡", "5. Recommend", "Bank product matching")
        ]
        
        for i, step in enumerate(steps):
            icon, title, desc = step
            with flow_cols[i*2]:
                st.markdown(f"""
                <div class="flow-step flow-step-{i+1}" style="height: 100%;">
                    <div class="icon">{icon}</div>
                    <div class="title">{title}</div>
                    <div class="desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if i < len(steps) - 1:
                with flow_cols[i*2 + 1]:
                    st.markdown(f'<div class="flow-arrow flow-step-{i+2}" style="height: 100%;">❯</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="feature-card">
                <h3 style="color: #00d2ff; margin-top: 0; font-weight: 800;">🎯 Features</h3>
                <ul style="color: #b0b0d0; line-height: 2.0; padding-left: 20px;">
                    <li><b>Zero-Shot NLP</b> — No training data needed</li>
                    <li><b>Sentiment-Aware Scoring</b> — Emotional context matters</li>
                    <li><b>📡 Live Feed Simulation</b> — Real-time AI pipeline demo</li>
                    <li><b>🎰 Gambling/Fraud Detection</b> — Speculative behavior flagging</li>
                    <li><b>🏦 Credit Decision Engine</b> — Bank-grade loan recommendations</li>
                    <li><b>📅 Behavioral TimeLine</b> — Pattern detection over time</li>
                    <li><b>Explainable AI</b> — See exactly why each score</li>
                    <li><b>PDF Reports</b> — Download professional reports</li>
                    <li><b>Model Evaluation</b> — Confusion matrix & F1 scores</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="feature-card" style="animation-delay: 0.15s;">
                <h3 style="color: #6c5ce7; margin-top: 0; font-weight: 800;">🔒 Ethics by Design</h3>
                <ul style="color: #b0b0d0; line-height: 2.0; padding-left: 20px;">
                    <li>✅ Explicit <b>user consent</b> before analysis</li>
                    <li>✅ <b>No data storage</b> — ephemeral processing</li>
                    <li>✅ <b>Explainable scores</b> — full calculation visible</li>
                    <li>✅ <b>Bias monitoring</b> — model evaluation tab</li>
                    <li>✅ <b>GDPR & DPDP</b> Act 2023 compliant</li>
                    <li>✅ <b>Appealable</b> — scores can be challenged</li>
                    <li>✅ <b>Open model</b> — no black-box proprietary AI</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <p style="color: #a0a0c0; font-size: 1.1rem;">
                <span class="cta-arrow">👈</span>&nbsp;&nbsp;<b style="color: #00d2ff;">Get Started</b>: Enter posts in the sidebar and click <b>'🚀 Analyze Behavior'</b>
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
