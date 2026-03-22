import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import sys, copy, pandas as pd
sys.path.append('.')
from src.model import TeacherCNN, StudentCNN
from src.quantization import apply_ptq
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="CompressionLab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS — mirrors the Flask app exactly ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F6F5F1;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1100px; }

/* NAV BAR */
.navbar {
    background: #FFFFFF;
    border-bottom: 1px solid #DDD9D2;
    padding: 0 28px;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
    box-shadow: 0 1px 0 #DDD9D2;
    position: sticky;
    top: 0;
    z-index: 999;
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-logo-icon {
    width: 32px; height: 32px;
    background: #4A47A3;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 16px;
}
.nav-logo-text {
    font-family: 'Playfair Display', serif;
    font-size: 17px; font-weight: 600;
    color: #17160F;
}
.nav-badge {
    background: #EFEDE8;
    border: 1px solid #DDD9D2;
    padding: 4px 14px;
    border-radius: 99px;
    font-size: 11px;
    color: #7A776E;
    font-family: 'DM Mono', monospace;
}

/* HERO */
.hero {
    background: #FFFFFF;
    border-bottom: 1px solid #DDD9D2;
    padding: 44px 28px 36px;
    text-align: center;
    margin-bottom: 0;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: 2px;
    text-transform: uppercase;
    color: #4A47A3; margin-bottom: 12px;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 34px; letter-spacing: -1px;
    line-height: 1.2; margin-bottom: 10px;
    color: #17160F;
}
.hero h1 em { color: #4A47A3; font-style: normal; }
.hero p { color: #7A776E; font-size: 14px; max-width: 480px; margin: 0 auto 28px; }

/* STAT STRIP */
.stat-strip {
    display: flex;
    border: 1px solid #DDD9D2;
    border-radius: 12px;
    overflow: hidden;
    max-width: 580px;
    margin: 0 auto;
    background: #FFFFFF;
}
.stat-item {
    flex: 1; padding: 14px; text-align: center;
    border-right: 1px solid #DDD9D2;
}
.stat-item:last-child { border-right: none; }
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 22px; color: #4A47A3; font-weight: 700;
}
.stat-lbl {
    font-size: 10px; color: #7A776E;
    font-family: 'DM Mono', monospace; margin-top: 2px;
}

/* SECTION LABEL */
.slabel {
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: 2px;
    text-transform: uppercase;
    color: #7A776E; margin-bottom: 10px;
    margin-top: 28px;
    display: block;
}

/* KPI CARDS */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #DDD9D2;
    border-radius: 13px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05), 0 4px 14px rgba(0,0,0,.06);
    position: relative;
    overflow: hidden;
    margin-bottom: 8px;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-purple::before { background: #4A47A3; }
.kpi-amber::before  { background: #B87020; }
.kpi-green::before  { background: #1E7A50; }
.kpi-blue::before   { background: #2060A8; }
.kpi-label {
    font-size: 10px; color: #7A776E;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 7px;
}
.kpi-val-purple { font-family:'Playfair Display',serif; font-size:30px; font-weight:700; color:#4A47A3; }
.kpi-val-amber  { font-family:'Playfair Display',serif; font-size:30px; font-weight:700; color:#B87020; }
.kpi-val-green  { font-family:'Playfair Display',serif; font-size:30px; font-weight:700; color:#1E7A50; }
.kpi-val-blue   { font-family:'Playfair Display',serif; font-size:30px; font-weight:700; color:#2060A8; }
.kpi-sub { font-size: 11px; color: #7A776E; margin-top: 3px; }

/* CHART CARDS */
.chart-card {
    background: #FFFFFF;
    border: 1px solid #DDD9D2;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05), 0 4px 14px rgba(0,0,0,.06);
    margin-bottom: 18px;
}
.chart-title {
    font-family: 'Playfair Display', serif;
    font-size: 15px; color: #17160F; margin-bottom: 3px;
}
.chart-sub {
    font-size: 11px; color: #7A776E;
    font-family: 'DM Mono', monospace; margin-bottom: 14px;
}

/* UPLOAD CARD */
.upload-card {
    background: #FFFFFF;
    border: 1px solid #DDD9D2;
    border-radius: 18px;
    padding: 32px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05), 0 4px 14px rgba(0,0,0,.06);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.upload-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4A47A3, #7B78D4);
}

/* PRED CARDS */
.pred-card {
    background: #FFFFFF;
    border: 1px solid #DDD9D2;
    border-radius: 16px;
    padding: 22px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05), 0 4px 14px rgba(0,0,0,.06);
    text-align: center;
}
.pred-accent { height: 3px; border-radius: 99px; margin-bottom: 14px; }
.pred-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: 1.5px;
    text-transform: uppercase; color: #7A776E; margin-bottom: 10px;
}
.pred-digit {
    font-family: 'Playfair Display', serif;
    font-size: 76px; font-weight: 700;
    line-height: 1; margin-bottom: 10px;
}
.pred-conf { font-size: 12px; color: #7A776E; margin-bottom: 8px; }

/* PILL BADGES */
.pill {
    display: inline-block; padding: 2px 10px;
    border-radius: 5px; font-size: 10px;
    font-family: 'DM Mono', monospace; font-weight: 500;
    margin: 2px;
}
.pill-purple { background:#EEEEFF; color:#4A47A3; border:1px solid #D4D6F5; }
.pill-amber  { background:#FFF5E8; color:#B87020; border:1px solid #F0DDB8; }
.pill-green  { background:#E8F5EF; color:#1E7A50; border:1px solid #B8DFD0; }
.pill-blue   { background:#E8F0FF; color:#2060A8; border:1px solid #B8C8F0; }

/* SUMMARY CARD */
.sum-card {
    background: #FFFFFF;
    border: 1px solid #DDD9D2;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05), 0 4px 14px rgba(0,0,0,.06);
    margin-top: 16px;
}
.sum-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: 2px;
    text-transform: uppercase; color: #7A776E; margin-bottom: 14px;
}
.sum-item {
    background: #F6F5F1;
    border: 1px solid #DDD9D2;
    border-radius: 10px; padding: 13px;
}
.sum-item-lbl { font-size: 10px; color: #7A776E; font-family: 'DM Mono', monospace; margin-bottom: 4px; }
.sum-item-val { font-size: 13px; font-weight: 600; color: #17160F; }

/* TABLE */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* COMP CARDS */
.comp-card {
    background: #FFFFFF;
    border: 1px solid #DDD9D2;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05), 0 4px 14px rgba(0,0,0,.06);
}
.comp-title { font-family:'Playfair Display',serif; font-size:15px; margin-bottom:4px; color:#17160F; }
.comp-sub { font-size:11px; color:#7A776E; font-family:'DM Mono',monospace; margin-bottom:14px; }

/* LAYER BLOCKS */
.layer-p { background:#EEEEFF; border:1px solid #D4D6F5; color:#4A47A3;
  border-radius:7px; padding:8px 12px; font-size:11px;
  font-family:'DM Mono',monospace; margin-bottom:5px; }
.layer-g { background:#E8F5EF; border:1px solid #B8DFD0; color:#1E7A50;
  border-radius:7px; padding:8px 12px; font-size:11px;
  font-family:'DM Mono',monospace; margin-bottom:5px; }

/* OBS */
.obs { padding:11px 13px; border-radius:9px; border-left:3px solid; margin-bottom:10px; }
.obs-t { font-size:12px; font-weight:600; margin-bottom:3px; }
.obs-d { font-size:12px; color:#7A776E; }

/* TIP */
.tip-box {
    background: #EEF0FF; border: 1px solid #D4D6F5;
    border-radius: 9px; padding: 11px 14px;
    font-size: 13px; color: #4A47A3; margin-bottom: 20px;
}

/* DIVIDER */
.div { border-top: 1px solid #DDD9D2; margin: 24px 0; }

/* Streamlit overrides */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-bottom: 1px solid #DDD9D2;
    padding: 0 28px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px; font-weight: 500;
    color: #7A776E;
    border-radius: 7px 7px 0 0;
    padding: 10px 16px;
}
.stTabs [aria-selected="true"] {
    background: #4A47A3 !important;
    color: white !important;
}
button[kind="primary"] {
    background: #4A47A3 !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ── Navbar ──────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="nav-logo">
    <div class="nav-logo-icon">🧠</div>
    <span class="nav-logo-text">CompressionLab</span>
  </div>
  <div class="nav-badge">Pradhi Bobade · Cummins College · DL March-2026</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Efficient Deep Learning</div>
  <h1>Model Compression <em>&amp; Quantization</em></h1>
  <p>Pruning · PTQ · QAT · Knowledge Distillation · MNIST Dataset</p>
  <div class="stat-strip">
    <div class="stat-item"><div class="stat-val">6</div><div class="stat-lbl">Techniques</div></div>
    <div class="stat-item"><div class="stat-val">99.49%</div><div class="stat-lbl">Best accuracy</div></div>
    <div class="stat-item"><div class="stat-val">63.6×</div><div class="stat-lbl">Max compression</div></div>
    <div class="stat-item"><div class="stat-val">0.43MB</div><div class="stat-lbl">Smallest model</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load models ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    teacher = TeacherCNN()
    teacher.load_state_dict(torch.load('results/teacher.pth', map_location='cpu'))
    teacher.eval()
    student = StudentCNN()
    student.load_state_dict(torch.load('results/student_distilled.pth', map_location='cpu'))
    student.eval()
    tf  = T.Compose([T.ToTensor(), T.Normalize((0.1307,),(0.3081,))])
    dl  = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=tf), batch_size=64)
    ptq = apply_ptq(copy.deepcopy(teacher), dl, 'cpu')
    return teacher, student, ptq

_tf = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

def preprocess(img):
    img = img.convert('L')
    arr = np.array(img)
    if arr.mean() > 127:
        img = Image.fromarray(255 - arr)
    return _tf(img).unsqueeze(0)

def predict(model, tensor):
    with torch.no_grad():
        prob = F.softmax(model(tensor), dim=1)[0]
    return prob.argmax().item(), float(prob.max()) * 100

# ── DATA ────────────────────────────────────────────────────
METHODS = ['Original','Unstructured','Structured','PTQ','QAT','Student KD']
ACC     = [99.34, 99.34, 99.27, 99.34, 99.49, 99.04]
SZ      = [27.20, 27.20, 27.20,  7.91,  7.91,  0.43]
RT      = [round(27.20/s, 1) for s in SZ]
COLORS  = ['#4A47A3','#7B78D4','#9B8FD4','#B87020','#1E7A50','#2060A8']

df_main = pd.DataFrame({'Method':METHODS,'Accuracy':ACC,'Size_MB':SZ,'Compression':RT}).set_index('Method')

# ── TABS ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  📊 Dashboard  ", "  🔍 Live Inference  ", "  📈 Model Comparison  "])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<span class="slabel">Key Results</span>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown("""<div class="kpi-card kpi-purple">
          <div class="kpi-label">Teacher accuracy</div>
          <div class="kpi-val-purple">99.34%</div>
          <div class="kpi-sub">Baseline model</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown("""<div class="kpi-card kpi-amber">
          <div class="kpi-label">Best QAT accuracy</div>
          <div class="kpi-val-amber">99.49%</div>
          <div class="kpi-sub">+0.15% vs baseline</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown("""<div class="kpi-card kpi-green">
          <div class="kpi-label">Max compression</div>
          <div class="kpi-val-green">63.6×</div>
          <div class="kpi-sub">Student KD model</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown("""<div class="kpi-card kpi-blue">
          <div class="kpi-label">Smallest model</div>
          <div class="kpi-val-blue">0.43MB</div>
          <div class="kpi-sub">vs 27.20MB original</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<span class="slabel">Performance Charts</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">Accuracy Comparison</div><div class="chart-sub">Test accuracy per method</div>', unsafe_allow_html=True)
        st.bar_chart(df_main['Accuracy'], color='#4A47A3', height=250)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">Model Size (MB)</div><div class="chart-sub">Saved file size per method</div>', unsafe_allow_html=True)
        st.bar_chart(df_main['Size_MB'], color='#B87020', height=250)
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="chart-card"><div class="chart-title">Compression Ratio</div><div class="chart-sub">Times smaller than original</div>', unsafe_allow_html=True)
        st.bar_chart(df_main['Compression'], color='#1E7A50', height=250)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card"><div class="chart-title">Accuracy vs Size Trade-off</div><div class="chart-sub">Top-left = best trade-off</div>', unsafe_allow_html=True)
        scatter_df = pd.DataFrame({'Size (MB)': SZ, 'Accuracy (%)': ACC}, index=METHODS)
        st.scatter_chart(scatter_df, x='Size (MB)', y='Accuracy (%)', height=250)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<span class="slabel">Full Results Table</span>', unsafe_allow_html=True)
    results_df = pd.DataFrame({
        'Method':      METHODS,
        'Accuracy':    ['99.34%','99.34%','99.27%','99.34%','99.49% ⭐','99.04%'],
        'Size (MB)':   SZ,
        'Compression': ['1.0×','1.0×','1.0×','3.4×','3.4×','63.6× 🚀'],
        'Category':    ['Baseline','Pruning','Pruning','Quantization','Quantization','Distillation'],
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — LIVE INFERENCE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="tip-box">
      💡 Upload a <strong>single handwritten digit (0–9)</strong>.
      White paper photos are auto-inverted. All 3 models run simultaneously.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="slabel">Upload Image</span>', unsafe_allow_html=True)

    col_up, col_prev = st.columns([2, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Choose a digit image",
            type=['png','jpg','jpeg'],
            label_visibility="collapsed"
        )

    with col_prev:
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", width=120)

    if uploaded:
        if st.button("▶  Run All 3 Models", type="primary", use_container_width=True):
            with st.spinner("Running inference on all 3 models..."):
                teacher, student, ptq_model = load_models()
                tensor = preprocess(img)
                t_p, t_c = predict(teacher,   tensor)
                p_p, p_c = predict(ptq_model, tensor)
                s_p, s_c = predict(student,   tensor)

            st.markdown('<span class="slabel">Predictions</span>', unsafe_allow_html=True)
            pc1, pc2, pc3 = st.columns(3)

            with pc1:
                st.markdown(f"""
                <div class="pred-card">
                  <div class="pred-accent" style="background:linear-gradient(90deg,#4A47A3,#7B78D4)"></div>
                  <div class="pred-label">Original Teacher · 27.20 MB</div>
                  <div class="pred-digit" style="color:#4A47A3">{t_p}</div>
                  <div class="pred-conf">Confidence: <strong>{t_c:.1f}%</strong></div>
                  <span class="pill pill-purple">27.20 MB</span>
                  <span class="pill pill-purple">99.34% acc</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(t_c / 100)

            with pc2:
                st.markdown(f"""
                <div class="pred-card">
                  <div class="pred-accent" style="background:linear-gradient(90deg,#B87020,#E09B50)"></div>
                  <div class="pred-label">PTQ · INT8 · 7.91 MB</div>
                  <div class="pred-digit" style="color:#B87020">{p_p}</div>
                  <div class="pred-conf">Confidence: <strong>{p_c:.1f}%</strong></div>
                  <span class="pill pill-amber">7.91 MB</span>
                  <span class="pill pill-amber">3.4× smaller</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(p_c / 100)

            with pc3:
                st.markdown(f"""
                <div class="pred-card">
                  <div class="pred-accent" style="background:linear-gradient(90deg,#1E7A50,#4DB87E)"></div>
                  <div class="pred-label">Student · KD · 0.43 MB</div>
                  <div class="pred-digit" style="color:#1E7A50">{s_p}</div>
                  <div class="pred-conf">Confidence: <strong>{s_c:.1f}%</strong></div>
                  <span class="pill pill-green">0.43 MB</span>
                  <span class="pill pill-green">63.6× smaller</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(s_c / 100)

            # Summary
            agree  = t_p == p_p == s_p
            best_c = max(t_c, p_c, s_c)
            best_n = ['Original Teacher','PTQ (INT8)','Student (KD)'][[t_c,p_c,s_c].index(best_c)]

            st.markdown('<span class="slabel">Prediction Summary</span>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)

            with s1:
                st.markdown(f"""<div class="sum-item">
                  <div class="sum-item-lbl">All models agree?</div>
                  <div class="sum-item-val" style="color:{'#1E7A50' if agree else '#B83030'}">
                    {'✅ Yes — all predicted ' + str(t_p) if agree else '⚠️ No — ' + str(t_p)+'/'+str(p_p)+'/'+str(s_p)}
                  </div></div>""", unsafe_allow_html=True)

            with s2:
                st.markdown(f"""<div class="sum-item">
                  <div class="sum-item-lbl">Highest confidence</div>
                  <div class="sum-item-val">{best_n} · {best_c:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            with s3:
                st.markdown(f"""<div class="sum-item">
                  <div class="sum-item-lbl">Student predicted</div>
                  <div class="sum-item-val">Digit {s_p} · {s_c:.1f}%</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="slabel">Technique Deep Dive</span>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("""<div class="comp-card">
          <div class="comp-title">✂️ Pruning</div>
          <div class="comp-sub">Structured vs Unstructured</div>
        </div>""", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Type':['Unstructured','Structured'],
            'Amount':['40%','30%'],
            'Accuracy':['99.34%','99.27%']
        }), hide_index=True, use_container_width=True)
        st.success("40% weights removed — zero accuracy drop!")

    with d2:
        st.markdown("""<div class="comp-card">
          <div class="comp-title">📉 Quantization</div>
          <div class="comp-sub">PTQ vs QAT</div>
        </div>""", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Type':['PTQ (INT8)','QAT (INT8) ⭐'],
            'Size':['7.91 MB','7.91 MB'],
            'Accuracy':['99.34%','99.49%']
        }), hide_index=True, use_container_width=True)
        st.success("QAT beats original by +0.15%!")

    with d3:
        st.markdown("""<div class="comp-card">
          <div class="comp-title">🎓 Distillation</div>
          <div class="comp-sub">Teacher → Student</div>
        </div>""", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Model':['Teacher','Student 🚀'],
            'Size':['27.20 MB','0.43 MB'],
            'Accuracy':['99.34%','99.04%']
        }), hide_index=True, use_container_width=True)
        st.success("63.6× smaller with only 0.30% drop!")

    st.markdown('<span class="slabel">Architecture Comparison</span>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("""<div class="comp-card">
          <div class="comp-title">Teacher CNN</div>
          <div class="comp-sub">27.20 MB · 6.9M parameters</div>
          <div class="layer-p">Conv2d(1 → 64) + ReLU + MaxPool</div>
          <div class="layer-p">Conv2d(64 → 128) + ReLU + MaxPool</div>
          <div class="layer-p">Conv2d(128 → 256) + ReLU</div>
          <div class="layer-p">Linear(12544 → 512) + Dropout</div>
          <div class="layer-p">Linear(512 → 10)</div>
        </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown("""<div class="comp-card">
          <div class="comp-title">Student CNN</div>
          <div class="comp-sub">0.43 MB · 0.1M parameters · 63.6× smaller</div>
          <div class="layer-g">Conv2d(1 → 16) + ReLU + MaxPool</div>
          <div class="layer-g">Conv2d(16 → 32) + ReLU + MaxPool</div>
          <div class="layer-g">Linear(1568 → 64)</div>
          <div class="layer-g">Linear(64 → 10)</div>
          <div style="margin-top:10px;padding:9px 12px;background:#F6F5F1;
            border-radius:7px;font-size:11px;color:#7A776E;">
            4 fewer layers · 63.6× fewer parameters · same task
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<span class="slabel">Key Observations</span>', unsafe_allow_html=True)
    o1, o2 = st.columns(2)

    with o1:
        st.markdown("""
        <div class="obs" style="background:#EEEEFF;border-color:#4A47A3">
          <div class="obs-t" style="color:#4A47A3">Pruning</div>
          <div class="obs-d">40% weights removed with zero accuracy drop. Models are highly over-parameterized.</div>
        </div>
        <div class="obs" style="background:#FFF5E8;border-color:#B87020">
          <div class="obs-t" style="color:#B87020">PTQ</div>
          <div class="obs-d">3.4× compression instantly. No retraining needed. Best for quick deployment.</div>
        </div>
        """, unsafe_allow_html=True)

    with o2:
        st.markdown("""
        <div class="obs" style="background:#E8F5EF;border-color:#1E7A50">
          <div class="obs-t" style="color:#1E7A50">QAT ⭐</div>
          <div class="obs-d">Beats original by +0.15%. Quantization noise acts as regularization reducing overfitting.</div>
        </div>
        <div class="obs" style="background:#E8F0FF;border-color:#2060A8">
          <div class="obs-t" style="color:#2060A8">Knowledge Distillation 🚀</div>
          <div class="obs-d">63.6× smaller with only 0.30% drop. Best choice for mobile/edge deployment.</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #DDD9D2;padding:18px 28px;text-align:center;
  font-size:11px;color:#7A776E;font-family:'DM Mono',monospace;margin-top:40px;">
  Efficient Deep Learning · Compression &amp; Quantization ·
  Pruning · PTQ · QAT · Knowledge Distillation · MNIST
</div>
""", unsafe_allow_html=True)