import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import plotly.graph_objects as go
from pathlib import Path

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (Polished UI)
# ==========================================
st.set_page_config(page_title="Hospital AI Command Center", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Modern Dark Theme Adjustments */
    .stApp { background-color: #0e1117; }
    .css-1d391kg { padding-top: 2rem; }
    .metric-card {
        background-color: #1e2530; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-left: 5px solid #3498db; text-align: center;
    }
    .metric-value { font-size: 32px; font-weight: 800; color: #ffffff; }
    .metric-label { font-size: 14px; color: #a4b0be; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;}
    .alert-card { border-left: 5px solid #ff4757; }
    .success-card { border-left: 5px solid #2ed573; }
    hr { border-color: #2f3542; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL DEFINITION & UTILS
# ==========================================
class DiscreteNHP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        out, _ = self.gru(x)
        rate = self.softplus(self.linear(out))
        return rate.squeeze(-1)

def build_features(df: pd.DataFrame, scaler: dict, use_cal: bool) -> np.ndarray:
    arr = df['arrivals'].values
    arr_scaled = (arr - scaler['arrivals_mean']) / scaler['arrivals_std']
    feats = [arr_scaled.reshape(-1, 1)]
    if use_cal:
        month_oh = np.eye(12)[df['month'].values - 1]
        dow_oh = np.eye(7)[df['dow'].values]
        wknd = df['is_weekend'].values.reshape(-1, 1)
        feats.extend([month_oh, dow_oh, wknd])
    return np.concatenate(feats, axis=1)

# ==========================================
# 3. DATA LOADING (CACHED FOR SPEED)
# ==========================================
@st.cache_resource
def load_environment():
    # Looks for the hospital_nhp folder locally
    base_paths = [".", "./hospital_nhp", "../hospital_nhp", "hospital_nhp"]
    root = None
    for p in base_paths:
        if os.path.exists(f"{p}/config.json"):
            root = Path(p)
            break
            
    if root is None:
        return None, "Error: Could not locate config.json. Make sure app.py is in or next to your 'hospital_nhp' folder."
        
    with open(root / "config.json", "r") as f:
        config = json.load(f)
        
    daily_df = pd.read_csv(root / "processed" / "daily.csv", parse_dates=['date'])
    pat_clean = pd.read_csv(root / "processed" / "patients_clean.csv", parse_dates=['admission_date', 'discharge_date'])
    
    device = torch.device("cpu") # Force CPU for local stability
    model_name = config["best_model"]
    hp = config["hyperparameters"]
    
    use_cal = "calendar" in model_name
    feat_dim = build_features(daily_df.head(10), config["feature_scaler_params"], use_cal).shape[1]
    
    model = DiscreteNHP(feat_dim, hp["hidden_dim"]).to(device)
    model.load_state_dict(torch.load(root / "models" / f"{model_name}.pt", map_location=device))
    model.eval()
    
    return root, config, daily_df, pat_clean, model, device

# ==========================================
# 4. APP LAYOUT & SIDEBAR
# ==========================================
st.title("🏥 AI Hospital Bed Command Center")
st.markdown("**Live Neural Hawkes Process (NHP) Forecasting Engine**")

env = load_environment()
if env[0] is None:
    st.error(env[1])
    st.stop()

root, config, daily_df, pat_clean, model, device = env
use_cal = "calendar" in config["best_model"]
scaler = config["feature_scaler_params"]

st.sidebar.header("⚙️ Simulation Controls")
horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30, step=1)
num_sims = st.sidebar.slider("Monte Carlo Paths (LOS)", min_value=100, max_value=1000, value=250, step=50)
dynamic_capacity = st.sidebar.number_input("Active Bed Capacity", min_value=50, max_value=500, value=config["capacity"], step=10)

st.sidebar.markdown("---")
st.sidebar.caption("System Diagnostics:")
st.sidebar.caption(f"✓ Model: {config['best_model']}")
st.sidebar.caption(f"✓ Data points: {len(daily_df):,} days")

# ==========================================
# 5. LIVE SIMULATION LOGIC
# ==========================================
if st.sidebar.button("🚀 Run Live Prediction", type="primary", use_container_width=True):
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # --- A. Autoregressive Arrival Forecast ---
    status_text.text("Status: Generating sequential event intensities (NHP)...")
    origin_date = daily_df['date'].max() 
    future_dates = [origin_date + pd.Timedelta(days=d+1) for d in range(horizon)]
    
    future_df = pd.DataFrame({'date': future_dates})
    future_df['dow'] = future_df['date'].dt.dayofweek
    future_df['month'] = future_df['date'].dt.month
    future_df['is_weekend'] = future_df['dow'].isin([5, 6]).astype(int)
    
    full_feats = build_features(daily_df, scaler, use_cal)
    seq = torch.tensor(full_feats[-60:], dtype=torch.float32).unsqueeze(0)
    
    lambdas_future = []
    with torch.no_grad():
        for step in range(horizon):
            rate = model(seq)[0, -1].item()
            lambdas_future.append(rate)
            
            if step < horizon - 1:
                arr_scaled = (rate - scaler['arrivals_mean']) / scaler['arrivals_std']
                f_list = [np.array([arr_scaled])]
                if use_cal:
                    month_oh = np.eye(12)[future_df['month'].iloc[step + 1] - 1]
                    dow_oh = np.eye(7)[future_df['dow'].iloc[step + 1]]
                    wknd = np.array([future_df['is_weekend'].iloc[step + 1]])
                    f_list.extend([month_oh, dow_oh, wknd])
                
                f_t = torch.tensor(np.concatenate(f_list), dtype=torch.float32).view(1, 1, -1)
                seq = torch.cat([seq[:, 1:, :], f_t], dim=1)
                
            progress_bar.progress(int((step / horizon) * 40))

    # --- B. Monte Carlo LOS Simulation ---
    status_text.text(f"Status: Simulating {num_sims} Monte Carlo occupancy paths...")
    classes = list(config["los_class_proportions"].keys())
    p_classes = list(config["los_class_proportions"].values())
    pmfs = {c: (np.array([int(k) for k in config["los_pmf"][c].keys()]), np.array(list(config["los_pmf"][c].values()))) for c in classes}
    
    ongoing = np.zeros(horizon)
    for idx, fd in enumerate(future_dates):
        mask = (pat_clean['admission_date'] <= origin_date) & (pat_clean['discharge_date'] > fd)
        ongoing[idx] = int(mask.sum())
        
    sim_occupancy = np.zeros((num_sims, horizon))
    
    for i in range(num_sims):
        arrs = np.atleast_1d(np.random.poisson(lambdas_future))
        occ_matrix = np.zeros(horizon)
        
        for d_offset, a in enumerate(arrs):
            a = int(np.round(a))
            arr_date = future_dates[d_offset]
            if a <= 0: continue
            
            c_samples = np.random.choice(classes, size=a, p=p_classes)
            for c_type in classes:
                count = np.sum(c_samples == c_type)
                if count > 0:
                    loses = np.random.choice(pmfs[c_type][0], size=count, p=pmfs[c_type][1])
                    discharge_dates = arr_date + pd.to_timedelta(loses, unit='D')
                    for fd_idx, fd in enumerate(future_dates):
                        occ_matrix[fd_idx] += np.sum((fd >= arr_date) & (fd < discharge_dates))
                        
        sim_occupancy[i, :] = ongoing + occ_matrix
        if i % 50 == 0:
            progress_bar.progress(40 + int((i / num_sims) * 60))

    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    # --- C. Compute Metrics ---
    req_mean = np.mean(sim_occupancy, axis=0)
    req_q10 = np.percentile(sim_occupancy, 10, axis=0)
    req_q90 = np.percentile(sim_occupancy, 90, axis=0)
    p_breach = np.mean(sim_occupancy > dynamic_capacity, axis=0)
    
    peak_occ = int(np.max(req_mean))
    breach_days = int(np.sum(p_breach > 0.5))
    avg_util = (np.mean(req_mean) / dynamic_capacity) * 100
    
    # --- D. Render KPIs ---
    st.markdown("### 📊 Operational Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Active Capacity</div><div class="metric-value">{dynamic_capacity}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Projected Peak Needs</div><div class="metric-value">{peak_occ}</div></div>', unsafe_allow_html=True)
    with col3:
        b_class = "alert-card" if breach_days > 0 else "success-card"
        st.markdown(f'<div class="metric-card {b_class}"><div class="metric-label">Critical Shortage Days</div><div class="metric-value" style="color: {"#ff4757" if breach_days > 0 else "#2ed573"};">{breach_days}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Utilisation</div><div class="metric-value">{avg_util:.1f}%</div></div>', unsafe_allow_html=True)

    st.write("")
    
    # --- E. Render Interactive Plotly Charts ---
    layout_style = dict(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=40, b=20))
    
    # Chart 1: Bed Occupancy Simulator
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=future_dates, y=req_q90, fill=None, mode='lines', line_color='rgba(52, 152, 219, 0.1)', showlegend=False, hoverinfo='skip'))
    fig1.add_trace(go.Scatter(x=future_dates, y=req_q10, fill='tonexty', mode='lines', line_color='rgba(52, 152, 219, 0.1)', name='80% Simulation Interval'))
    fig1.add_trace(go.Scatter(x=future_dates, y=req_mean, name='Expected Bed Occupancy', line=dict(color='#3498db', width=3)))
    fig1.add_hline(y=dynamic_capacity, line_dash="dash", line_color="#ff4757", annotation_text="Active Bed Limit", annotation_font_color="#ff4757")
    fig1.update_layout(title="Monte Carlo Projected Hospital Bed Requirements", yaxis_title="Number of Beds Occupied", **layout_style)
    
    # Chart 2: Arrival Intensities
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=future_dates, y=lambdas_future, name='Predicted Arrival Rate', line=dict(color='#e67e22', width=3, shape='spline')))
    fig2.update_layout(title="NHP Estimated Daily Patient Arrivals", yaxis_title="Patients per Day", **layout_style)
    
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    
else:
    st.info("👈 Adjust the sliders on the left and click **Run Live Prediction** to execute the Neural Hawkes Process and Monte Carlo Simulator.")