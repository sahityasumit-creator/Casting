# app.py - Hugging Face + Gradio version (inference only)
import gradio as gr
import numpy as np
import joblib
import torch
import os
from datetime import datetime

# ------------------- CONFIG & PATHS -------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models_saved")

# Hard limits
PT_MIN, PT_MAX = 1524.0, 1560.0
PTE_MIN, PTE_MAX = 16.5, 23.5
RF_RISK_THRESHOLD = 0.25
SMOOTH_ALPHA = 0.4
MAX_DELTA_PT = 200.0
MAX_DELTA_PTE = 2.0

def clamp(pt, pte):
    return float(np.clip(pt, PT_MIN, PT_MAX)), float(np.clip(pte, PTE_MIN, PTE_MAX))

def smooth(current_pt, current_pte, sug_pt, sug_pte):
    sug_pt = np.clip(sug_pt, current_pt - MAX_DELTA_PT, current_pt + MAX_DELTA_PT)
    sug_pte = np.clip(sug_pte, current_pte - MAX_DELTA_PTE, current_pte + MAX_DELTA_PTE)
    pt = SMOOTH_ALPHA * current_pt + (1 - SMOOTH_ALPHA) * sug_pt
    pte = SMOOTH_ALPHA * current_pte + (1 - SMOOTH_ALPHA) * sug_pte
    return clamp(pt, pte)

# ------------------- LOAD MODELS -------------------
print("Loading models...")
try:
    # FBC + KNN
    fbc_pt = joblib.load(os.path.join(MODEL_DIR, "fbc_pt.joblib"))
    fbc_pte = joblib.load(os.path.join(MODEL_DIR, "fbc_pte.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    knn = joblib.load(os.path.join(MODEL_DIR, "knn.joblib"))

    # RF risk models
    rf_sh = joblib.load(os.path.join(MODEL_DIR, "rf_shrinkage.pkl"))
    rf_ce = joblib.load(os.path.join(MODEL_DIR, "rf_ceramic.pkl"))

    # RL Actor
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(6, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, 2), torch.nn.Tanh()
            )
        def forward(self, x):
            return self.net(x) * torch.tensor([25.0, 0.7])  # scale

    actor = Actor()
    actor.load_state_dict(torch.load(os.path.join(MODEL_DIR, "rl_actor.pt"), map_location="cpu"))
    actor.eval()

    MODELS_READY = True
    print("All models loaded successfully!")
except Exception as e:
    MODELS_READY = False
    print("Models not found → running in demo mode")

# ------------------- MAIN RECOMMENDER FUNCTION -------------------
def recommend(RT, RH, WP, LT, Current_PT, Current_PTE):
    if not MODELS_READY:
        return "Models not loaded yet. Train locally and upload to models_saved/ folder."

    state_raw = np.array([RT, RH, WP, LT], dtype=np.float32)
    state_full = np.array([RT, RH, WP, LT, Current_PT, Current_PTE], dtype=np.float32)

    suggestions = []

    # 1. FBC
    s = scaler.transform(state_raw.reshape(1, -1))
    fbc_pt_pred = float(fbc_pt.predict(s)[0])
    fbc_pte_pred = float(fbc_pte.predict(s)[0])
    fbc_pt_pred, fbc_pte_pred = clamp(fbc_pt_pred, fbc_pte_pred)
    X_fbc = np.array([[RT, RH, WP, LT, fbc_pt_pred, fbc_pte_pred]])
    risk_fbc = max(rf_sh.predict_proba(X_fbc)[0][1], rf_ce.predict_proba(X_fbc)[0][1])
    suggestions.append(("FBC", fbc_pt_pred, fbc_pte_pred, risk_fbc))

    # 2. KNN
    dist, idx = knn.kneighbors(s, n_neighbors=min(7, knn.n_neighbors))
    knn_pt_pred = float(np.mean([success_row["PT"] for success_row in success_df.iloc[idx[0]].to_dict("records")]))
    knn_pte_pred = float(np.mean([success_row["PTE"] for success_row in success_df.iloc[idx[0]].to_dict("records")]))
    knn_pt_pred, knn_pte_pred = clamp(knn_pt_pred, knn_pte_pred)
    X_knn = np.array([[RT, RH, WP, LT, knn_pt_pred, knn_pte_pred]])
    risk_knn = max(rf_sh.predict_proba(X_knn)[0][1], rf_ce.predict_proba(X_knn)[0][1])
    suggestions.append(("KNN", knn_pt_pred, knn_pte_pred, risk_knn))

    # 3. RL
    with torch.no_grad():
        delta = actor(torch.from_numpy(state_full)).numpy()
    rl_pt = Current_PT + delta[0]
    rl_pte = Current_PTE + delta[1]
    rl_pt, rl_pte = clamp(rl_pt, rl_pte)
    X_rl = np.array([[RT, RH, WP, LT, rl_pt, rl_pte]])
    risk_rl = max(rf_sh.predict_proba(X_rl)[0][1], rf_ce.predict_proba(X_rl)[0][1])
    suggestions.append(("RL", rl_pt, rl_pte, risk_rl))

    # Choose safest below threshold
    safe = [s for s in suggestions if s[3] < RF_RISK_THRESHOLD]
    if safe:
        best = min(safe, key=lambda x: x[3])
    else:
        best = ("CURRENT", Current_PT, Current_PTE, 999)

    final_pt, final_pte = smooth(Current_PT, Current_PTE, best[1], best[2])

    # Pretty output
    result = f"""
**BEST RECOMMENDATION → {best[0]}**

**Apply:**
• **PT**  →  **{final_pt:.2f}**  ← was {Current_PT:.1f}
• **PTE** →  **{final_pte:.3f}**  ← was {Current_PTE:.2f}
• Predicted defect risk: **{best[3]:.3f}** → {'Safe' if best[3] < RF_RISK_THRESHOLD else 'High Risk'}

**All models:**
"""
    for name, pt, pte, risk in suggestions:
        color = "Safe" if risk < RF_RISK_THRESHOLD else "High Risk"
        result += f"• **{name}** → PT={pt:.2f} | PTE={pte:.3f} | Risk={risk:.3f} → {color}\n"

    return result

# ------------------- GRADIO UI -------------------
with gr.Blocks(title="Casting AI") as demo:
    gr.Markdown("# Casting AI – Smart PT/PTE Recommender")
    gr.Markdown("Used daily by operators • Reduces defects dramatically")

    with gr.Row():
        with gr.Column():
            RT = gr.Slider(400, 900, value=680, label="RT – Room Temperature")
            RH = gr.Slider(20, 80, value=45, label="RH – Humidity (%)")
            WP = gr.Slider(0, 100, value=72, label="WP – Water Pressure")
            LT = gr.Slider(20, 80, value=58, label="LT – Line Temperature")
            Current_PT = gr.Number(value=1542, label="Current PT")
            Current_PTE = gr.Number(value=19.2, label="Current PTE")

        with gr.Column():
            output = gr.Markdown()

    btn = gr.Button("Get AI Recommendation", variant="primary")
    btn.click(fn=recommend, inputs=[RT, RH, WP, LT, Current_PT, Current_PTE], outputs=output)

    gr.Markdown("---\nDeployed for free on Hugging Face Spaces")

demo.launch()