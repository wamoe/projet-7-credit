import os
import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_API_URL = "http://127.0.0.1:5001/predict"
API_URL_ENV = os.environ.get("API_URL", DEFAULT_API_URL)

st.set_page_config(
    page_title="Scoring Crédit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
/* Layout global */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(0,0,0,.06);}
[data-testid="stSidebar"] .block-container {padding-top: 1.2rem;}

/* Typo */
h1, h2, h3 {letter-spacing: -0.02em;}
.small-muted {color: rgba(0,0,0,.55); font-size: 0.95rem;}

/* Cards */
.card {
  background: rgba(255,255,255,.78);
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  box-shadow: 0 6px 18px rgba(0,0,0,.06);
}
.card-tight {padding: .85rem 1rem;}
.card-title {font-weight: 700; font-size: 0.95rem; color: rgba(0,0,0,.7); margin-bottom: .35rem;}
.card-value {font-weight: 800; font-size: 1.55rem; letter-spacing: -0.02em;}
.card-sub {color: rgba(0,0,0,.55); font-size: .9rem; margin-top: .25rem;}

/* Badges */
.badge {
  display: inline-block;
  padding: .25rem .55rem;
  border-radius: 999px;
  font-weight: 800;
  font-size: .85rem;
  border: 1px solid rgba(0,0,0,.08);
  background: rgba(0,0,0,.03);
}
.badge-ok {background: rgba(16,185,129,.12); border-color: rgba(16,185,129,.35); color: rgb(6,95,70);}
.badge-ko {background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.35); color: rgb(153,27,27);}
.badge-warn {background: rgba(245,158,11,.12); border-color: rgba(245,158,11,.35); color: rgb(120,53,15);}

/* Jauge */
.gauge-wrap {margin-top: .25rem;}
.gauge-track {
  width: 100%;
  height: 14px;
  border-radius: 999px;
  background: rgba(0,0,0,.08);
  overflow: hidden;
  position: relative;
}
.gauge-fill {
  height: 100%;
  border-radius: 999px;
  transition: width .35s ease;
}
.gauge-marker {
  position: absolute;
  top: -6px;
  width: 2px;
  height: 26px;
  background: rgba(0,0,0,.55);
}
.gauge-labels {display:flex; justify-content:space-between; margin-top:.35rem; font-size:.85rem; color: rgba(0,0,0,.55);}
.hr-soft {border: none; height: 1px; background: rgba(0,0,0,.06); margin: .85rem 0;}

/* Streamlit tweaks */
.stButton button {
  border-radius: 12px !important;
  padding: .6rem .9rem !important;
  font-weight: 800 !important;
}
.stTextInput input {
  border-radius: 12px !important;
}
[data-testid="stMetricValue"] {font-size: 1.35rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI helpers
# -----------------------------
def header_card():
    st.markdown("""
    <div class="card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem;">
        <div>
          <div style="font-size:1.85rem; font-weight:900; letter-spacing:-0.03em;">
            Tableau de Bord — Scoring Crédit
          </div>
          <div class="small-muted">
            Sélectionnez un client, calculez le score via l’API, puis analysez la décision.
          </div>
        </div>
        <div class="badge">v1.0</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

def kpi_card(title, value, sub=""):
    st.markdown(f"""
    <div class="card card-tight">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
      <div class="card-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def gauge(prob, threshold):
    prob = float(np.clip(prob, 0.0, 1.0))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    fill_color = "rgba(239,68,68,.75)" if prob >= threshold else "rgba(16,185,129,.75)"

    st.markdown("#### Jauge de risque")
    st.markdown(f"""
    <div class="gauge-wrap">
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{prob*100:.1f}%; background:{fill_color};"></div>
        <div class="gauge-marker" style="left:{threshold*100:.1f}%;"></div>
      </div>
      <div class="gauge-labels">
        <span>0%</span>
        <span>Seuil: {threshold:.2f}</span>
        <span>100%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

def decision_banner(prob, threshold, status_text=None):
    is_refused = prob >= threshold
    badge_cls = "badge-ko" if is_refused else "badge-ok"
    label = "REFUSÉ" if is_refused else "ACCORDÉ"
    small = f"proba={prob:.2%} • seuil={threshold:.2f}"
    if status_text:
        small += f" • statut API={status_text}"

    st.markdown(f"""
    <div class="card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem;">
        <div style="font-size:1.1rem; font-weight:900;">
          Décision: <span class="badge {badge_cls}">{label}</span>
        </div>
        <div class="small-muted">{small}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

def clean_record_for_json(record: dict):
    clean_record = {}
    for k, v in record.items():
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                clean_record[k] = None
            else:
                clean_record[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            clean_record[k] = int(v)
        else:
            try:
                if pd.isna(v):
                    clean_record[k] = None
                else:
                    clean_record[k] = v
            except Exception:
                clean_record[k] = v
    return clean_record

# -----------------------------
# HEADER
# -----------------------------
header_card()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Paramètres")
st.sidebar.caption("API & données")

api_url = st.sidebar.text_input("API URL", value=API_URL_ENV, help="Ex: http://127.0.0.1:5001/predict")
data_path = st.sidebar.text_input("Chemin des données", value="model_production/test_sample_processed.csv")

st.sidebar.divider()
st.sidebar.header("Sélection client")

try:
    df = load_data(data_path)
except Exception as e:
    st.error(f"Impossible de charger le fichier: {data_path}\n\nDétail: {e}")
    st.stop()

if "SK_ID_CURR" in df.columns:
    id_list = df["SK_ID_CURR"].tolist()
    client_id = st.sidebar.selectbox("ID Client", id_list)
    client_row = df[df["SK_ID_CURR"] == client_id]
else:
    client_id = st.sidebar.selectbox("Index Client", df.index)
    client_row = df.loc[[client_id]]

st.sidebar.divider()
call_api = st.sidebar.button(" Calculer le scoring", use_container_width=True)
show_raw = st.sidebar.checkbox("Afficher les données brutes", value=False)
show_details = st.sidebar.checkbox("Afficher l'analyse détaillée des variables", value=True) 

# -----------------------------
# MAIN: CLIENT HEADER
# -----------------------------
top1, top2, top3 = st.columns([2, 1, 1])
with top1:
    st.subheader(f"Client sélectionné : {client_id}")
with top2:
    st.caption("Dataset")
    st.markdown(f"<div class='badge'>lignes: {len(df)}</div>", unsafe_allow_html=True)
with top3:
    st.caption("Features")
    st.markdown(f"<div class='badge'>colonnes: {client_row.shape[1]}</div>", unsafe_allow_html=True)

if show_raw:
    with st.expander(" Données brutes du client", expanded=False):
        st.dataframe(client_row, use_container_width=True)

# -----------------------------
# API STATE
# -----------------------------
if "api_result" not in st.session_state:
    st.session_state.api_result = None
if "api_error" not in st.session_state:
    st.session_state.api_error = None

def do_api_call():
    record = client_row.to_dict(orient="records")[0]
    payload = [clean_record_for_json(record)]

    try:
        with st.spinner("Appel de l’API en cours..."):
            r = requests.post(api_url, json=payload, timeout=10)
        if r.status_code == 200:
            st.session_state.api_result = r.json()
            st.session_state.api_error = None
        else:
            st.session_state.api_result = None
            st.session_state.api_error = f"Erreur API {r.status_code} — {r.text}"
    except requests.exceptions.Timeout:
        st.session_state.api_result = None
        st.session_state.api_error = "Timeout: l’API ne répond pas (10s). Vérifier qu’elle tourne et que l’URL est correcte."
    except Exception as e:
        st.session_state.api_result = None
        st.session_state.api_error = f"Erreur de connexion à l’API: {e}"

if call_api:
    do_api_call()

res = st.session_state.api_result
err = st.session_state.api_error

if err:
    st.error(err)
    st.info("tester l’API : `curl -i http://127.0.0.1:5001/` puis `curl -i http://127.0.0.1:5001/predict`")
    st.stop()

if res is None:
    st.warning("Cliquer sur **« Calculer le scoring »** dans la sidebar pour obtenir une décision.")
    st.stop()

# -----------------------------
# RESULTS
# -----------------------------
proba = float(res.get("probability", 0.0))
threshold = float(res.get("threshold", 0.45))
status = res.get("status", "—")
prediction = int(res.get("prediction", 0))

decision_banner(proba, threshold, status_text=status)
st.write("")

# KPI cards
k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Probabilité de défaut", f"{proba:.2%}", "Plus élevé = plus risqué")
with k2:
    kpi_card("Seuil", f"{threshold:.2f}", "Règle de décision")
with k3:
    kpi_card("Écart au seuil", f"{(proba - threshold):+.2%}", "Proba - seuil")
with k4:
    kpi_card("Classe (0/1)", f"{prediction}", "0=OK • 1=Défaut")

st.write("")
gauge(proba, threshold)
st.caption("La barre représente la probabilité estimée. Si elle dépasse le seuil, la décision est un refus.")
st.write("")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([" Résumé", " Détails variables", " Comparaison dataset"])

with tab1:
    st.markdown("### Résumé")
    st.markdown(f"""
    <div class="card">
      <div style="display:flex; gap:1rem; align-items:flex-start; justify-content:space-between;">
        <div>
          <div class="card-title">Interprétation</div>
          <div class="small-muted">
            • Proba de défaut: <b>{proba:.2%}</b><br/>
            • Seuil: <b>{threshold:.2f}</b><br/>
            • Décision API: <b>{status}</b><br/>
          </div>
        </div>
        <div class="badge {'badge-ko' if proba>=threshold else 'badge-ok'}">
          {'Risque élevé' if proba>=threshold else 'Risque faible'}
        </div>
      </div>
      <hr class="hr-soft"/>
      <div class="small-muted">
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    fig, ax = plt.subplots(figsize=(8, 1.6))
    ax.barh(["Risque"], [proba])
    ax.set_xlim(0, 1)
    ax.axvline(threshold, linestyle="--")
    ax.set_xlabel("Probabilité")
    ax.set_title("Probabilité vs seuil")
    st.pyplot(fig, use_container_width=True)

with tab2:
    st.markdown("### Détails variables (profil client)")
    if not show_details:
        st.info("Active « Afficher l'analyse détaillée » dans la sidebar pour voir cette section.")
    else:
        numeric_cols = client_row.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            st.info("Pas de variables numériques détectées pour ce client.")
        else:
            df_num = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            mu = df_num.mean(skipna=True)
            sigma = df_num.std(skipna=True).replace(0, np.nan)

            client_num = client_row[numeric_cols].iloc[0]
            z = ((client_num - mu) / sigma).abs().sort_values(ascending=False)
            topk = z.head(12).index.tolist()

            st.caption("Top variables les plus atypiques (écart à la moyenne du dataset, en valeur absolue).")
            show_df = pd.DataFrame({
                "feature": topk,
                "valeur_client": [client_num[c] for c in topk],
                "moyenne_dataset": [mu[c] for c in topk],
                "z_abs": [z[c] for c in topk],
            }).sort_values("z_abs", ascending=False)

            st.dataframe(show_df, use_container_width=True)

with tab3:
    st.markdown("### Comparaison (repère client vs seuil)")
    st.caption("Visualisation simple: position du client et seuil de décision.")

    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.scatter([proba], [0.5], s=140)
    ax.axvline(threshold, linestyle="--")
    ax.set_yticks([])
    ax.set_xlabel("Probabilité de défaut (0→1)")
    ax.set_title("Position du client vs seuil")
    st.pyplot(fig, use_container_width=True)

st.divider()

