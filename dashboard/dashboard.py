import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURATION ---
# L'URL de mon API (locale pour l'instant)
API_URL = "http://127.0.0.1:5001/predict"

st.set_page_config(layout="wide")
st.title("Tableau de Bord - Scoring Crédit")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    # On charge les données DEJA traitées (mêmes colonnes que le modèle)
    # On vérifie le chemin est correct par rapport à où tu lances le script
    try:
        df = pd.read_csv('model_production/test_sample_processed.csv')
        return df
    except FileNotFoundError:
        st.error("Fichier de données introuvable. Vérifiez le chemin.")
        return None

df = load_data()

if df is not None:
    st.sidebar.header("Sélection Client")
    
    # On suppose que l'ID est dans les colonnes, sinon on utilise l'index
    if 'SK_ID_CURR' in df.columns:
        id_list = df['SK_ID_CURR'].tolist()
        client_id = st.sidebar.selectbox("Choisir un ID Client", id_list)
        client_row = df[df['SK_ID_CURR'] == client_id]
    else:
        # Si pas d'ID, on utilise l'index
        client_id = st.sidebar.selectbox("Choisir un Index Client", df.index)
        client_row = df.loc[[client_id]]

    st.write(f"### Analyse du client : {client_id}")
    
    # Affichage des infos brutes (Optionnel)
    with st.expander("Voir les données brutes du client"):
        st.dataframe(client_row)

   # --- APPEL API ---
    if st.button("Calculer le Scoring"):
        # 1. Conversion en dictionnaire Python pur
        # orient='records' renvoie une liste [{'col': val, ...}], on prend le 1er élément
        record = client_row.to_dict(orient='records')[0]
        
        # 2. Nettoyage MANUEL et conversion des types Numpy
        # C'est la méthode la plus sûre pour éviter les erreurs JSON
        clean_record = {}
        for key, value in record.items():
            # Si c'est un nombre flottant (float ou numpy float)
            if isinstance(value, (float, np.floating)):
                if np.isnan(value) or np.isinf(value):
                    clean_record[key] = None  # Devient 'null' en JSON (valide)
                else:
                    clean_record[key] = float(value) # Force le type float Python standard
            
            # Si c'est un entier numpy (int64, int32...)
            elif isinstance(value, (int, np.integer)):
                clean_record[key] = int(value) # Force le type int Python standard
            
            # Autres types (str, bool...)
            else:
                clean_record[key] = value

        # 3. On remet dans une liste car l'API s'attend à recevoir une liste d'enregistrements
        json_data = [clean_record]
        
        try:
            with st.spinner('Appel de l\'API en cours...'):
                response = requests.post(API_URL, json=json_data)
            
            if response.status_code == 200:
                res = response.json()
                
                # Récupération des résultats
                proba = res['probability']
                threshold = res['threshold']
                decision = res['status']
                
                # --- AFFICHAGE ---
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Probabilité de Défaut", value=f"{proba:.2%}")
                    st.metric(label="Seuil du Modèle", value=f"{threshold}")
                
                with col2:
                    if decision == "Accordé":
                        st.success(f" Crédit {decision}")
                    else:
                        st.error(f" Crédit {decision}")
                
                # Jauge visuelle simple
                fig, ax = plt.subplots(figsize=(6, 1))
                plt.barh(['Risque'], [proba], color='red' if proba > threshold else 'green')
                plt.xlim(0, 1)
                plt.axvline(x=threshold, color='black', linestyle='--', label=f'Seuil ({threshold})')
                plt.legend()
                st.pyplot(fig)
                
            else:
                st.error(f"Erreur API : {response.status_code}")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")
            st.warning("Vérifiez que l'API (app.py) est bien lancée dans un autre terminal !")