import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Home Credit Scoring",
    page_icon="🏦",
    layout="wide"
)

API_URL = "https://home-credit-mlops.onrender.com/predict"

# ── Navigation ────────────────────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "Navigation",
    ["🏦 Scoring Client", "📊 Monitoring Data Drift"]
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 : SCORING
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏦 Scoring Client":

    st.title("🏦 Home Credit Default Risk")
    st.markdown("### Système intelligent de Scoring Crédit")
    st.markdown("---")
    st.write("Remplissez les informations du client puis cliquez sur **Analyser**.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Informations financières")
        revenu = st.number_input("Revenu annuel", value=150000.0, step=10000.0)
        credit = st.number_input("Montant du crédit", value=500000.0, step=50000.0)
        annuite = st.number_input("Annuité", value=25000.0, step=5000.0)
        bien = st.number_input("Prix du bien", value=450000.0, step=50000.0)

    with col2:
        st.subheader("Informations personnelles")
        age = st.slider("Âge", 18, 80, 35)
        anciennete = st.slider("Ancienneté emploi (années)", 0, 40, 5)
        enfants = st.number_input("Nombre d'enfants", min_value=0, value=1)
        famille = st.number_input("Nombre de membres de la famille", min_value=1.0, value=3.0)

    st.markdown("---")
    st.subheader("Scores externes")
    c1, c2, c3 = st.columns(3)
    with c1:
        ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.45)
    with c2:
        ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.60)
    with c3:
        ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.55)

    st.markdown("---")

    if st.button("🔍 Analyser le dossier", use_container_width=True):
        payload = {
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "AMT_CREDIT": credit,
            "AMT_INCOME_TOTAL": revenu,
            "AMT_ANNUITY": annuite,
            "AMT_GOODS_PRICE": bien,
            "DAYS_BIRTH": -age * 365.25,
            "DAYS_EMPLOYED": -anciennete * 365.25,
            "REGION_RATING_CLIENT": 2,
            "DAYS_ID_PUBLISH": -2500,
            "DAYS_REGISTRATION": -4000,
            "DAYS_LAST_PHONE_CHANGE": -300,
            "CNT_CHILDREN": enfants,
            "CNT_FAM_MEMBERS": famille
        }
        try:
            r = requests.post(API_URL, json=payload)
            if r.status_code == 200:
                # --- SAUVEGARDE DES DONNÉES EN PRODUCTION ---
nouvelle_entree = pd.DataFrame([{
    'AMT_INCOME_TOTAL': revenu,
    'AMT_CREDIT': credit,
    'AGE_YEARS': age,
    'EXT_SOURCE_MEAN': (ext1 + ext2 + ext3) / 3,
    'ANNUITY_INCOME_RATIO': annuite / revenu if revenu > 0 else 0,
    'CREDIT_INCOME_RATIO': credit / revenu if revenu > 0 else 0
}])

# Ajouter la ligne dans un fichier CSV local
nouvelle_entree.to_csv("production_logs.csv", mode='a', header=not os.path.exists("production_logs.csv"), index=False)
                resultat = r.json()
                score    = resultat["score_defaut"]
                decision = resultat["decision"]
                risque   = resultat["niveau_risque"]
                message  = resultat["message"]

                st.markdown("---")
                st.subheader("📊 Résultat du scoring")
                st.metric("Probabilité de défaut", f"{score*100:.2f}%")
                st.progress(score)

                if decision == "ACCORDÉ":
                    st.success("✅ Crédit ACCORDÉ")
                else:
                    st.error("❌ Crédit REFUSÉ")

                if risque == "FAIBLE":
                    st.success("🟢 Risque faible")
                elif risque == "MODÉRÉ":
                    st.warning("🟡 Risque modéré")
                elif risque == "ÉLEVÉ":
                    st.warning("🟠 Risque élevé")
                else:
                    st.error("🔴 Risque très élevé")

                st.info(message)
                st.write(f"Seuil utilisé : **{resultat['seuil_utilise']:.2f}**")
            else:
                st.error("Erreur API")
        except Exception as e:
            st.error("Impossible de contacter l'API")
            st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 : MONITORING DATA DRIFT
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("📊 Monitoring — Data Drift")
    st.markdown("### Surveillance de la stabilité du modèle en production")
    st.markdown("---")

    # Simuler données de référence
    np.random.seed(42)
    n = 5000
    reference = pd.DataFrame({
        'AMT_INCOME_TOTAL'   : np.random.lognormal(11.5, 0.8, n),
        'AMT_CREDIT'         : np.random.lognormal(12.5, 0.7, n),
        'AGE_YEARS'          : np.random.normal(44, 11, n),
        'EXT_SOURCE_MEAN'    : np.random.beta(3, 4, n),
        'ANNUITY_INCOME_RATIO': np.random.normal(0.17, 0.08, n),
        'CREDIT_INCOME_RATIO': np.random.normal(3.5, 1.5, n)
    })

    # Simuler données de production avec drift
  # ----------------------------------------------------------------------
# Données de production : Chargement des vRAIES requêtes utilisateurs
# ----------------------------------------------------------------------
if os.path.exists("production_logs.csv"):
    production = pd.read_csv("production_logs.csv")
    st.info(f"📊 Monitoring alimenté par **{len(production)}** analyse(s) client(s) réelle(s).")
else:
    # Option de secours si personne n'a encore utilisé l'application
    st.warning("⚠️ Aucune donnée réelle enregistrée. Veuillez analyser au moins un dossier dans la page Scoring.")
    production = reference.copy() # Évite de faire crasher l'application

    # Test KS pour chaque feature
    st.subheader("🔬 Résultats du Test de Kolmogorov-Smirnov")

    features = list(reference.columns)
    resultats = []

    for feat in features:
        ks_stat, p_value = stats.ks_2samp(reference[feat], production[feat])
        drift = p_value < 0.05
        ref_mean  = reference[feat].mean()
        prod_mean = production[feat].mean()
        diff_pct  = (prod_mean - ref_mean) / abs(ref_mean) * 100
        resultats.append({
            'Feature'       : feat,
            'Moy. Référence': round(ref_mean, 3),
            'Moy. Production': round(prod_mean, 3),
            'Évolution'     : f"{diff_pct:+.1f}%",
            'p-value'       : round(p_value, 4),
            'Statut'        : '⚠️ DRIFT' if drift else '✅ Stable'
        })

    df_resultats = pd.DataFrame(resultats)

    # Afficher tableau
    for _, row in df_resultats.iterrows():
        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1, 1.5])
        col1.write(f"**{row['Feature']}**")
        col2.write(row['Moy. Référence'])
        col3.write(row['Moy. Production'])
        col4.write(row['Évolution'])
        if 'DRIFT' in row['Statut']:
            col5.error(row['Statut'])
        else:
            col5.success(row['Statut'])

    # Bilan
    n_drift = sum(1 for r in resultats if 'DRIFT' in r['Statut'])
    st.markdown("---")

    if n_drift >= 3:
        st.error(f"🚨 ALERTE : {n_drift} features en drift — Réentraînement recommandé !")
    elif n_drift > 0:
        st.warning(f"⚠️ {n_drift} feature(s) en drift — Surveillance renforcée")
    else:
        st.success("✅ Aucun drift détecté — Modèle stable")

    # Graphiques
    st.markdown("---")
    st.subheader("📈 Distribution Référence vs Production")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        axes[i].hist(reference[feat], bins=40, alpha=0.6,
                     color='#2C5F8A', label='Référence', edgecolor='white')
        axes[i].hist(production[feat], bins=40, alpha=0.6,
                     color='#E07B4F', label='Production', edgecolor='white')
        axes[i].set_title(feat, fontweight='bold')
        axes[i].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Stratégie
    st.markdown("---")
    st.subheader("🔄 Stratégie de Réentraînement")
    st.markdown("""
    | Condition | Action |
    |---|---|
    | p-value < 0.05 sur 1-2 features | Surveillance renforcée |
    | p-value < 0.05 sur 3+ features | **Alerte + Réentraînement** |
    | Drift sur EXT_SOURCE_MEAN | **Réentraînement immédiat** |
    
    **Fréquence de monitoring :** Hebdomadaire automatique
    """)