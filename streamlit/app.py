import streamlit as st
import requests

# ------------------------------
# Configuration
# ------------------------------
st.set_page_config(
    page_title="Home Credit Scoring",
    page_icon="🏦",
    layout="wide"
)

API_URL = "https://home-credit-mlops.onrender.com/predict"

# ------------------------------
# Titre
# ------------------------------
st.title("🏦 Home Credit Default Risk")
st.markdown("### Système intelligent de Scoring Crédit")
st.markdown("---")

st.write("Remplissez les informations du client puis cliquez sur **Analyser**.")

# ------------------------------
# Formulaire
# ------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Informations financières")

    revenu = st.number_input(
        "Revenu annuel",
        value=150000.0,
        step=10000.0
    )

    credit = st.number_input(
        "Montant du crédit",
        value=500000.0,
        step=50000.0
    )

    annuite = st.number_input(
        "Annuité",
        value=25000.0,
        step=5000.0
    )

    bien = st.number_input(
        "Prix du bien",
        value=450000.0,
        step=50000.0
    )

with col2:

    st.subheader("Informations personnelles")

    age = st.slider(
        "Âge",
        18,
        80,
        35
    )

    anciennete = st.slider(
        "Ancienneté emploi (années)",
        0,
        40,
        5
    )

    enfants = st.number_input(
        "Nombre d'enfants",
        min_value=0,
        value=1
    )

    famille = st.number_input(
        "Nombre de membres de la famille",
        min_value=1.0,
        value=3.0
    )

st.markdown("---")

st.subheader("Scores externes")

c1, c2, c3 = st.columns(3)

with c1:
    ext1 = st.slider("EXT_SOURCE_1",0.0,1.0,0.45)

with c2:
    ext2 = st.slider("EXT_SOURCE_2",0.0,1.0,0.60)

with c3:
    ext3 = st.slider("EXT_SOURCE_3",0.0,1.0,0.55)

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

        "DAYS_BIRTH": -age*365.25,
        "DAYS_EMPLOYED": -anciennete*365.25,

        "REGION_RATING_CLIENT":2,
        "DAYS_ID_PUBLISH":-2500,
        "DAYS_REGISTRATION":-4000,
        "DAYS_LAST_PHONE_CHANGE":-300,

        "CNT_CHILDREN": enfants,
        "CNT_FAM_MEMBERS": famille

    }

    try:

        r = requests.post(API_URL, json=payload)

        if r.status_code == 200:

            resultat = r.json()

            score = resultat["score_defaut"]
            decision = resultat["decision"]
            risque = resultat["niveau_risque"]
            seuil = resultat["seuil_utilise"]
            message = resultat["message"]

            st.markdown("---")

            st.subheader("📊 Résultat du scoring")

            st.metric(
                "Probabilité de défaut",
                f"{score*100:.2f}%"
            )

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

            st.write(f"Seuil utilisé : **{seuil:.2f}**")

        else:

            st.error("Erreur API")
            st.write(r.text)

    except Exception as e:

        st.error("Impossible de contacter FastAPI")
        st.exception(e)
