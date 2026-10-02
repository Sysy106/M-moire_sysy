import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# CHEMINS DU PROJET
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"

# Permet au dashboard d'importer minio_utils.py
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from minio_utils import read_excel_from_minio


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Analyse des sujets Facebook",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# SOURCE MINIO
# ============================================================

MINIO_OBJECT = (
    "processed/facebook/"
    "Data_Topics_Automatic_Prioritization.xlsx"
)


# ============================================================
# CHARGEMENT DEPUIS MINIO
# ============================================================

@st.cache_data(ttl=60)
def load_data():
    """
    Charge les résultats de priorisation directement depuis MinIO.
    Le cache est renouvelé périodiquement.
    """
    return read_excel_from_minio(
        MINIO_OBJECT
    )


try:
    topics = load_data()

except Exception as error:

    st.error(
        "Impossible de charger les données depuis MinIO."
    )

    st.exception(error)

    st.stop()


# ============================================================
# EN-TÊTE
# ============================================================

st.title(
    "Analyse de sentiment pour la priorisation automatique "
    "des sujets d'intérêt gouvernemental"
)

st.caption(
    "Application aux données issues du réseau social Facebook"
)
st.success(
    "Source des données : MinIO — "
    "social-media-data/processed/facebook/"
)

st.info(
    "Le système identifie automatiquement les sujets d'intérêt "
    "gouvernemental, regroupe les publications par sujet, analyse "
    "la tonalité des commentaires dans un contexte multilingue, "
    "puis calcule des indicateurs permettant de prioriser "
    "automatiquement les sujets selon les signaux observés."
)

# ============================================================
# INDICATEURS GÉNÉRAUX
# ============================================================

total_topics = len(topics)

very_high = int(
    (
        topics["signalIntensityLevel"]
        == "TRES_ELEVE"
    ).sum()
)

high = int(
    (
        topics["signalIntensityLevel"]
        == "ELEVE"
    ).sum()
)

reaction_available = int(
    topics["reactionScore"]
    .notna()
    .sum()
)

engagement_available = int(
    topics["engagementScore"]
    .notna()
    .sum()
)


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Sujets",
    total_topics
)

col2.metric(
    "Très élevé",
    very_high
)

col3.metric(
    "Élevé",
    high
)

col4.metric(
    "Réaction mesurable",
    reaction_available
)

col5.metric(
    "Engagement mesurable",
    engagement_available
)


# ============================================================
# FILTRES
# ============================================================

st.sidebar.header(
    "Filtres"
)


# ------------------------------------------------------------
# Domaines
# ------------------------------------------------------------

domains = sorted(
    topics["mainDomain"]
    .dropna()
    .unique()
    .tolist()
)

selected_domains = st.sidebar.multiselect(
    "Domaine",
    options=domains,
    default=domains
)


# ------------------------------------------------------------
# Niveaux
# ------------------------------------------------------------

level_order = [
    "TRES_ELEVE",
    "ELEVE",
    "MODERE",
    "FAIBLE"
]

available_levels = [
    level
    for level in level_order
    if level in topics[
        "signalIntensityLevel"
    ].unique()
]

selected_levels = st.sidebar.multiselect(
    "Niveau d'intensité",
    options=available_levels,
    default=available_levels
)


# ------------------------------------------------------------
# Filtrage
# ------------------------------------------------------------

filtered = topics[
    topics["mainDomain"].isin(
        selected_domains
    )
    &
    topics[
        "signalIntensityLevel"
    ].isin(
        selected_levels
    )
].copy()


# ============================================================
# TABLEAU SYNTHÉTIQUE
# ============================================================

st.subheader(
    "Vue synthétique"
)

st.write(
    f"{len(filtered)} sujet(s) affiché(s)"
)


display_table = filtered[
    [
        "topicCluster",
        "mainDomain",
        "availableSignalsIntensity",
        "signalIntensityLevel",
        "attentionScore",
        "reactionScore",
        "engagementScore",
        "dataConfidence",
        "availableSignalDimensions",
        "signalProfile"
    ]
].copy()


display_table = display_table.rename(
    columns={
        "topicCluster":
            "Sujet",

        "mainDomain":
            "Domaine",

        "availableSignalsIntensity":
            "Intensité",

        "signalIntensityLevel":
            "Niveau",

        "attentionScore":
            "Attention",

        "reactionScore":
            "Réaction",

        "engagementScore":
            "Engagement",

        "dataConfidence":
            "Confiance",

        "availableSignalDimensions":
            "Dimensions",

        "signalProfile":
            "Profil"
    }
)


display_table = display_table.sort_values(
    "Intensité",
    ascending=False
)


st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# GRAPHIQUE
# ============================================================

st.subheader(
    "Intensité des signaux observés"
)


if len(filtered) > 0:

    chart_data = (
        filtered[
            [
                "topicCluster",
                "availableSignalsIntensity"
            ]
        ]
        .sort_values(
            "availableSignalsIntensity",
            ascending=False
        )
        .set_index(
            "topicCluster"
        )
    )

    st.bar_chart(
        chart_data
    )


st.caption(
    "Le tri représente l'intensité des signaux disponibles "
    "dans le corpus Facebook."
)


# ============================================================
# ANALYSE DÉTAILLÉE
# ============================================================

st.subheader(
    "Analyse détaillée d'un sujet"
)


if len(filtered) == 0:

    st.warning(
        "Aucun sujet ne correspond aux filtres sélectionnés."
    )

else:

    topic_options = (
        filtered["topicCluster"]
        .tolist()
    )

    selected_topic = st.selectbox(
        "Sujet à examiner",
        topic_options
    )


    topic = filtered[
        filtered["topicCluster"]
        == selected_topic
    ].iloc[0]


    # --------------------------------------------------------
    # Identification
    # --------------------------------------------------------

    st.markdown(
        f"### {topic['topicCluster']}"
    )

    st.write(
        "**Domaine :**",
        topic["mainDomain"]
    )

    st.write(
        "**Publication représentative :**"
    )

    st.write(
        topic["representativeText"]
    )


    # --------------------------------------------------------
    # INTENSITÉ ET CONFIANCE
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Intensité des signaux",
        f"{topic['availableSignalsIntensity']:.2f} / 100"
    )

    c2.metric(
        "Niveau",
        topic["signalIntensityLevel"]
    )

    c3.metric(
        "Confiance documentaire",
        f"{topic['dataConfidence']:.2f} / 100"
    )


    st.write(
        "**Profil des signaux :**",
        topic["signalProfile"]
    )

    st.write(
        "**Dimensions disponibles :**",
        f"{int(topic['availableSignalDimensions'])} / 3"
    )


    # --------------------------------------------------------
    # COMPOSANTES
    # --------------------------------------------------------

    st.markdown(
        "#### Composantes du signal"
    )


    score1, score2, score3 = st.columns(3)


    score1.metric(
        "Attention",
        f"{topic['attentionScore']:.2f}"
    )


    if pd.notna(
        topic["reactionScore"]
    ):

        reaction_display = (
            f"{topic['reactionScore']:.2f}"
        )

    else:

        reaction_display = "N/A"


    score2.metric(
        "Réaction",
        reaction_display
    )


    if pd.notna(
        topic["engagementScore"]
    ):

        engagement_display = (
            f"{topic['engagementScore']:.2f}"
        )

    else:

        engagement_display = "N/A"


    score3.metric(
        "Engagement",
        engagement_display
    )


    # --------------------------------------------------------
    # DONNÉES OBSERVÉES
    # --------------------------------------------------------

    st.markdown(
        "#### Données observées"
    )


    obs1, obs2, obs3 = st.columns(3)


    obs1.metric(
        "Publications",
        int(
            topic["numberOfPosts"]
        )
    )


    platform_comments = topic[
        "platformCommentCountSumAvailable"
    ]

    if pd.notna(platform_comments):

        platform_comments_display = int(
            platform_comments
        )

    else:

        platform_comments_display = (
            "Non disponible"
        )


    obs2.metric(
        "Commentaires plateforme",
        platform_comments_display
    )


    obs3.metric(
        "Commentaires analysés",
        int(
            topic["analyzedComments"]
        )
    )


    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    st.markdown(
        "#### Réactions dans les commentaires"
    )


    if topic["analyzedComments"] > 0:

        sentiment = pd.DataFrame(
            {
                "Tonalité": [
                    "Positive",
                    "Neutre",
                    "Négative"
                ],

                "Pourcentage": [
                    topic["positiveRate"] * 100,
                    topic["neutralRate"] * 100,
                    topic["negativeRate"] * 100
                ]
            }
        )

        sentiment[
            "Pourcentage"
        ] = sentiment[
            "Pourcentage"
        ].round(1)


        st.dataframe(
            sentiment,
            hide_index=True,
            use_container_width=True
        )

    else:

        st.info(
            "Aucun commentaire exploitable pour "
            "l'analyse de sentiment."
        )


    # --------------------------------------------------------
    # ENGAGEMENT
    # --------------------------------------------------------

    st.markdown(
        "#### Engagement disponible"
    )


    eng1, eng2 = st.columns(2)


    likes = topic[
        "likesSumAvailable"
    ]

    shares = topic[
        "sharesSumAvailable"
    ]


    if pd.notna(likes):

        likes_display = int(likes)

    else:

        likes_display = (
            "Non disponible"
        )


    if pd.notna(shares):

        shares_display = int(shares)

    else:

        shares_display = (
            "Non disponible"
        )


    eng1.metric(
        "Likes",
        likes_display
    )

    eng2.metric(
        "Partages",
        shares_display
    )


# ============================================================
# MÉTHODOLOGIE
# ============================================================

st.divider()

st.subheader(
    "Méthodologie"
)

st.markdown(
    """
L'analyse automatique repose sur trois dimensions :

- **Attention** : présence du sujet et activité de commentaires ;
- **Réaction** : tonalité négative observée, corrigée selon le
  nombre de commentaires analysés ;
- **Engagement** : likes et partages lorsqu'ils sont disponibles.

La **confiance documentaire** est calculée séparément afin de
rendre visible la couverture des données.

Les valeurs absentes ne sont pas remplacées par zéro.

L'indicateur d'intensité représente la moyenne des dimensions
effectivement disponibles. Il décrit les signaux observés dans
le corpus Facebook et doit être interprété avec leur niveau de
documentation.
"""
)