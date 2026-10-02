from pathlib import Path

import numpy as np
import pandas as pd
from minio_utils import upload_file


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "datasets"
    / "facebook"
    / "processed"
    / "Data_Topics_Priority_Indicators.xlsx"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "datasets"
    / "facebook"
    / "processed"
    / "Data_Topics_Automatic_Prioritization.xlsx"
)

# Paramètre de prudence utilisé pour le sentiment.
#
# Le facteur n / (n + K) réduit l'importance d'un taux de
# sentiment négatif lorsqu'il repose sur très peu de commentaires.
K_SENTIMENT = 5


# ============================================================
# FONCTIONS
# ============================================================

def normalize_0_100(series):
    """
    Normalise une série entre 0 et 100 avec une normalisation min-max.

    Les valeurs manquantes restent manquantes.
    """

    s = pd.to_numeric(series, errors="coerce")

    valid = s.dropna()

    if valid.empty:
        return pd.Series(np.nan, index=s.index)

    minimum = valid.min()
    maximum = valid.max()

    if maximum == minimum:
        result = pd.Series(np.nan, index=s.index)
        result.loc[s.notna()] = 100.0
        return result

    return 100 * (s - minimum) / (maximum - minimum)


def percentile_score(series):
    """
    Transforme les valeurs disponibles en rangs percentiles
    compris entre 0 et 100.

    Les valeurs manquantes restent manquantes.
    """

    s = pd.to_numeric(series, errors="coerce")

    result = pd.Series(
        np.nan,
        index=s.index,
        dtype=float
    )

    valid = s.notna()

    if valid.sum() == 0:
        return result

    result.loc[valid] = (
        s.loc[valid]
        .rank(
            method="average",
            pct=True
        )
        * 100
    )

    return result


def determine_signal_profile(row):
    """
    Détermine les principales dimensions présentant un signal marqué.

    Les seuils servent uniquement à caractériser le profil
    des signaux observés.
    """

    has_attention = (
        pd.notna(row["attentionScore"])
        and row["attentionScore"] >= 50
    )

    has_reaction = (
        pd.notna(row["reactionScore"])
        and row["reactionScore"] >= 25
    )

    has_engagement = (
        pd.notna(row["engagementScore"])
        and row["engagementScore"] >= 50
    )

    active_signals = []

    if has_attention:
        active_signals.append("ATTENTION")

    if has_reaction:
        active_signals.append("REACTION")

    if has_engagement:
        active_signals.append("ENGAGEMENT")

    if len(active_signals) == 0:
        return "SIGNAL_FAIBLE_OU_LIMITE"

    return "_ET_".join(active_signals)


def intensity_level(value):
    """
    Transforme l'intensité numérique des signaux disponibles
    en catégorie descriptive.

    Ces catégories ne représentent pas des niveaux de priorité
    gouvernementale.
    """

    if pd.isna(value):
        return "NON_CALCULABLE"

    if value < 25:
        return "FAIBLE"

    if value < 50:
        return "MODERE"

    if value < 75:
        return "ELEVE"

    return "TRES_ELEVE"


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

df = pd.read_excel(INPUT_FILE)

print("Nombre de sujets chargés :", len(df))


# ============================================================
# 1. SCORE D'ATTENTION OBSERVÉE
# ============================================================

# L'attention repose sur :
#
# - le nombre de publications appartenant au sujet ;
# - le nombre de commentaires indiqué par Facebook.
#
# log1p réduit l'effet des volumes très élevés.


number_of_posts = pd.to_numeric(
    df["numberOfPosts"],
    errors="coerce"
)

platform_comments = pd.to_numeric(
    df["platformCommentCountSumAvailable"],
    errors="coerce"
)

posts_signal = np.log1p(
    number_of_posts
)

comments_signal = np.log1p(
    platform_comments
)

raw_attention = (
    posts_signal
    + comments_signal
)

df["attentionScore"] = normalize_0_100(
    raw_attention
)


# ============================================================
# 2. SCORE DE RÉACTION NÉGATIVE DOCUMENTÉE
# ============================================================

negative_rate = pd.to_numeric(
    df["negativeRate"],
    errors="coerce"
)

analyzed_comments = pd.to_numeric(
    df["analyzedComments"],
    errors="coerce"
)

# Facteur de prudence :
#
#            n
#       -----------
#          n + K
#
# Exemple :
#
# 1 commentaire -> 1 / 6
# 8 commentaires -> 8 / 13
#
# Cela évite qu'un taux négatif de 100 % basé sur un seul
# commentaire soit interprété comme un signal très fort.


documentation_factor = (
    analyzed_comments
    / (
        analyzed_comments
        + K_SENTIMENT
    )
)

df["reactionScore"] = (
    100
    * negative_rate
    * documentation_factor
)

# Aucun commentaire analysé signifie :
#
# "réaction non mesurable"
#
# et non :
#
# "réaction négative = 0".

df.loc[
    analyzed_comments == 0,
    "reactionScore"
] = np.nan


# ============================================================
# 3. SCORE D'ENGAGEMENT FACEBOOK
# ============================================================

likes = pd.to_numeric(
    df["likesSumAvailable"],
    errors="coerce"
)

shares = pd.to_numeric(
    df["sharesSumAvailable"],
    errors="coerce"
)

# Transformation logarithmique pour limiter l'influence
# des très grandes valeurs.

log_likes = np.log1p(
    likes
)

log_shares = np.log1p(
    shares
)

# Comparaison des sujets disposant réellement de ces données.

df["likesScore"] = percentile_score(
    log_likes
)

df["sharesScore"] = percentile_score(
    log_shares
)

# L'engagement est la moyenne des composantes disponibles.

df["engagementScore"] = (
    df[
        [
            "likesScore",
            "sharesScore"
        ]
    ]
    .mean(
        axis=1,
        skipna=True
    )
)

# Si aucune information d'engagement n'est disponible,
# le score reste manquant.
#
# Une donnée absente n'est jamais remplacée par zéro.

no_engagement_data = (
    df["likesScore"].isna()
    & df["sharesScore"].isna()
)

df.loc[
    no_engagement_data,
    "engagementScore"
] = np.nan


# ============================================================
# 4. CONFIANCE DOCUMENTAIRE
# ============================================================

# Cette dimension est séparée de l'intensité des signaux.
#
# Elle décrit la quantité / couverture d'information disponible
# pour interpréter les résultats.


comment_confidence = normalize_0_100(
    np.log1p(
        analyzed_comments
    )
)

likes_coverage = (
    pd.to_numeric(
        df["likesCoverageRate"],
        errors="coerce"
    )
    * 100
)

shares_coverage = (
    pd.to_numeric(
        df["sharesCoverageRate"],
        errors="coerce"
    )
    * 100
)

platform_comments_coverage = (
    pd.to_numeric(
        df["platformCommentsCoverageRate"],
        errors="coerce"
    )
    * 100
)

confidence_components = pd.concat(
    [
        comment_confidence.rename(
            "commentConfidence"
        ),
        likes_coverage.rename(
            "likesCoverage"
        ),
        shares_coverage.rename(
            "sharesCoverage"
        ),
        platform_comments_coverage.rename(
            "platformCommentsCoverage"
        )
    ],
    axis=1
)

df["dataConfidence"] = (
    confidence_components
    .mean(
        axis=1,
        skipna=True
    )
)


# ============================================================
# 5. PROFIL DES SIGNAUX
# ============================================================

df["signalProfile"] = df.apply(
    determine_signal_profile,
    axis=1
)


# ============================================================
# 6. NOMBRE DE DIMENSIONS DISPONIBLES
# ============================================================

signal_columns = [
    "attentionScore",
    "reactionScore",
    "engagementScore"
]

df["availableSignalDimensions"] = (
    df[signal_columns]
    .notna()
    .sum(axis=1)
)


# ============================================================
# 7. INTENSITÉ DES SIGNAUX DISPONIBLES
# ============================================================

# IMPORTANT :
#
# Cet indicateur n'est PAS une mesure de priorité
# gouvernementale.
#
# Il représente la moyenne de l'intensité des dimensions
# effectivement disponibles dans les données Facebook :
#
# - attention ;
# - réaction ;
# - engagement.
#
# Les valeurs manquantes restent manquantes et ne sont
# jamais transformées en zéro.


df["availableSignalsIntensity"] = (
    df[signal_columns]
    .mean(
        axis=1,
        skipna=True
    )
)


# ============================================================
# 8. NIVEAU D'INTENSITÉ
# ============================================================

df["signalIntensityLevel"] = (
    df["availableSignalsIntensity"]
    .apply(
        intensity_level
    )
)


# ============================================================
# 9. TABLE DE SORTIE
# ============================================================

output_columns = [

    # Identification du sujet
    "topicCluster",
    "mainDomain",
    "representativeText",

    # Volumes observés
    "numberOfPosts",
    "platformCommentCountSumAvailable",
    "usableComments",
    "analyzedComments",

    # Sentiment
    "positiveRate",
    "neutralRate",
    "negativeRate",

    # Engagement brut disponible
    "likesSumAvailable",
    "sharesSumAvailable",

    # Scores techniques
    "attentionScore",
    "reactionScore",
    "likesScore",
    "sharesScore",
    "engagementScore",

    # Qualité documentaire
    "dataConfidence",
    "availableSignalDimensions",

    # Interprétation des signaux
    "signalProfile",
    "availableSignalsIntensity",
    "signalIntensityLevel"
]

result = df[
    output_columns
].copy()


# ============================================================
# 10. TRI POUR LA VISUALISATION
# ============================================================

# Le tri sert uniquement à faciliter la lecture de
# l'intensité des signaux disponibles dans le corpus Facebook.
#
# Il ne constitue pas un classement de décision publique.

result = result.sort_values(
    "availableSignalsIntensity",
    ascending=False
)


# ============================================================
# 11. ARRONDI
# ============================================================

score_columns = [
    "attentionScore",
    "reactionScore",
    "likesScore",
    "sharesScore",
    "engagementScore",
    "dataConfidence",
    "availableSignalsIntensity"
]

result[score_columns] = (
    result[score_columns]
    .round(2)
)


# ============================================================
# 12. EXPORT EXCEL
# ============================================================

result.to_excel(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# 13. SAUVEGARDE DANS MINIO
# ============================================================

MINIO_OBJECT_NAME = (
    "processed/facebook/"
    "Data_Topics_Automatic_Prioritization.xlsx"
)

upload_file(
    OUTPUT_FILE,
    MINIO_OBJECT_NAME
)
# ============================================================
# 13. AFFICHAGE DES RÉSULTATS
# ============================================================

print()
print(
    "=== ANALYSE AUTOMATIQUE DES SIGNAUX TERMINÉE ==="
)
print()

print(
    "Nombre de sujets :",
    len(result)
)

print(
    "Sujets avec réaction mesurable :",
    result["reactionScore"]
    .notna()
    .sum()
)

print(
    "Sujets avec engagement mesurable :",
    result["engagementScore"]
    .notna()
    .sum()
)

print()

print(
    "Répartition des niveaux d'intensité :"
)

print(
    result["signalIntensityLevel"]
    .value_counts()
)

print()

print(
    "Résultat synthétique :"
)

print(
    result[
        [
            "topicCluster",
            "attentionScore",
            "reactionScore",
            "engagementScore",
            "dataConfidence",
            "availableSignalDimensions",
            "availableSignalsIntensity",
            "signalIntensityLevel",
            "signalProfile"
        ]
    ]
    .to_string(
        index=False
    )
)

print()

print(
    "Fichier créé :"
)

print(
    OUTPUT_FILE
)