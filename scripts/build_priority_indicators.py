from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

TOPICS_FILE = Path(
    r".\datasets\facebook\processed\Data_Topics_Analysis.xlsx"
)

SENTIMENT_FILE = Path(
    r".\datasets\facebook\processed\Data_Topics_Sentiment.xlsx"
)

OUTPUT_FILE = Path(
    r".\datasets\facebook\processed\Data_Topics_Priority_Indicators.xlsx"
)


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 76)
print("CONSTRUCTION DES INDICATEURS DESCRIPTIFS PAR SUJET")
print("=" * 76)

topics = pd.read_excel(TOPICS_FILE)
sentiment = pd.read_excel(SENTIMENT_FILE)

print("\nSujets dans Data_Topics_Analysis :", len(topics))
print("Sujets dans Data_Topics_Sentiment :", len(sentiment))


# ============================================================
# COLONNES DE SENTIMENT
# ============================================================

sentiment_columns = [
    "topicCluster",
    "analyzedComments",
    "positiveComments",
    "neutralComments",
    "negativeComments",
    "positiveRate",
    "neutralRate",
    "negativeRate",
    "meanSentiment",
    "meanSentimentConfidence"
]

sentiment = sentiment[sentiment_columns].copy()


# ============================================================
# FUSION
# ============================================================

df = topics.merge(
    sentiment,
    on="topicCluster",
    how="left"
)


# ============================================================
# VERIFICATIONS
# ============================================================

if df["topicCluster"].duplicated().any():
    raise ValueError(
        "Un topicCluster apparaît plusieurs fois."
    )

if len(df) != len(topics):
    raise ValueError(
        "Le nombre de sujets a changé pendant la fusion."
    )


# ============================================================
# COUVERTURE DES COMMENTAIRES
# ============================================================

df["commentCoverageRate"] = np.where(
    df["collectedComments"] > 0,
    df["usableComments"] / df["collectedComments"],
    np.nan
)


# ============================================================
# INTENSITE DE DISCUSSION
#
# Transformation logarithmique uniquement pour faciliter
# les comparaisons entre petits et grands volumes.
# Ce n'est PAS un score politique.
# ============================================================

df["discussionIntensity"] = np.log1p(
    df["usableComments"]
)


# ============================================================
# INTENSITE DES REACTIONS
#
# Mesure la proportion de commentaires exprimant
# une polarité positive OU négative plutôt que neutre.
#
# 0 = réactions entièrement neutres
# 1 = aucune réaction neutre
# ============================================================

df["expressedReactionRate"] = np.where(
    df["analyzedComments"] > 0,
    (
        df["positiveComments"]
        + df["negativeComments"]
    ) / df["analyzedComments"],
    np.nan
)


# ============================================================
# AMPLITUDE DU SENTIMENT
#
# Valeur absolue du sentiment moyen.
#
# Proche de 0 : tonalité moyenne équilibrée/neutre
# Proche de 1 : tonalité moyenne fortement orientée
#
# Attention : ne signifie PAS "priorité".
# ============================================================

df["sentimentAmplitude"] = (
    df["meanSentiment"].abs()
)


# ============================================================
# DISPONIBILITE DES DONNEES D'ENGAGEMENT
# ============================================================

df["likesCoverageRate"] = np.where(
    df["numberOfPosts"] > 0,
    df["postsWithLikesData"] / df["numberOfPosts"],
    np.nan
)

df["sharesCoverageRate"] = np.where(
    df["numberOfPosts"] > 0,
    df["postsWithSharesData"] / df["numberOfPosts"],
    np.nan
)

df["platformCommentsCoverageRate"] = np.where(
    df["numberOfPosts"] > 0,
    df["postsWithPlatformCommentCount"]
    / df["numberOfPosts"],
    np.nan
)


# ============================================================
# PRESENCE DE SIGNAUX
# ============================================================

df["hasDiscussionSignal"] = (
    df["usableComments"] > 0
)

df["hasSentimentSignal"] = (
    df["analyzedComments"] > 0
)

df["hasLikesSignal"] = (
    df["postsWithLikesData"] > 0
)

df["hasSharesSignal"] = (
    df["postsWithSharesData"] > 0
)


# ============================================================
# NIVEAU DE DOCUMENTATION
#
# Ce champ décrit la quantité de réactions disponibles.
# Il ne classe PAS l'importance politique du sujet.
# ============================================================

def documentation_level(n):

    if n == 0:
        return "AUCUNE_REACTION"

    if n <= 2:
        return "TRES_FAIBLE"

    if n <= 5:
        return "FAIBLE"

    if n <= 10:
        return "MODEREE"

    return "PLUS_DOCUMENTEE"


df["reactionDocumentation"] = (
    df["analyzedComments"]
    .apply(documentation_level)
)


# ============================================================
# ORDRE DES COLONNES
# ============================================================

preferred_columns = [

    # Identification du sujet
    "topicCluster",
    "mainDomain",
    "representativePostId",
    "representativeText",

    # Publications
    "numberOfPosts",

    # Discussion
    "collectedComments",
    "usableComments",
    "commentCoverageRate",
    "discussionIntensity",
    "reactionDocumentation",

    # Sentiment
    "analyzedComments",
    "positiveComments",
    "neutralComments",
    "negativeComments",
    "positiveRate",
    "neutralRate",
    "negativeRate",
    "meanSentiment",
    "sentimentAmplitude",
    "expressedReactionRate",
    "meanSentimentConfidence",

    # Engagement disponible
    "postsWithLikesData",
    "likesSumAvailable",
    "likesCoverageRate",

    "postsWithSharesData",
    "sharesSumAvailable",
    "sharesCoverageRate",

    "postsWithPlatformCommentCount",
    "platformCommentCountSumAvailable",
    "platformCommentsCoverageRate",

    # Disponibilité des signaux
    "hasDiscussionSignal",
    "hasSentimentSignal",
    "hasLikesSignal",
    "hasSharesSignal"
]

existing_columns = [
    col
    for col in preferred_columns
    if col in df.columns
]

df = df[existing_columns]


# ============================================================
# TRI POUR LECTURE
#
# Attention :
# ce tri est fondé sur le volume de discussion observé.
# Il ne constitue pas un classement politique des sujets.
# ============================================================

df = df.sort_values(
    by=[
        "usableComments",
        "numberOfPosts"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)


# ============================================================
# EXPORT
# ============================================================

df.to_excel(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RAPPORT
# ============================================================

print("\n" + "=" * 76)
print("RESULTATS DES INDICATEURS")
print("=" * 76)

print("\nNombre de sujets :", len(df))

print(
    "Sujets avec discussion exploitable :",
    int(df["hasDiscussionSignal"].sum())
)

print(
    "Sujets sans discussion exploitable :",
    int((~df["hasDiscussionSignal"]).sum())
)

print(
    "Commentaires analysés :",
    int(df["analyzedComments"].sum())
)


print("\nDocumentation des réactions :")

print(
    df["reactionDocumentation"]
    .value_counts()
)


print("\n" + "=" * 76)
print("TABLEAU SYNTHETIQUE")
print("=" * 76)

for _, row in df.iterrows():

    n = int(row["analyzedComments"])

    if n > 0:

        sentiment_info = (
            f"neg={row['negativeRate']:.1%}"
            f" | neutre={row['neutralRate']:.1%}"
            f" | pos={row['positiveRate']:.1%}"
        )

    else:

        sentiment_info = "sentiment=non disponible"

    print(
        f"\n{row['topicCluster']}"
        f" | {row['mainDomain']}"
        f" | posts={int(row['numberOfPosts'])}"
        f" | commentaires={int(row['usableComments'])}"
        f" | {sentiment_info}"
        f" | documentation={row['reactionDocumentation']}"
    )


print("\nFichier créé :")
print(OUTPUT_FILE.resolve())

print("\nConstruction des indicateurs terminée.")