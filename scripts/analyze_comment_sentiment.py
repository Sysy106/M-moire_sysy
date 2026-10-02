from pathlib import Path
import pandas as pd
import numpy as np
import torch

from transformers import pipeline


# ============================================================
# CONFIGURATION
# ============================================================

POSTS_FILE = Path(
    r".\datasets\facebook\processed\Data_Posts_Topic_Clusters_Final.xlsx"
)

COMMENTS_FILE = Path(
    r".\datasets\facebook\processed\Data_Comments_Deduplicated.xlsx"
)

OUTPUT_COMMENTS = Path(
    r".\datasets\facebook\processed\Data_Comments_Sentiment.xlsx"
)

OUTPUT_TOPICS = Path(
    r".\datasets\facebook\processed\Data_Topics_Sentiment.xlsx"
)

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 76)
print("ANALYSE AUTOMATIQUE DU SENTIMENT DES COMMENTAIRES")
print("=" * 76)

posts = pd.read_excel(POSTS_FILE)
comments = pd.read_excel(COMMENTS_FILE)

posts["postId"] = posts["postId"].astype(str)
comments["postId"] = comments["postId"].astype(str)

print("\nPublications pertinentes :", len(posts))
print("Commentaires disponibles :", len(comments))


# ============================================================
# RATTACHEMENT DES SUJETS
# ============================================================

post_topics = (
    posts[
        [
            "postId",
            "topicCluster",
            "autoDomain"
        ]
    ]
    .drop_duplicates(subset=["postId"])
)

comments = comments.merge(
    post_topics,
    on="postId",
    how="left"
)

# Seulement les commentaires appartenant
# aux publications pertinentes
comments_topic = comments[
    comments["topicCluster"].notna()
].copy()

print(
    "Commentaires des sujets pertinents :",
    len(comments_topic)
)


# ============================================================
# COMMENTAIRES UTILISABLES
# ============================================================

usable = comments_topic[
    comments_topic["usableForNLP"] == True
].copy()

usable = usable[
    usable["cleanCommentText"].notna()
].copy()

usable["cleanCommentText"] = (
    usable["cleanCommentText"]
    .astype(str)
    .str.strip()
)

usable = usable[
    usable["cleanCommentText"].str.len() > 0
].copy()

print(
    "Commentaires envoyés au modèle :",
    len(usable)
)


# ============================================================
# MODELE
# ============================================================

print("\nChargement du modèle de sentiment...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1
)

print("Modèle chargé.")

print("\nAnalyse des commentaires...")


# ============================================================
# PREDICTION
# ============================================================

texts = usable[
    "cleanCommentText"
].tolist()

results = sentiment_pipeline(
    texts,
    truncation=True,
    max_length=512,
    batch_size=8
)


# ============================================================
# NORMALISATION DES LABELS
# ============================================================

def normalize_label(label):

    label = str(label).lower()

    if "negative" in label:
        return "NEGATIF"

    if "neutral" in label:
        return "NEUTRE"

    if "positive" in label:
        return "POSITIF"

    return label.upper()


usable["sentimentLabel"] = [
    normalize_label(
        result["label"]
    )
    for result in results
]

usable["sentimentConfidence"] = [
    float(result["score"])
    for result in results
]


# ============================================================
# SCORE NUMERIQUE DESCRIPTIF
#
# -1 = négatif
#  0 = neutre
# +1 = positif
#
# Ce score décrit la tonalité.
# Ce n'est PAS un score de priorité.
# ============================================================

sentiment_values = {
    "NEGATIF": -1,
    "NEUTRE": 0,
    "POSITIF": 1
}

usable["sentimentValue"] = (
    usable["sentimentLabel"]
    .map(sentiment_values)
)


# ============================================================
# EXPORT COMMENTAIRE PAR COMMENTAIRE
# ============================================================

usable.to_excel(
    OUTPUT_COMMENTS,
    index=False
)


# ============================================================
# AGREGATION PAR SUJET
# ============================================================

topic_records = []

for topic, group in usable.groupby(
    "topicCluster"
):

    total = len(group)

    positive = int(
        (group["sentimentLabel"] == "POSITIF").sum()
    )

    neutral = int(
        (group["sentimentLabel"] == "NEUTRE").sum()
    )

    negative = int(
        (group["sentimentLabel"] == "NEGATIF").sum()
    )

    mean_sentiment = (
        group["sentimentValue"].mean()
    )

    mean_confidence = (
        group["sentimentConfidence"].mean()
    )

    topic_records.append(
        {
            "topicCluster": topic,

            "analyzedComments":
                total,

            "positiveComments":
                positive,

            "neutralComments":
                neutral,

            "negativeComments":
                negative,

            "positiveRate":
                positive / total,

            "neutralRate":
                neutral / total,

            "negativeRate":
                negative / total,

            "meanSentiment":
                mean_sentiment,

            "meanSentimentConfidence":
                mean_confidence
        }
    )


topic_sentiment = pd.DataFrame(
    topic_records
)


# ============================================================
# AJOUT DES 17 SUJETS
#
# Les sujets sans commentaire exploitable doivent
# rester présents dans le fichier final.
# ============================================================

all_topics = (
    posts[
        [
            "topicCluster",
            "autoDomain"
        ]
    ]
    .drop_duplicates(
        subset=["topicCluster"]
    )
)

topic_sentiment = all_topics.merge(
    topic_sentiment,
    on="topicCluster",
    how="left"
)

topic_sentiment["analyzedComments"] = (
    topic_sentiment["analyzedComments"]
    .fillna(0)
    .astype(int)
)

for col in [
    "positiveComments",
    "neutralComments",
    "negativeComments"
]:

    topic_sentiment[col] = (
        topic_sentiment[col]
        .fillna(0)
        .astype(int)
    )


# IMPORTANT :
# Pour un sujet sans commentaire,
# les taux et le sentiment moyen restent NaN.
# Ils ne sont pas transformés en zéro.


# ============================================================
# TRI POUR AFFICHAGE
# ============================================================

topic_sentiment = (
    topic_sentiment
    .sort_values(
        by="analyzedComments",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# EXPORT SUJETS
# ============================================================

topic_sentiment.to_excel(
    OUTPUT_TOPICS,
    index=False
)


# ============================================================
# RAPPORT
# ============================================================

print("\n" + "=" * 76)
print("RESULTATS DU SENTIMENT")
print("=" * 76)

print(
    "\nCommentaires analysés :",
    len(usable)
)

print("\nRépartition globale :")

print(
    usable["sentimentLabel"]
    .value_counts()
)

print(
    "\nConfiance moyenne du modèle :",
    round(
        usable[
            "sentimentConfidence"
        ].mean(),
        4
    )
)

print("\n" + "=" * 76)
print("SENTIMENT PAR SUJET")
print("=" * 76)


for _, row in topic_sentiment.iterrows():

    print(
        f"\n{row['topicCluster']}"
        f" | n={row['analyzedComments']}"
        f" | positif={row['positiveComments']}"
        f" | neutre={row['neutralComments']}"
        f" | negatif={row['negativeComments']}"
    )

    if row["analyzedComments"] > 0:

        print(
            "  taux :",
            f"positif={row['positiveRate']:.2%}",
            f"| neutre={row['neutralRate']:.2%}",
            f"| negatif={row['negativeRate']:.2%}",
            f"| sentiment moyen={row['meanSentiment']:.3f}"
        )

    else:

        print(
            "  Aucun commentaire exploitable."
        )


print("\nFichiers créés :")

print(
    OUTPUT_COMMENTS.resolve()
)

print(
    OUTPUT_TOPICS.resolve()
)

print("\nAnalyse du sentiment terminée.")