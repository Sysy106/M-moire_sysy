from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

POSTS_FILE = Path(
    r".\datasets\facebook\processed\Data_Posts_Topic_Clusters_Final.xlsx"
)

COMMENTS_FILE = Path(
    r".\datasets\facebook\processed\Data_Comments_Deduplicated.xlsx"
)

OUTPUT_FILE = Path(
    r".\datasets\facebook\processed\Data_Topics_Analysis.xlsx"
)


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 76)
print("CONSTRUCTION DE LA TABLE ANALYTIQUE PAR SUJET")
print("=" * 76)

posts = pd.read_excel(POSTS_FILE)
comments = pd.read_excel(COMMENTS_FILE)

print("\nPublications pertinentes :", len(posts))
print("Commentaires disponibles :", len(comments))


# ============================================================
# PREPARATION DES COMMENTAIRES
# ============================================================

# On conserve uniquement les commentaires rattachés
# aux publications présentes dans le corpus pertinent.

relevant_post_ids = set(
    posts["postId"].astype(str)
)

comments["postId"] = comments["postId"].astype(str)
posts["postId"] = posts["postId"].astype(str)

comments_relevant = comments[
    comments["postId"].isin(relevant_post_ids)
].copy()

print(
    "Commentaires rattachés aux publications pertinentes :",
    len(comments_relevant)
)


# ============================================================
# RATTACHEMENT DU TOPIC AUX COMMENTAIRES
# ============================================================

post_to_topic = (
    posts[
        ["postId", "topicCluster"]
    ]
    .drop_duplicates()
)

comments_relevant = comments_relevant.merge(
    post_to_topic,
    on="postId",
    how="left"
)


# ============================================================
# FONCTIONS
# ============================================================

def safe_sum(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0).sum()


def aggregate_clean_comments(group):

    usable = group[
        group["usableForNLP"] == True
    ]

    texts = (
        usable["cleanCommentText"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    texts = texts[
        texts.str.len() > 0
    ]

    return " || ".join(texts.tolist())


# ============================================================
# STATISTIQUES DES COMMENTAIRES PAR SUJET
# ============================================================

comment_records = []

for topic, group in comments_relevant.groupby(
    "topicCluster"
):

    usable = group[
        group["usableForNLP"] == True
    ]

    spam = group[
        group["isPotentialSpam"] == True
    ]

    empty = group[
        group["isEmpty"] == True
    ]

    comment_records.append(
        {
            "topicCluster": topic,

            "collectedComments":
                len(group),

            "usableComments":
                len(usable),

            "potentialSpamComments":
                len(spam),

            "emptyComments":
                len(empty),

            "commentLikesSum":
                safe_sum(
                    group["commentLikes"]
                ),

            "commentsTextClean":
                aggregate_clean_comments(group)
        }
    )


comments_by_topic = pd.DataFrame(
    comment_records
)


# ============================================================
# STATISTIQUES DES PUBLICATIONS PAR SUJET
# ============================================================

topic_records = []

for topic, group in posts.groupby(
    "topicCluster"
):

    # Domaine dominant
    domain_counts = (
        group["autoDomain"]
        .dropna()
        .value_counts()
    )

    if len(domain_counts) > 0:
        main_domain = domain_counts.index[0]
    else:
        main_domain = "INDETERMINE"

    # Publication représentative
    representative = group[
        group["isTopicRepresentative"] == True
    ]

    if len(representative) > 0:

        representative_row = (
            representative.iloc[0]
        )

    else:

        representative_row = group.iloc[0]

    representative_post_id = (
        representative_row["postId"]
    )

    representative_text = (
        representative_row["postText"]
    )

    # Engagement des publications.
    #
    # Important :
    # les valeurs manquantes restent manquantes.
    # On ne les remplace PAS par zéro.

    likes_available = pd.to_numeric(
        group["likes"],
        errors="coerce"
    )

    shares_available = pd.to_numeric(
        group["shares"],
        errors="coerce"
    )

    platform_comments_available = pd.to_numeric(
        group["commentCount"],
        errors="coerce"
    )

    topic_records.append(
        {
            "topicCluster": topic,

            "numberOfPosts":
                len(group),

            "mainDomain":
                main_domain,

            "representativePostId":
                representative_post_id,

            "representativeText":
                representative_text,

            "postsWithLikesData":
                int(
                    likes_available.notna().sum()
                ),

            "likesSumAvailable":
                (
                    likes_available.sum()
                    if likes_available.notna().any()
                    else np.nan
                ),

            "postsWithSharesData":
                int(
                    shares_available.notna().sum()
                ),

            "sharesSumAvailable":
                (
                    shares_available.sum()
                    if shares_available.notna().any()
                    else np.nan
                ),

            "postsWithPlatformCommentCount":
                int(
                    platform_comments_available
                    .notna()
                    .sum()
                ),

            "platformCommentCountSumAvailable":
                (
                    platform_comments_available.sum()
                    if platform_comments_available
                    .notna()
                    .any()
                    else np.nan
                ),
        }
    )


topics = pd.DataFrame(
    topic_records
)


# ============================================================
# FUSION
# ============================================================

analysis = topics.merge(
    comments_by_topic,
    on="topicCluster",
    how="left"
)


# Sujets sans commentaires collectés
count_columns = [
    "collectedComments",
    "usableComments",
    "potentialSpamComments",
    "emptyComments",
    "commentLikesSum"
]

for col in count_columns:

    analysis[col] = (
        analysis[col]
        .fillna(0)
    )


analysis["commentsTextClean"] = (
    analysis["commentsTextClean"]
    .fillna("")
)


# ============================================================
# INDICATEURS DESCRIPTIFS
# ============================================================

analysis["usableCommentRate"] = np.where(
    analysis["collectedComments"] > 0,

    analysis["usableComments"]
    / analysis["collectedComments"],

    np.nan
)


analysis["hasCitizenDiscussion"] = (
    analysis["usableComments"] > 0
)


# ============================================================
# TRI
#
# Ce tri sert uniquement à faciliter la lecture.
# Il ne constitue PAS encore la priorisation finale.
# ============================================================

analysis = analysis.sort_values(
    by=[
        "collectedComments",
        "numberOfPosts"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(
    drop=True
)


# ============================================================
# EXPORT
# ============================================================

analysis.to_excel(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RAPPORT
# ============================================================

print("\n" + "=" * 76)
print("RESULTATS PAR SUJET")
print("=" * 76)

print("\nNombre de sujets :", len(analysis))

print(
    "Nombre total de publications :",
    int(analysis["numberOfPosts"].sum())
)

print(
    "Commentaires collectés sur les sujets :",
    int(analysis["collectedComments"].sum())
)

print(
    "Commentaires exploitables NLP :",
    int(analysis["usableComments"].sum())
)

print(
    "Sujets avec au moins un commentaire exploitable :",
    int(analysis["hasCitizenDiscussion"].sum())
)


print("\n" + "=" * 76)
print("DETAIL")
print("=" * 76)


for _, row in analysis.iterrows():

    text = str(
        row["representativeText"]
    ).replace(
        "\n",
        " "
    )[:150]

    print(
        f"\n{row['topicCluster']}"
        f" | posts={int(row['numberOfPosts'])}"
        f" | commentaires={int(row['collectedComments'])}"
        f" | utilisables={int(row['usableComments'])}"
        f" | domaine={row['mainDomain']}"
    )

    print(
        "Sujet représentatif :",
        text
    )


print("\nFichier créé :")
print(OUTPUT_FILE.resolve())

print("\nConstruction terminée.")