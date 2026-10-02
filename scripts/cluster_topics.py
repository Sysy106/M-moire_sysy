import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    r".\datasets\facebook\processed\Data_Posts_Auto_Classified_Final.xlsx"
)

OUTPUT_FILE = Path(
    r".\datasets\facebook\processed\Data_Posts_Topic_Clusters.xlsx"
)

SUMMARY_FILE = Path(
    r".\datasets\facebook\processed\Data_Topic_Clusters_Summary.xlsx"
)

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Seuil de distance cosinus.
# 0.35 signifie qu'on regroupe les textes ayant
# une similarité suffisamment forte.
DISTANCE_THRESHOLD = 0.35


# ============================================================
# NORMALISATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 76)
print("REGROUPEMENT AUTOMATIQUE DES PUBLICATIONS EN SUJETS")
print("=" * 76)

df = pd.read_excel(INPUT_FILE)

required_columns = [
    "postId",
    "postText",
    "autoLabel",
    "autoDomain"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Colonne obligatoire absente : {col}"
        )


# ============================================================
# CORPUS PERTINENT
# ============================================================

pertinent = df[
    df["autoLabel"] == "PERTINENT"
].copy()

pertinent["normalizedText"] = (
    pertinent["postText"].apply(normalize_text)
)

pertinent = pertinent[
    pertinent["normalizedText"].str.len() > 0
].copy()

pertinent.reset_index(
    drop=True,
    inplace=True
)

print(
    "\nPublications PERTINENT à regrouper :",
    len(pertinent)
)

if len(pertinent) < 2:
    raise ValueError(
        "Pas assez de publications pour effectuer le clustering."
    )


# ============================================================
# MODELE SEMANTIQUE
# ============================================================

print("\nChargement du modèle...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Création des embeddings...")

embeddings = model.encode(
    pertinent["normalizedText"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# MATRICE DE DISTANCE
# ============================================================

similarity_matrix = cosine_similarity(
    embeddings
)

distance_matrix = 1 - similarity_matrix

# Correction de petites erreurs numériques éventuelles
distance_matrix = np.clip(
    distance_matrix,
    0,
    2
)

np.fill_diagonal(
    distance_matrix,
    0
)


# ============================================================
# CLUSTERING HIERARCHIQUE
# ============================================================

print("\nClustering automatique...")

clustering = AgglomerativeClustering(
    n_clusters=None,
    metric="precomputed",
    linkage="average",
    distance_threshold=DISTANCE_THRESHOLD
)

labels = clustering.fit_predict(
    distance_matrix
)

pertinent["clusterRaw"] = labels


# ============================================================
# RENUMEROTATION DES CLUSTERS
#
# On attribue les numéros TOPIC_001, TOPIC_002...
# selon la taille décroissante.
# ============================================================

cluster_sizes = (
    pertinent["clusterRaw"]
    .value_counts()
    .to_dict()
)

ordered_clusters = sorted(
    cluster_sizes.keys(),
    key=lambda x: (
        -cluster_sizes[x],
        x
    )
)

cluster_mapping = {
    old: f"TOPIC_{i:03d}"
    for i, old in enumerate(
        ordered_clusters,
        start=1
    )
}

pertinent["topicCluster"] = (
    pertinent["clusterRaw"]
    .map(cluster_mapping)
)

pertinent["topicSize"] = (
    pertinent["clusterRaw"]
    .map(cluster_sizes)
)


# ============================================================
# PUBLICATION REPRESENTATIVE DU SUJET
#
# On choisit la publication la plus proche
# du centroïde sémantique du cluster.
# ============================================================

pertinent["isTopicRepresentative"] = False
pertinent["similarityToTopicCenter"] = np.nan

topic_records = []


for raw_cluster, group in pertinent.groupby(
    "clusterRaw"
):

    indices = group.index.tolist()

    cluster_vectors = embeddings[indices]

    centroid = np.mean(
        cluster_vectors,
        axis=0
    )

    centroid = centroid / np.linalg.norm(
        centroid
    )

    center_similarities = np.dot(
        cluster_vectors,
        centroid
    )

    best_local_position = int(
        np.argmax(center_similarities)
    )

    representative_index = indices[
        best_local_position
    ]

    pertinent.loc[
        indices,
        "similarityToTopicCenter"
    ] = center_similarities

    pertinent.at[
        representative_index,
        "isTopicRepresentative"
    ] = True

    representative_text = pertinent.at[
        representative_index,
        "postText"
    ]

    representative_post_id = pertinent.at[
        representative_index,
        "postId"
    ]

    topic_id = cluster_mapping[
        raw_cluster
    ]

    domains = (
        group["autoDomain"]
        .value_counts()
    )

    main_domain = domains.index[0]

    # Similarité moyenne interne du cluster
    if len(indices) > 1:

        internal_matrix = cosine_similarity(
            cluster_vectors
        )

        upper_values = internal_matrix[
            np.triu_indices(
                len(indices),
                k=1
            )
        ]

        mean_internal_similarity = float(
            np.mean(upper_values)
        )

    else:

        mean_internal_similarity = 1.0

    topic_records.append(
        {
            "topicCluster": topic_id,
            "topicSize": len(indices),
            "mainDomain": main_domain,
            "representativePostId":
                representative_post_id,
            "representativeText":
                representative_text,
            "meanInternalSimilarity":
                round(
                    mean_internal_similarity,
                    4
                )
        }
    )


# ============================================================
# TABLEAU RESUME DES SUJETS
# ============================================================

topics = pd.DataFrame(
    topic_records
)

topics = topics.sort_values(
    by=[
        "topicSize",
        "topicCluster"
    ],
    ascending=[
        False,
        True
    ]
).reset_index(
    drop=True
)


# ============================================================
# EXPORT
# ============================================================

pertinent.drop(
    columns=["clusterRaw"],
    inplace=True
)

pertinent.to_excel(
    OUTPUT_FILE,
    index=False
)

topics.to_excel(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# RAPPORT
# ============================================================

print("\n" + "=" * 76)
print("RESULTATS DU CLUSTERING")
print("=" * 76)

print(
    "\nPublications analysées :",
    len(pertinent)
)

print(
    "Nombre de sujets détectés :",
    pertinent["topicCluster"].nunique()
)

print("\nTaille des sujets :")

print(
    pertinent["topicCluster"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 76)
print("DETAIL DES SUJETS")
print("=" * 76)


for _, topic in topics.iterrows():

    topic_id = topic["topicCluster"]

    print(
        f"\n{topic_id}"
        f" | taille={topic['topicSize']}"
        f" | domaine={topic['mainDomain']}"
        f" | similarité interne="
        f"{topic['meanInternalSimilarity']}"
    )

    group = pertinent[
        pertinent["topicCluster"] == topic_id
    ]

    for _, row in group.iterrows():

        text_preview = str(
            row["postText"]
        ).replace(
            "\n",
            " "
        )[:180]

        marker = (
            "*"
            if row["isTopicRepresentative"]
            else "-"
        )

        print(
            f" {marker} {row['postId']} : "
            f"{text_preview}"
        )


print("\nFichiers créés :")

print(
    OUTPUT_FILE.resolve()
)

print(
    SUMMARY_FILE.resolve()
)

print(
    "\n* = publication représentative du sujet"
)

print("\nClustering terminé.")