from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    r".\datasets\facebook\processed\Data_Posts_Topic_Clusters.xlsx"
)

OUTPUT_FILE = Path(
    r".\datasets\facebook\processed\Data_Posts_Topic_Clusters_Final.xlsx"
)

df = pd.read_excel(INPUT_FILE)

print("=" * 76)
print("CORRECTION FINALE DU CLUSTER TOPIC_002")
print("=" * 76)

topic = df[df["topicCluster"] == "TOPIC_002"]

print("\nContenu actuel de TOPIC_002 :")

for _, row in topic.iterrows():
    print(
        "\n",
        row["postId"],
        ":",
        str(row["postText"])[:250]
    )

# ------------------------------------------------------------
# Séparation des deux sujets
# ------------------------------------------------------------

# PAM / sécurité alimentaire
mask_pam = (
    (df["topicCluster"] == "TOPIC_002")
    &
    (df["postId"].astype(str) == "RECONSTRUCTED_007")
)

# Recherche scientifique / souveraineté sanitaire
mask_health = (
    (df["topicCluster"] == "TOPIC_002")
    &
    (df["postId"].astype(str) == "RECONSTRUCTED_013")
)

df.loc[mask_pam, "topicCluster"] = "TOPIC_002"
df.loc[mask_health, "topicCluster"] = "TOPIC_017"

# Les deux clusters ont désormais une publication chacun
df.loc[
    df["topicCluster"].isin(["TOPIC_002", "TOPIC_017"]),
    "topicSize"
] = 1

df.loc[
    df["topicCluster"].isin(["TOPIC_002", "TOPIC_017"]),
    "isTopicRepresentative"
] = True

df.loc[
    df["topicCluster"].isin(["TOPIC_002", "TOPIC_017"]),
    "similarityToTopicCenter"
] = 1.0

# Correction du domaine du sujet de souveraineté sanitaire
df.loc[
    mask_health,
    "autoDomain"
] = "SANTE"

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 76)
print("RESULTAT")
print("=" * 76)

print(
    "\nTOPIC_002 :",
    len(df[df["topicCluster"] == "TOPIC_002"]),
    "publication"
)

print(
    "TOPIC_017 :",
    len(df[df["topicCluster"] == "TOPIC_017"]),
    "publication"
)

print(
    "\nNombre total de sujets :",
    df["topicCluster"].nunique()
)

print(
    "Nombre total de publications :",
    len(df)
)

print("\nFichier créé :")
print(OUTPUT_FILE.resolve())

print("\nCorrection terminée.")