from pathlib import Path
import pandas as pd

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")
processed = project / "datasets" / "facebook" / "processed"

original_file = processed / "Data_Posts_Processed.xlsx"
reconstructed_file = processed / "Data_Posts_Reconstructed.xlsx"
output_file = processed / "Data_Posts_All.xlsx"

# --------------------------------------------------
# 1. Lecture
# --------------------------------------------------

original = pd.read_excel(original_file, dtype={"postId": str})
reconstructed = pd.read_excel(reconstructed_file, dtype={"postId": str})

print("=" * 70)
print("CONSTRUCTION DU CORPUS GLOBAL")
print("=" * 70)

print(f"Posts originaux       : {len(original)}")
print(f"Posts reconstruits    : {len(reconstructed)}")
print(f"Total avant fusion    : {len(original) + len(reconstructed)}")

# --------------------------------------------------
# 2. Identifier la provenance
# --------------------------------------------------

original["sourceType"] = "original"

if "sourceType" not in reconstructed.columns:
    reconstructed["sourceType"] = "reconstructed_from_comments"

# --------------------------------------------------
# 3. Schéma commun
# --------------------------------------------------

columns = [
    "postId",
    "postUrl",
    "pageName",
    "text",
    "commentCount",
    "createdAt",
    "likes",
    "shares",
    "sourceType"
]

# Ajouter les colonnes absentes sans inventer de valeur
for col in columns:
    if col not in original.columns:
        original[col] = pd.NA

    if col not in reconstructed.columns:
        reconstructed[col] = pd.NA

original = original[columns]
reconstructed = reconstructed[columns]

# --------------------------------------------------
# 4. Fusion
# --------------------------------------------------

all_posts = pd.concat(
    [original, reconstructed],
    ignore_index=True
)

# --------------------------------------------------
# 5. Contrôles
# --------------------------------------------------

print("\nAprès fusion")
print("-" * 70)

print(f"Nombre de lignes       : {len(all_posts)}")
print(f"Posts avec texte       : {all_posts['text'].notna().sum()}")
print(f"Posts sans texte       : {all_posts['text'].isna().sum()}")

print("\nRépartition sourceType :")
print(all_posts["sourceType"].value_counts(dropna=False))

# --------------------------------------------------
# 6. Recherche de doublons EXACTS de texte
# --------------------------------------------------

def normalize_text(value):
    if pd.isna(value):
        return ""

    return " ".join(
        str(value)
        .lower()
        .strip()
        .split()
    )

all_posts["_text_norm"] = all_posts["text"].apply(normalize_text)

duplicates = all_posts[
    (all_posts["_text_norm"] != "") &
    all_posts.duplicated("_text_norm", keep=False)
].sort_values("_text_norm")

print("\nDoublons exacts potentiels :")
print("-" * 70)

if len(duplicates) == 0:
    print("Aucun doublon exact détecté.")
else:
    print(
        duplicates[
            [
                "postId",
                "pageName",
                "sourceType",
                "commentCount",
                "text"
            ]
        ].to_string(
            index=False,
            max_colwidth=100
        )
    )

# Retirer seulement la colonne technique
all_posts = all_posts.drop(columns=["_text_norm"])

# --------------------------------------------------
# 7. Export
# --------------------------------------------------

all_posts.to_excel(output_file, index=False)

print("\n" + "=" * 70)
print("FICHIER CREE")
print("=" * 70)
print(output_file)

print("\nIMPORTANT :")
print("Aucun doublon n'a encore été supprimé.")
print("Aucune publication n'a encore été classifiée.")