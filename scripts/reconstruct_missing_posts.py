from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")
processed = project / "datasets" / "facebook" / "processed"

comments_file = processed / "Data_Comments_Processed.xlsx"
missing_file = processed / "Posts_Manquants_Apify.xlsx"

comments = pd.read_excel(comments_file, dtype={"commentId": str})
missing = pd.read_excel(missing_file)


def normalize_url(value):
    if pd.isna(value):
        return ""
    return str(value).strip().rstrip("/")


def extract_page_name(url):
    if pd.isna(url):
        return pd.NA

    try:
        path = urlparse(str(url)).path.strip("/")
        parts = path.split("/")

        # Exemple :
        # facebook.com/Burkina24/posts/...
        if len(parts) >= 2 and parts[0].lower() != "reel":
            return parts[0]

    except Exception:
        pass

    return pd.NA


# Normalisation
comments["inputUrl_norm"] = comments["inputUrl"].apply(normalize_url)
missing["postUrl"] = missing["postUrl"].apply(normalize_url)

rows = []

for i, row in missing.reset_index(drop=True).iterrows():

    post_url = row["postUrl"]
    post_title = row["postTitle"]

    related_comments = comments[
        comments["inputUrl_norm"] == post_url
    ]

    rows.append({
        "postId": f"RECONSTRUCTED_{i + 1:03d}",
        "postUrl": post_url,
        "pageName": extract_page_name(post_url),
        "text": post_title,
        "commentCount": len(related_comments),
        "sourceType": "reconstructed_from_comments"
    })


reconstructed = pd.DataFrame(rows)

# Export
output = processed / "Data_Posts_Reconstructed.xlsx"
reconstructed.to_excel(output, index=False)


# Contrôles
print("=" * 70)
print("PUBLICATIONS RECONSTRUITES")
print("=" * 70)

print(f"Nombre de publications : {len(reconstructed)}")
print(f"Publications avec texte : {reconstructed['text'].notna().sum()}")
print(f"Commentaires rattachés : {reconstructed['commentCount'].sum()}")

print("\nRépartition par page :")
print(
    reconstructed["pageName"]
    .fillna("source_non_identifiable")
    .value_counts()
)

print("\n" + "=" * 70)
print("LES 23 PUBLICATIONS")
print("=" * 70)

for _, row in reconstructed.iterrows():

    print(f"\nID : {row['postId']}")
    print(f"Page : {row['pageName']}")
    print(f"Commentaires : {row['commentCount']}")
    print(f"Texte : {row['text']}")
    print("-" * 70)

print("\nFichier créé :")
print(output)