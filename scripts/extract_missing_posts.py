from pathlib import Path
import pandas as pd

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")
processed = project / "datasets" / "facebook" / "processed"

posts_file = processed / "Data_Posts_Processed.xlsx"
comments_file = processed / "Data_Comments_Processed.xlsx"

posts = pd.read_excel(posts_file)
comments = pd.read_excel(comments_file)


def normalize_url(value):
    if pd.isna(value):
        return ""
    return str(value).strip().rstrip("/")


# Normalisation
posts["postUrl_norm"] = posts["postUrl"].apply(normalize_url)
comments["inputUrl_norm"] = comments["inputUrl"].apply(normalize_url)

# URL des posts déjà disponibles
existing_urls = set(
    posts.loc[
        posts["postUrl_norm"] != "",
        "postUrl_norm"
    ]
)

# Commentaires dont la publication n'est pas dans Posts
missing = comments[
    (comments["inputUrl_norm"] != "")
    & (~comments["inputUrl_norm"].isin(existing_urls))
].copy()

# Une ligne par publication manquante
missing_posts = (
    missing
    .groupby("inputUrl_norm", as_index=False)
    .agg(
        postTitle=("postTitle", "first"),
        nombreCommentaires=("commentId", "count")
    )
)

missing_posts.rename(
    columns={"inputUrl_norm": "postUrl"},
    inplace=True
)

missing_posts = missing_posts.sort_values(
    "nombreCommentaires",
    ascending=False
)

# Fichiers de sortie
output_excel = processed / "Posts_Manquants_Apify.xlsx"
output_txt = processed / "Posts_Manquants_Apify.txt"

missing_posts.to_excel(output_excel, index=False)

with open(output_txt, "w", encoding="utf-8") as f:
    for url in missing_posts["postUrl"]:
        f.write(url + "\n")

# Résultats
print("=" * 60)
print("PUBLICATIONS MANQUANTES POUR APIFY")
print("=" * 60)

print("Posts déjà disponibles :", len(existing_urls))
print("Commentaires non associés :", len(missing))
print("Publications manquantes :", len(missing_posts))

print("\nRépartition :")
print(
    missing_posts[
        ["nombreCommentaires", "postUrl"]
    ].to_string(index=False)
)

print("\nFichiers créés :")
print(output_excel)
print(output_txt)

print("\nEXTRACTION TERMINEE")