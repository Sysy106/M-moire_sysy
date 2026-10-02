from pathlib import Path
import pandas as pd
import re

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")
processed = project / "datasets" / "facebook" / "processed"

posts = pd.read_excel(
    processed / "Data_Posts_Processed.xlsx",
    dtype={"postId": str}
)

comments = pd.read_excel(
    processed / "Data_Comments_Processed.xlsx",
    dtype={"commentId": str}
)


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)

    return value


# Normalisation du texte des posts
posts["text_norm"] = posts["text"].apply(normalize_text)

# Normalisation du titre du post présent dans les commentaires
comments["postTitle_norm"] = comments["postTitle"].apply(normalize_text)


# Ensemble des textes de posts disponibles
post_texts = set(
    posts.loc[
        posts["text_norm"] != "",
        "text_norm"
    ]
)


# Correspondance exacte postTitle <-> text
comments["match_text"] = comments["postTitle_norm"].isin(post_texts)


print("=" * 60)
print("DIAGNOSTIC ASSOCIATION POSTS / COMMENTAIRES")
print("=" * 60)

print("Posts :", len(posts))
print("Commentaires :", len(comments))

print("\nPosts avec texte :", (posts["text_norm"] != "").sum())
print(
    "Commentaires avec postTitle :",
    (comments["postTitle_norm"] != "").sum()
)

matched = comments["match_text"].sum()
unmatched = (~comments["match_text"]).sum()

print("\nCorrespondances exactes :", matched)
print("Non correspondants :", unmatched)

print(
    "Taux de correspondance :",
    round(matched / len(comments) * 100, 2),
    "%"
)


print("\nNombre de postTitle différents :")
print(comments["postTitle_norm"].nunique())


print("\nExemples de postTitle non associés :")

unmatched_titles = (
    comments.loc[
        ~comments["match_text"],
        "postTitle"
    ]
    .drop_duplicates()
    .head(10)
)

for title in unmatched_titles:
    print("-", str(title)[:200])