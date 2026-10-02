from pathlib import Path
import pandas as pd

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")
processed = project / "datasets" / "facebook" / "processed"

posts_file = processed / "Data_Posts_All.xlsx"
comments_file = processed / "Data_Comments_Clean.xlsx"

output_file = processed / "Data_Posts_Comments_Analysis.xlsx"

posts = pd.read_excel(
    posts_file,
    dtype={"postId": str}
)

comments = pd.read_excel(
    comments_file,
    dtype={
        "postId": str,
        "commentId": str
    }
)


# =========================================================
# 1. Commentaires réellement exploitables
# =========================================================

usable = comments[
    comments["usableForNLP"] == True
].copy()


# =========================================================
# 2. Fonction d'agrégation
# =========================================================

def combine_text(series):

    texts = []

    for value in series:

        if pd.notna(value):

            value = str(value).strip()

            if value:
                texts.append(value)

    return "\n\n--- COMMENT ---\n\n".join(texts)


# =========================================================
# 3. Statistiques par publication
# =========================================================

total_stats = (
    comments
    .groupby("postId")
    .agg(
        collectedCommentCount=("commentId", "count"),
        emptyCommentCount=("isEmpty", "sum"),
        potentialSpamCount=("isPotentialSpam", "sum")
    )
    .reset_index()
)


# =========================================================
# 4. Commentaires utilisables par publication
# =========================================================

usable_stats = (
    usable
    .groupby("postId")
    .agg(
        usableCommentCount=("commentId", "count"),
        commentsTextClean=("cleanCommentText", combine_text)
    )
    .reset_index()
)


# =========================================================
# 5. Fusion avec les publications
# =========================================================

dataset = posts.merge(
    total_stats,
    on="postId",
    how="left"
)

dataset = dataset.merge(
    usable_stats,
    on="postId",
    how="left"
)


# =========================================================
# 6. Valeurs manquantes des compteurs
# =========================================================

count_columns = [
    "collectedCommentCount",
    "emptyCommentCount",
    "potentialSpamCount",
    "usableCommentCount"
]

for col in count_columns:

    dataset[col] = (
        dataset[col]
        .fillna(0)
        .astype(int)
    )


# =========================================================
# 7. Renommer explicitement le texte du post
# =========================================================

dataset = dataset.rename(
    columns={
        "text": "postText"
    }
)


# =========================================================
# 8. Indicateurs
# =========================================================

dataset["hasPostText"] = (
    dataset["postText"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

dataset["hasUsableComments"] = (
    dataset["usableCommentCount"] > 0
)


# =========================================================
# 9. Source textuelle disponible
# =========================================================

def determine_text_source(row):

    has_post = row["hasPostText"]
    has_comments = row["hasUsableComments"]

    if has_post and has_comments:
        return "POST_AND_COMMENTS"

    if has_post:
        return "POST_ONLY"

    if has_comments:
        return "COMMENTS_ONLY"

    return "NO_TEXT"


dataset["availableTextSource"] = dataset.apply(
    determine_text_source,
    axis=1
)


# =========================================================
# 10. Export
# =========================================================

dataset.to_excel(
    output_file,
    index=False
)


# =========================================================
# 11. Rapport
# =========================================================

print("=" * 75)
print("DATASET POSTS + COMMENTAIRES")
print("=" * 75)

print(f"Publications totales : {len(dataset)}")

print(
    "Commentaires collectés :",
    dataset["collectedCommentCount"].sum()
)

print(
    "Commentaires exploitables NLP :",
    dataset["usableCommentCount"].sum()
)

print(
    "Commentaires vides :",
    dataset["emptyCommentCount"].sum()
)

print(
    "Spam potentiel :",
    dataset["potentialSpamCount"].sum()
)

print("\nDisponibilité textuelle :")

print(
    dataset["availableTextSource"]
    .value_counts()
    .to_string()
)

print("\nPosts avec texte :", dataset["hasPostText"].sum())

print(
    "Posts avec commentaires exploitables :",
    dataset["hasUsableComments"].sum()
)

print("\nFichier créé :")
print(output_file)