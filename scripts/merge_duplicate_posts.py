from pathlib import Path
import pandas as pd

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")
processed = project / "datasets" / "facebook" / "processed"

posts_file = processed / "Data_Posts_All.xlsx"
comments_file = processed / "Data_Comments_Clean.xlsx"

output_posts = processed / "Data_Posts_Deduplicated.xlsx"
output_comments = processed / "Data_Comments_Deduplicated.xlsx"
output_dataset = processed / "Data_Posts_Comments_Deduplicated.xlsx"
output_log = processed / "Post_Deduplication_Log.xlsx"


# ==========================================================
# 1. Chargement
# ==========================================================

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


# ==========================================================
# 2. Correspondance du doublon confirmé
# ==========================================================

duplicate_id = "RECONSTRUCTED_002"
canonical_id = "1583320583827786"

print("=" * 75)
print("FUSION D'UN DOUBLON DE PUBLICATION")
print("=" * 75)

print(f"Publication conservée : {canonical_id}")
print(f"Publication fusionnée : {duplicate_id}")


# ==========================================================
# 3. Vérifications avant modification
# ==========================================================

canonical_rows = posts[
    posts["postId"] == canonical_id
]

duplicate_rows = posts[
    posts["postId"] == duplicate_id
]

if len(canonical_rows) != 1:
    raise ValueError(
        f"Le post original {canonical_id} "
        f"apparaît {len(canonical_rows)} fois."
    )

if len(duplicate_rows) != 1:
    raise ValueError(
        f"Le doublon {duplicate_id} "
        f"apparaît {len(duplicate_rows)} fois."
    )


comments_canonical_before = (
    comments["postId"] == canonical_id
).sum()

comments_duplicate_before = (
    comments["postId"] == duplicate_id
).sum()

print("\nAvant fusion :")
print(
    f"Commentaires du post original : "
    f"{comments_canonical_before}"
)
print(
    f"Commentaires du post reconstruit : "
    f"{comments_duplicate_before}"
)


# ==========================================================
# 4. Journal de déduplication
# ==========================================================

duplicate_row = duplicate_rows.iloc[0]
canonical_row = canonical_rows.iloc[0]

log = pd.DataFrame([{
    "removedPostId": duplicate_id,
    "canonicalPostId": canonical_id,
    "removedSourceType": duplicate_row.get(
        "sourceType", pd.NA
    ),
    "canonicalSourceType": canonical_row.get(
        "sourceType", pd.NA
    ),
    "removedPostUrl": duplicate_row.get(
        "postUrl", pd.NA
    ),
    "canonicalPostUrl": canonical_row.get(
        "postUrl", pd.NA
    ),
    "reason": (
        "duplicate_publication_confirmed_"
        "by_text_review"
    ),
    "commentsTransferred": (
        comments_duplicate_before
    )
}])


# ==========================================================
# 5. Transférer les commentaires
# ==========================================================

comments["originalLinkedPostId"] = comments["postId"]

mask_duplicate = (
    comments["postId"] == duplicate_id
)

comments.loc[
    mask_duplicate,
    "postId"
] = canonical_id

comments["linkWasReassigned"] = (
    comments["originalLinkedPostId"]
    != comments["postId"]
)


# ==========================================================
# 6. Retirer uniquement le doublon du corpus analytique
# ==========================================================

posts_dedup = posts[
    posts["postId"] != duplicate_id
].copy()


# ==========================================================
# 7. Vérifications après fusion
# ==========================================================

comments_canonical_after = (
    comments["postId"] == canonical_id
).sum()

comments_still_duplicate = (
    comments["postId"] == duplicate_id
).sum()

print("\nAprès fusion :")
print(
    f"Nombre de publications : "
    f"{len(posts_dedup)}"
)

print(
    f"Commentaires maintenant rattachés "
    f"au post original : "
    f"{comments_canonical_after}"
)

print(
    f"Commentaires encore rattachés "
    f"au doublon : "
    f"{comments_still_duplicate}"
)

print(
    f"Commentaires totaux conservés : "
    f"{len(comments)}"
)


# Sécurité
if len(posts_dedup) != len(posts) - 1:
    raise ValueError(
        "Erreur : le nombre de posts après "
        "déduplication est inattendu."
    )

if len(comments) != 318:
    raise ValueError(
        "Erreur : le nombre total de commentaires "
        "a changé."
    )

if comments_still_duplicate != 0:
    raise ValueError(
        "Erreur : certains commentaires sont encore "
        "rattachés au doublon."
    )


# ==========================================================
# 8. Reconstruire l'agrégation des commentaires
# ==========================================================

usable = comments[
    comments["usableForNLP"] == True
].copy()


def combine_text(series):

    values = []

    for value in series:

        if pd.notna(value):

            value = str(value).strip()

            if value:
                values.append(value)

    return "\n\n--- COMMENT ---\n\n".join(values)


total_stats = (
    comments
    .groupby("postId")
    .agg(
        collectedCommentCount=(
            "commentId", "count"
        ),
        emptyCommentCount=(
            "isEmpty", "sum"
        ),
        potentialSpamCount=(
            "isPotentialSpam", "sum"
        )
    )
    .reset_index()
)


usable_stats = (
    usable
    .groupby("postId")
    .agg(
        usableCommentCount=(
            "commentId", "count"
        ),
        commentsTextClean=(
            "cleanCommentText",
            combine_text
        )
    )
    .reset_index()
)


# ==========================================================
# 9. Dataset analytique final
# ==========================================================

dataset = posts_dedup.merge(
    total_stats,
    on="postId",
    how="left"
)

dataset = dataset.merge(
    usable_stats,
    on="postId",
    how="left"
)

dataset = dataset.rename(
    columns={"text": "postText"}
)


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


def determine_source(row):

    if (
        row["hasPostText"]
        and row["hasUsableComments"]
    ):
        return "POST_AND_COMMENTS"

    if row["hasPostText"]:
        return "POST_ONLY"

    if row["hasUsableComments"]:
        return "COMMENTS_ONLY"

    return "NO_TEXT"


dataset["availableTextSource"] = dataset.apply(
    determine_source,
    axis=1
)


# ==========================================================
# 10. Exports
# ==========================================================

posts_dedup.to_excel(
    output_posts,
    index=False
)

comments.to_excel(
    output_comments,
    index=False
)

dataset.to_excel(
    output_dataset,
    index=False
)

log.to_excel(
    output_log,
    index=False
)


# ==========================================================
# 11. Rapport final
# ==========================================================

print("\n" + "=" * 75)
print("RESULTAT FINAL")
print("=" * 75)

print(
    f"Publications analytiques : {len(dataset)}"
)

print(
    f"Commentaires conservés : {len(comments)}"
)

print(
    "Commentaires réaffectés :",
    comments["linkWasReassigned"].sum()
)

print(
    "Commentaires exploitables NLP :",
    dataset["usableCommentCount"].sum()
)

print("\nDisponibilité textuelle :")

print(
    dataset["availableTextSource"]
    .value_counts()
    .to_string()
)

print("\nFichiers créés :")
print(output_posts)
print(output_comments)
print(output_dataset)
print(output_log)

print(
    "\nAUCUN COMMENTAIRE N'A ETE SUPPRIME."
)