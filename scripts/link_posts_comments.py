from pathlib import Path
import pandas as pd

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")
processed = project / "datasets" / "facebook" / "processed"

posts_file = processed / "Data_Posts_All.xlsx"
comments_file = processed / "Data_Comments_Processed.xlsx"

output_links = processed / "Post_Comment_Links.xlsx"
output_dataset = processed / "Data_Posts_With_Comments.xlsx"

posts = pd.read_excel(posts_file, dtype={"postId": str})
comments = pd.read_excel(comments_file, dtype={"commentId": str})


def normalize_url(value):
    if pd.isna(value):
        return ""
    return str(value).strip().rstrip("/")


# --------------------------------------------------
# 1. Normaliser les URL
# --------------------------------------------------

posts["_postUrl_norm"] = posts["postUrl"].apply(normalize_url)
comments["_inputUrl_norm"] = comments["inputUrl"].apply(normalize_url)


# --------------------------------------------------
# 2. Construire une table URL -> postId
# --------------------------------------------------

url_to_posts = {}

for _, row in posts.iterrows():

    url = row["_postUrl_norm"]

    if not url:
        continue

    if url not in url_to_posts:
        url_to_posts[url] = []

    url_to_posts[url].append(row["postId"])


# --------------------------------------------------
# 3. Rattacher chaque commentaire
# --------------------------------------------------

links = []

for _, comment in comments.iterrows():

    url = comment["_inputUrl_norm"]

    matching_posts = url_to_posts.get(url, [])

    if len(matching_posts) == 1:

        matched_post_id = matching_posts[0]
        status = "MATCHED"

    elif len(matching_posts) > 1:

        # Important :
        # ne pas choisir arbitrairement si plusieurs posts
        # possèdent exactement la même URL.
        matched_post_id = pd.NA
        status = "AMBIGUOUS"

    else:

        matched_post_id = pd.NA
        status = "UNMATCHED"

    links.append({
        "commentId": comment["commentId"],
        "postId": matched_post_id,
        "status": status,
        "postUrl": url,
        "commentText": comment.get("commentText", pd.NA),
        "commentDate": comment.get("date", pd.NaT),
        "commentLikes": comment.get("likesCount", pd.NA)
    })


links_df = pd.DataFrame(links)


# --------------------------------------------------
# 4. Contrôle des rattachements
# --------------------------------------------------

print("=" * 75)
print("RATTACHEMENT POSTS - COMMENTAIRES")
print("=" * 75)

print(f"Publications : {len(posts)}")
print(f"Commentaires : {len(comments)}")

print("\nStatut des commentaires :")
print(links_df["status"].value_counts(dropna=False))


# --------------------------------------------------
# 5. Agréger uniquement les commentaires
#    rattachés sans ambiguïté
# --------------------------------------------------

matched = links_df[
    links_df["status"] == "MATCHED"
].copy()


def combine_comments(series):

    values = []

    for value in series:

        if pd.notna(value):
            value = str(value).strip()

            if value:
                values.append(value)

    return "\n\n--- COMMENT ---\n\n".join(values)


if len(matched) > 0:

    aggregated = matched.groupby(
        "postId",
        as_index=False
    ).agg(
        collectedCommentCount=("commentId", "count"),
        commentsText=("commentText", combine_comments),
        commentsLikesSum=("commentLikes", "sum")
    )

else:

    aggregated = pd.DataFrame(
        columns=[
            "postId",
            "collectedCommentCount",
            "commentsText",
            "commentsLikesSum"
        ]
    )


# --------------------------------------------------
# 6. Fusionner avec les publications
# --------------------------------------------------

dataset = posts.merge(
    aggregated,
    on="postId",
    how="left"
)

dataset["collectedCommentCount"] = (
    dataset["collectedCommentCount"]
    .fillna(0)
    .astype(int)
)

# Indique simplement si nous disposons
# de commentaires textuels exploitables.
dataset["hasCommentsText"] = (
    dataset["commentsText"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)


# --------------------------------------------------
# 7. Export
# --------------------------------------------------

links_df.to_excel(output_links, index=False)

dataset.to_excel(output_dataset, index=False)


# --------------------------------------------------
# 8. Rapport final
# --------------------------------------------------

print("\n" + "=" * 75)
print("COUVERTURE")
print("=" * 75)

print(
    "Commentaires rattachés :",
    (links_df["status"] == "MATCHED").sum()
)

print(
    "Commentaires ambigus :",
    (links_df["status"] == "AMBIGUOUS").sum()
)

print(
    "Commentaires non rattachés :",
    (links_df["status"] == "UNMATCHED").sum()
)

print(
    "Posts avec au moins un commentaire rattaché :",
    (dataset["collectedCommentCount"] > 0).sum()
)

print(
    "Posts sans commentaire rattaché :",
    (dataset["collectedCommentCount"] == 0).sum()
)

print(
    "Posts avec commentaires textuels :",
    dataset["hasCommentsText"].sum()
)

print("\nFichiers créés :")
print(output_links)
print(output_dataset)