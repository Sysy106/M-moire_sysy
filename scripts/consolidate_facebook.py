from pathlib import Path
import pandas as pd

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")

posts_dir = project / "datasets" / "facebook" / "raw_posts"
comments_dir = project / "datasets" / "facebook" / "raw_comments"

output_dir = project / "datasets" / "facebook" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CONSOLIDATION DES POSTS
# ============================================================

post_files = sorted(posts_dir.glob("*.xlsx"))

posts_list = []

for file in post_files:
    df = pd.read_excel(file, dtype={"postId": str})
    df["sourceFile"] = file.name
    posts_list.append(df)

posts = pd.concat(posts_list, ignore_index=True, sort=False)

print("Posts avant nettoyage :", len(posts))


# Colonnes utiles
post_columns = [
    "postId",
    "postUrl",
    "inputUrl",
    "createdAt",
    "pageName",
    "text",
    "comments",
    "likes",
    "shares",
    "sourceFile"
]
# Ne garder que les colonnes réellement disponibles
post_columns = [c for c in post_columns if c in posts.columns]

posts = posts[post_columns].copy()

# Renommage
posts.rename(
    columns={
        "comments": "commentCount"
    },
    inplace=True
)

# Dates
if "createdAt" in posts.columns:
    posts["createdAt"] = pd.to_datetime(
        posts["createdAt"],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)

# Suppression des doublons
if "postId" in posts.columns:
    posts = posts.drop_duplicates(
        subset=["postId"],
        keep="first"
    )

print("Posts après suppression des doublons :", len(posts))


# ============================================================
# 2. CONSOLIDATION DES COMMENTS
# ============================================================

comment_files = sorted(comments_dir.glob("*.xlsx"))

comments_list = []

for file in comment_files:
    df = pd.read_excel(
        file,
        dtype={
            "commentId": str,
            "facebookId": str
        }
    )

    df["sourceFile"] = file.name
    comments_list.append(df)

comments = pd.concat(
    comments_list,
    ignore_index=True,
    sort=False
)

print("Commentaires avant nettoyage :", len(comments))


comment_columns = [
    "commentId",
    "commentUrl",
    "inputUrl",
    "date",
    "postTitle",
    "text",
    "likesCount",
    "sourceFile"
]

comment_columns = [
    c for c in comment_columns
    if c in comments.columns
]

comments = comments[comment_columns].copy()

# Renommage du texte
comments.rename(
    columns={
        "text": "commentText"
    },
    inplace=True
)

# Dates
if "date" in comments.columns:
    comments["date"] = pd.to_datetime(
        comments["date"],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)

# Suppression des doublons
if "commentId" in comments.columns:
    comments = comments.drop_duplicates(
        subset=["commentId"],
        keep="first"
    )

print(
    "Commentaires après suppression des doublons :",
    len(comments)
)


# ============================================================
# 3. EXPORT
# ============================================================

posts_output = output_dir / "Data_Posts_Processed.xlsx"
comments_output = output_dir / "Data_Comments_Processed.xlsx"

posts.to_excel(
    posts_output,
    index=False
)

comments.to_excel(
    comments_output,
    index=False
)


print("\n====================================")
print("CONSOLIDATION TERMINEE")
print("====================================")

print("Posts :", len(posts))
print("Commentaires :", len(comments))

print("\nFichiers créés :")
print(posts_output)
print(comments_output)