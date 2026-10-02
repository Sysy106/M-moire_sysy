from pathlib import Path
import pandas as pd

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")

processed = project / "datasets" / "facebook" / "processed"

posts_file = processed / "Data_Posts_Processed.xlsx"
comments_file = processed / "Data_Comments_Processed.xlsx"

posts = pd.read_excel(posts_file, dtype={"postId": str})
comments = pd.read_excel(
    comments_file,
    dtype={
        "commentId": str,
        "facebookId": str
    }
)

print("=" * 60)
print("CONTROLE QUALITE - POSTS")
print("=" * 60)

print("Nombre de posts :", len(posts))
print("Nombre de colonnes :", len(posts.columns))
print("\nColonnes :")
print(list(posts.columns))

print("\nValeurs manquantes :")
print(posts.isna().sum())

if "postId" in posts.columns:
    print("\npostId manquants :", posts["postId"].isna().sum())
    print("postId uniques :", posts["postId"].nunique())

if "text" in posts.columns:
    empty_posts = (
        posts["text"].isna()
        | posts["text"].astype(str).str.strip().eq("")
    )
    print("Posts sans texte :", empty_posts.sum())

if "createdAt" in posts.columns:
    print("Dates manquantes :", posts["createdAt"].isna().sum())
    print("Date minimale :", posts["createdAt"].min())
    print("Date maximale :", posts["createdAt"].max())


print("\n" + "=" * 60)
print("CONTROLE QUALITE - COMMENTAIRES")
print("=" * 60)

print("Nombre de commentaires :", len(comments))
print("Nombre de colonnes :", len(comments.columns))

print("\nColonnes :")
print(list(comments.columns))

print("\nValeurs manquantes :")
print(comments.isna().sum())

if "commentId" in comments.columns:
    print(
        "\ncommentId manquants :",
        comments["commentId"].isna().sum()
    )

    print(
        "commentId uniques :",
        comments["commentId"].nunique()
    )

if "commentText" in comments.columns:
    empty_comments = (
        comments["commentText"].isna()
        | comments["commentText"].astype(str).str.strip().eq("")
    )

    print(
        "Commentaires sans texte :",
        empty_comments.sum()
    )

if "date" in comments.columns:
    print(
        "Dates commentaires manquantes :",
        comments["date"].isna().sum()
    )

    print(
        "Date minimale :",
        comments["date"].min()
    )

    print(
        "Date maximale :",
        comments["date"].max()
    )

print("\n" + "=" * 60)
print("PREPARATION DE LA FUTURE ASSOCIATION")
print("=" * 60)

if "postUrl" in posts.columns:
    print(
        "Posts avec postUrl :",
        posts["postUrl"].notna().sum()
    )

if "commentUrl" in comments.columns:
    print(
        "Commentaires avec commentUrl :",
        comments["commentUrl"].notna().sum()
    )

if "postTitle" in comments.columns:
    print(
        "Commentaires avec postTitle :",
        comments["postTitle"].notna().sum()
    )

print("\nCONTROLE TERMINE")