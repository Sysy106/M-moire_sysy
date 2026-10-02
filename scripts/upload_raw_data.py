from minio import Minio
from pathlib import Path

# -----------------------------
# Connexion à MinIO
# -----------------------------
client = Minio(
    "127.0.0.1:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket = "social-media-data"

# -----------------------------
# Chemins locaux
# -----------------------------
project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")

posts_dir = project / "datasets" / "facebook" / "raw_posts"
comments_dir = project / "datasets" / "facebook" / "raw_comments"

# -----------------------------
# Fonction d'envoi
# -----------------------------
def upload_folder(local_folder, minio_prefix):

    files = list(local_folder.glob("*.xlsx"))

    print(f"\n{len(files)} fichier(s) trouvé(s) dans {local_folder}")

    for file in files:

        object_name = f"{minio_prefix}/{file.name}"

        client.fput_object(
            bucket,
            object_name,
            str(file)
        )

        print(f"OK : {object_name}")

    return len(files)


# -----------------------------
# Envoi des Posts
# -----------------------------
nb_posts = upload_folder(
    posts_dir,
    "raw/facebook/posts"
)

# -----------------------------
# Envoi des Comments
# -----------------------------
nb_comments = upload_folder(
    comments_dir,
    "raw/facebook/comments"
)

# -----------------------------
# Résumé
# -----------------------------
print("\n-----------------------------")
print("IMPORT MINIO TERMINE")
print("-----------------------------")
print(f"Posts    : {nb_posts}")
print(f"Comments : {nb_comments}")
print(f"Total    : {nb_posts + nb_comments}")