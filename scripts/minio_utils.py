from pathlib import Path
from io import BytesIO

import pandas as pd
from minio import Minio


# ============================================================
# CONFIGURATION MINIO
# ============================================================

MINIO_ENDPOINT = "127.0.0.1:9000"

MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

MINIO_BUCKET = "social-media-data"

MINIO_SECURE = False


# ============================================================
# CLIENT MINIO
# ============================================================

def get_minio_client():
    """
    Crée et retourne un client MinIO.
    """

    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )


# ============================================================
# VÉRIFICATION DU BUCKET
# ============================================================

def ensure_bucket():
    """
    Vérifie que le bucket existe.
    Le crée s'il n'existe pas.
    """

    client = get_minio_client()

    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)

        print(
            f"Bucket créé : {MINIO_BUCKET}"
        )

    return client


# ============================================================
# ENVOI D'UN FICHIER
# ============================================================

def upload_file(
    local_file,
    object_name
):
    """
    Envoie un fichier local dans MinIO.

    Exemple :

    upload_file(
        "datasets/facebook/processed/fichier.xlsx",
        "processed/facebook/fichier.xlsx"
    )
    """

    local_file = Path(local_file)

    if not local_file.exists():
        raise FileNotFoundError(
            f"Fichier local introuvable : {local_file}"
        )

    client = ensure_bucket()

    client.fput_object(
        MINIO_BUCKET,
        object_name,
        str(local_file)
    )

    print(
        f"MinIO upload : "
        f"{MINIO_BUCKET}/{object_name}"
    )


# ============================================================
# TÉLÉCHARGEMENT D'UN FICHIER
# ============================================================

def download_file(
    object_name,
    local_file
):
    """
    Télécharge un objet MinIO vers un fichier local.
    """

    local_file = Path(local_file)

    local_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    client = ensure_bucket()

    client.fget_object(
        MINIO_BUCKET,
        object_name,
        str(local_file)
    )

    print(
        f"MinIO download : "
        f"{MINIO_BUCKET}/{object_name}"
        f" -> {local_file}"
    )

    return local_file


# ============================================================
# TEST D'EXISTENCE
# ============================================================

def object_exists(object_name):
    """
    Retourne True si l'objet existe dans MinIO.
    """

    client = ensure_bucket()

    try:
        client.stat_object(
            MINIO_BUCKET,
            object_name
        )

        return True

    except Exception:
        return False


# ============================================================
# LISTE DES OBJETS
# ============================================================

def list_objects(prefix=""):
    """
    Liste les objets présents sous un préfixe MinIO.
    """

    client = ensure_bucket()

    return [
        obj.object_name
        for obj in client.list_objects(
            MINIO_BUCKET,
            prefix=prefix,
            recursive=True
        )
    ]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Test de connexion à MinIO...")
    print()

    client = ensure_bucket()

    print(
        f"Connexion réussie au bucket : "
        f"{MINIO_BUCKET}"
    )

    print()
    print("Objets actuellement présents :")

    objects = list_objects()

    if objects:

        for object_name in objects:
            print(
                " -",
                object_name
            )

    else:

        print(
            "Aucun objet trouvé."
        )

        # ============================================================
# LECTURE EXCEL DIRECTEMENT DEPUIS MINIO
# ============================================================

def read_excel_from_minio(object_name):
    """
    Lit directement un fichier Excel stocké dans MinIO
    et retourne un DataFrame pandas.
    """

    client = ensure_bucket()

    response = client.get_object(
        MINIO_BUCKET,
        object_name
    )

    try:
        data = response.read()

        dataframe = pd.read_excel(
            BytesIO(data)
        )

        return dataframe

    finally:
        response.close()
        response.release_conn()