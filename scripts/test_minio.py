from minio import Minio

client = Minio(
    "127.0.0.1:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket_name = "social-media-data"

if client.bucket_exists(bucket_name):
    print("Connexion réussie !")
    print(f"Le bucket '{bucket_name}' existe.")
else:
    print("Connexion réussie, mais le bucket n'existe pas.")