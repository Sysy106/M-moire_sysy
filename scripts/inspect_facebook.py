from pathlib import Path
import pandas as pd

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")

posts_dir = project / "datasets" / "facebook" / "raw_posts"
comments_dir = project / "datasets" / "facebook" / "raw_comments"


def inspect_folder(folder, label):
    files = sorted(folder.glob("*.xlsx"))

    print("\n" + "=" * 70)
    print(f"{label}")
    print("=" * 70)

    print(f"Nombre de fichiers : {len(files)}")

    total_rows = 0

    for i, file in enumerate(files, start=1):
        try:
            df = pd.read_excel(file)

            total_rows += len(df)

            print(f"\n[{i}] {file.name}")
            print(f"    Lignes   : {len(df)}")
            print(f"    Colonnes : {len(df.columns)}")
            print(f"    Noms     : {list(df.columns)}")

        except Exception as e:
            print(f"\nERREUR : {file.name}")
            print(e)

    print("\n" + "-" * 70)
    print(f"TOTAL LIGNES AVANT NETTOYAGE : {total_rows}")
    print("-" * 70)


inspect_folder(posts_dir, "PUBLICATIONS FACEBOOK")
inspect_folder(comments_dir, "COMMENTAIRES FACEBOOK")