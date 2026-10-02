from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd
import re
import unicodedata

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")
processed = project / "datasets" / "facebook" / "processed"

input_file = processed / "Data_Posts_All.xlsx"
output_file = processed / "Post_Similarities.xlsx"

df = pd.read_excel(input_file, dtype={"postId": str})


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).lower()

    # Retirer les accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    # Retirer URL
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Garder lettres et chiffres
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["_text_norm"] = df["text"].apply(normalize_text)

# Seulement les publications ayant du texte
with_text = df[df["_text_norm"] != ""].copy()

results = []

for i in range(len(with_text)):
    for j in range(i + 1, len(with_text)):

        row1 = with_text.iloc[i]
        row2 = with_text.iloc[j]

        text1 = row1["_text_norm"]
        text2 = row2["_text_norm"]

        similarity = SequenceMatcher(
            None,
            text1,
            text2
        ).ratio()

        # On conserve seulement les ressemblances >= 50 %
        if similarity >= 0.50:

            if similarity >= 0.90:
                relation = "DOUBLON_POTENTIEL"

            elif similarity >= 0.70:
                relation = "TRES_PROCHE"

            else:
                relation = "MEME_SUJET_POSSIBLE"

            results.append({
                "postId_1": row1["postId"],
                "sourceType_1": row1["sourceType"],
                "pageName_1": row1["pageName"],

                "postId_2": row2["postId"],
                "sourceType_2": row2["sourceType"],
                "pageName_2": row2["pageName"],

                "similarity": round(similarity, 3),
                "relation": relation,

                "text_1": row1["text"],
                "text_2": row2["text"]
            })


similarities = pd.DataFrame(results)

if len(similarities) > 0:
    similarities = similarities.sort_values(
        "similarity",
        ascending=False
    )

similarities.to_excel(
    output_file,
    index=False
)

print("=" * 75)
print("CONTROLE DES PUBLICATIONS SIMILAIRES")
print("=" * 75)

print(f"Publications totales : {len(df)}")
print(f"Publications avec texte : {len(with_text)}")
print(f"Paires examinées : {len(with_text) * (len(with_text)-1) // 2}")
print(f"Paires avec similarité >= 50 % : {len(similarities)}")

if len(similarities) > 0:

    print("\nRépartition :")
    print(similarities["relation"].value_counts())

    print("\n" + "=" * 75)
    print("PAIRES DETECTEES")
    print("=" * 75)

    for _, row in similarities.iterrows():

        print(
            f"\n{row['postId_1']}  <-->  "
            f"{row['postId_2']}"
        )

        print(
            f"Sources : {row['pageName_1']} "
            f"<--> {row['pageName_2']}"
        )

        print(
            f"Similarité : "
            f"{row['similarity'] * 100:.1f}%"
        )

        print(
            f"Relation : {row['relation']}"
        )

        print(
            f"Texte 1 : {str(row['text_1'])[:180]}"
        )

        print(
            f"Texte 2 : {str(row['text_2'])[:180]}"
        )

        print("-" * 75)

print("\nFichier créé :")
print(output_file)

print("\nAUCUNE PUBLICATION N'A ETE SUPPRIMEE.")