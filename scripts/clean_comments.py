from pathlib import Path
import pandas as pd
import re
import unicodedata

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")
processed = project / "datasets" / "facebook" / "processed"

input_file = processed / "Post_Comment_Links.xlsx"
output_file = processed / "Data_Comments_Clean.xlsx"

df = pd.read_excel(
    input_file,
    dtype={
        "commentId": str,
        "postId": str
    }
)


# ==========================================================
# 1. Fonctions
# ==========================================================

def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_for_detection(text):
    """
    Version normalisée utilisée uniquement
    pour détecter certains motifs.
    """
    text = safe_text(text).lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_text(text):
    """
    Version destinée au futur traitement NLP.

    On retire les URL et normalise les espaces,
    mais on conserve le contenu linguistique.
    """
    text = safe_text(text)

    if not text:
        return ""

    # Supprimer les URL
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Supprimer certains caractères invisibles
    text = text.replace("\u200b", " ")
    text = text.replace("\xa0", " ")

    # Normaliser les espaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)

    return text.strip()


def contains_url(text):
    text = safe_text(text)

    return bool(
        re.search(
            r"https?://\S+|www\.\S+|wa\.me/\S+",
            text,
            flags=re.IGNORECASE
        )
    )


def detect_potential_spam(text):
    """
    Détection prudente.
    Ce champ indique un SPAM POTENTIEL,
    pas une vérité définitive.
    """

    original = safe_text(text)
    normalized = normalize_for_detection(text)

    if not original:
        return False, ""

    reasons = []

    # WhatsApp / contact direct
    if re.search(
        r"wa\.me/|whatsapp",
        original,
        flags=re.IGNORECASE
    ):
        reasons.append("contact_whatsapp")

    # Numéro international
    if re.search(
        r"\+\d{1,3}[\s\-]?\d{2,}",
        original
    ):
        reasons.append("numero_telephone")

    # Expressions promotionnelles observables
    promotional_patterns = [
        "contactez moi",
        "contactez-moi",
        "ecrivez moi en prive",
        "ecrivez-moi en prive",
        "envoie moi un message en prive",
        "envoyez moi un message",
        "maitre spirituel"
    ]

    if any(
        pattern in normalized
        for pattern in promotional_patterns
    ):
        reasons.append("sollicitation_promotionnelle")

    # Plusieurs URL dans le même commentaire
    urls = re.findall(
        r"https?://\S+|www\.\S+|wa\.me/\S+",
        original,
        flags=re.IGNORECASE
    )

    if len(urls) >= 2:
        reasons.append("plusieurs_urls")

    reasons = list(dict.fromkeys(reasons))

    return len(reasons) > 0, ", ".join(reasons)


# ==========================================================
# 2. Variables de qualité
# ==========================================================

df["originalCommentText"] = df["commentText"].apply(safe_text)

df["isEmpty"] = (
    df["originalCommentText"]
    .str.strip()
    .eq("")
)

df["hasUrl"] = (
    df["originalCommentText"]
    .apply(contains_url)
)

spam_results = (
    df["originalCommentText"]
    .apply(detect_potential_spam)
)

df["isPotentialSpam"] = spam_results.apply(
    lambda x: x[0]
)

df["spamReason"] = spam_results.apply(
    lambda x: x[1]
)


# ==========================================================
# 3. Nettoyage du texte
# ==========================================================

df["cleanCommentText"] = (
    df["originalCommentText"]
    .apply(clean_text)
)


# ==========================================================
# 4. Longueur du commentaire
# ==========================================================

df["cleanLength"] = (
    df["cleanCommentText"]
    .str.len()
)


# ==========================================================
# 5. Commentaire exploitable pour NLP
# ==========================================================

# Pour cette première version :
# - texte non vide
# - pas marqué spam potentiel
# - au moins 2 caractères après nettoyage
#
# Les commentaires courts ("Oui", "Non", etc.)
# restent donc conservés.

df["usableForNLP"] = (
    (~df["isEmpty"])
    & (~df["isPotentialSpam"])
    & (df["cleanLength"] >= 2)
)


# ==========================================================
# 6. Export
# ==========================================================

columns_order = [
    "commentId",
    "postId",
    "status",
    "postUrl",
    "originalCommentText",
    "cleanCommentText",
    "commentDate",
    "commentLikes",
    "isEmpty",
    "hasUrl",
    "isPotentialSpam",
    "spamReason",
    "cleanLength",
    "usableForNLP"
]

existing_columns = [
    col for col in columns_order
    if col in df.columns
]

df[existing_columns].to_excel(
    output_file,
    index=False
)


# ==========================================================
# 7. Rapport
# ==========================================================

print("=" * 75)
print("NETTOYAGE DES COMMENTAIRES")
print("=" * 75)

print(f"Commentaires totaux : {len(df)}")

print(
    "Commentaires vides :",
    df["isEmpty"].sum()
)

print(
    "Commentaires contenant une URL :",
    df["hasUrl"].sum()
)

print(
    "Spam potentiel :",
    df["isPotentialSpam"].sum()
)

print(
    "Commentaires exploitables NLP :",
    df["usableForNLP"].sum()
)

print(
    "Commentaires écartés du NLP :",
    (~df["usableForNLP"]).sum()
)


print("\n" + "=" * 75)
print("RAISONS DE SPAM POTENTIEL")
print("=" * 75)

spam = df[df["isPotentialSpam"]]

if len(spam) == 0:
    print("Aucun spam potentiel détecté.")
else:
    print(
        spam["spamReason"]
        .value_counts()
        .to_string()
    )


print("\n" + "=" * 75)
print("EXEMPLES DE SPAM POTENTIEL")
print("=" * 75)

for _, row in spam.head(20).iterrows():

    print(
        f"\nCommentaire : {row['commentId']}"
    )

    print(
        f"Raison : {row['spamReason']}"
    )

    print(
        row["originalCommentText"][:400]
    )

    print("-" * 75)


print("\n" + "=" * 75)
print("FICHIER CREE")
print("=" * 75)

print(output_file)

print("\nAUCUN COMMENTAIRE ORIGINAL N'A ETE SUPPRIME.")