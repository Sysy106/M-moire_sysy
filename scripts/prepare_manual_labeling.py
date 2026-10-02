from pathlib import Path
import pandas as pd

project = Path(r"C:\Users\ouedr\Desktop\memoire_priorisation")
processed = project / "datasets" / "facebook" / "processed"

input_file = (
    processed /
    "Data_Posts_Comments_Deduplicated.xlsx"
)

output_file = (
    processed /
    "Data_Posts_Manual_Labeling.xlsx"
)

df = pd.read_excel(
    input_file,
    dtype={"postId": str}
)


# ==========================================================
# 1. Colonnes utiles pour la labellisation
# ==========================================================

columns = [
    "postId",
    "postUrl",
    "pageName",
    "sourceType",
    "createdAt",
    "postText",
    "collectedCommentCount",
    "usableCommentCount",
    "availableTextSource"
]

columns = [
    col for col in columns
    if col in df.columns
]

labeling = df[columns].copy()


# ==========================================================
# 2. Ajouter les colonnes de vérité terrain
# ==========================================================

labeling["label_manual"] = ""
labeling["domain_manual"] = ""
labeling["justification_manual"] = ""


# ==========================================================
# 3. Les lignes sans texte sont non classifiables
# ==========================================================

no_text = (
    labeling["availableTextSource"] == "NO_TEXT"
)

labeling.loc[
    no_text,
    "label_manual"
] = "NON_CLASSIFIABLE"

labeling.loc[
    no_text,
    "justification_manual"
] = "Texte de publication indisponible"


# ==========================================================
# 4. Numéro pour faciliter la revue
# ==========================================================

labeling.insert(
    0,
    "reviewNumber",
    range(1, len(labeling) + 1)
)


# ==========================================================
# 5. Export
# ==========================================================

labeling.to_excel(
    output_file,
    index=False
)


# ==========================================================
# 6. Rapport
# ==========================================================

print("=" * 75)
print("PREPARATION DE LA LABELLISATION MANUELLE")
print("=" * 75)

print(
    f"Publications totales : {len(labeling)}"
)

print(
    "Publications à examiner :",
    (
        labeling["label_manual"] == ""
    ).sum()
)

print(
    "Non classifiables (NO_TEXT) :",
    (
        labeling["label_manual"]
        == "NON_CLASSIFIABLE"
    ).sum()
)

print("\nValeurs autorisées pour label_manual :")
print("PERTINENT")
print("NON_PERTINENT")
print("A_VERIFIER")
print("NON_CLASSIFIABLE")

print("\nExemples de domaines :")
print(
    "education, sante, securite, infrastructures, "
    "agriculture_elevage, economie_emploi, "
    "administration, environnement_energie, "
    "recherche_innovation, protection_sociale, "
    "cooperation, autre"
)

print("\nFichier créé :")
print(output_file)