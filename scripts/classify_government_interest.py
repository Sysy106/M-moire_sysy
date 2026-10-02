import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    r".\datasets\facebook\processed\Data_Posts_Comments_Deduplicated.xlsx"
)

OUTPUT_FILE = Path(
    r".\datasets\facebook\processed\Data_Posts_Auto_Classified.xlsx"
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================
# 2. DESCRIPTION SEMANTIQUE DES DOMAINES
# ============================================================

DOMAINS = {
    "EDUCATION_FORMATION": (
        "Éducation publique, école, enseignement, formation, alphabétisation, "
        "enseignants, élèves, étudiants, rentrée scolaire, apprentissage, "
        "université, infrastructures éducatives et politique éducative."
    ),

    "SANTE": (
        "Santé publique, soins, hôpitaux, maladies, prévention, santé maternelle, "
        "santé communautaire, accès aux soins et politique sanitaire."
    ),

    "SECURITE_DEFENSE": (
        "Sécurité publique, défense nationale, forces de défense et de sécurité, "
        "protection des populations, criminalité et sécurité du territoire."
    ),

    "ADMINISTRATION_GOUVERNANCE": (
        "Administration publique, gouvernement, collectivités territoriales, "
        "services publics, gouvernance, réglementation, institutions publiques "
        "et fonctionnement de l'État."
    ),

    "INFRASTRUCTURES_TRANSPORT": (
        "Routes, autoroutes, transport, mobilité, sécurité routière, bâtiments "
        "publics, infrastructures et aménagement du territoire."
    ),

    "AGRICULTURE_ELEVAGE_ALIMENTATION": (
        "Agriculture, élevage, production agricole, sécurité alimentaire, "
        "agriculture durable, monde rural et développement agricole."
    ),

    "ECONOMIE_INDUSTRIE_EMPLOI": (
        "Économie, industrie, commerce, entreprises, emploi, recrutement, "
        "développement économique, production et politique économique."
    ),

    "SCIENCE_TECHNOLOGIE_INNOVATION": (
        "Recherche scientifique, innovation, technologie, numérique, robotique, "
        "intelligence artificielle, recherche publique et développement scientifique."
    ),

    "PROTECTION_SOCIALE": (
        "Protection sociale, aide aux personnes vulnérables, solidarité, "
        "enfance, inclusion sociale et assistance aux populations."
    ),

    "ENVIRONNEMENT_ENERGIE": (
        "Environnement, changement climatique, énergie, énergie solaire, "
        "ressources naturelles, développement durable et transition énergétique."
    ),

    "COOPERATION_RELATIONS_INTERNATIONALES": (
        "Coopération internationale, relations diplomatiques, partenariats entre "
        "États, organisations internationales et développement international."
    ),

    "CULTURE_SPORT_COHESION": (
        "Culture, sport, cohésion sociale, vivre ensemble, jeunesse, "
        "engagement communautaire et activités collectives d'intérêt public."
    ),
}


# ============================================================
# 3. INDICES TEXTUELS EXPLICABLES
# ============================================================

BURKINA_TERMS = [
    "burkina", "burkinabe", "ouagadougou", "bobo-dioulasso",
    "bobo dioulasso", "faso", "po", "saaba"
]

PUBLIC_ACTION_TERMS = [
    "politique publique", "service public", "projet public",
    "programme national", "acces", "amelioration", "developpement",
    "protection", "prevention", "infrastructure", "education",
    "enseignement", "sante", "securite", "administration",
    "gouvernance", "emploi", "agriculture", "elevage",
    "transport", "recherche", "innovation", "cohesion sociale",
    "securite alimentaire", "collectivite", "population",
    "citoyen", "citoyenne"
]

INSTITUTION_TERMS = [
    "ministere", "ministre", "gouvernement", "presidence",
    "president du faso", "primature", "commune", "administration",
    "direction generale", "secretariat general", "autorite",
    "service public", "etat", "collectivite territoriale"
]

GENERIC_GREETING_TERMS = [
    "excellente journee",
    "bonne journee",
    "bon debut de semaine",
    "bon weekend",
    "bon week-end"
]

DEATH_NOTICE_TERMS = [
    "faire-part de deces",
    "avis de deces",
    "deces de",
    "programme des obseques",
    "inhumation",
    "funerailles"
]

PRIVATE_RECRUITMENT_TERMS = [
    "ecole privee recrute",
    "etablissement prive recrute",
    "recrutement d'enseignants pour une ecole privee"
]


# ============================================================
# 4. NORMALISATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_any(text, terms):
    return any(term in text for term in terms)


# ============================================================
# 5. CHARGEMENT DES DONNEES
# ============================================================

print("=" * 72)
print("CLASSIFICATION AUTOMATIQUE DES PUBLICATIONS FACEBOOK")
print("=" * 72)

df = pd.read_excel(INPUT_FILE)

if "postText" not in df.columns:
    raise ValueError("La colonne 'postText' est absente du fichier.")

df["normalizedText"] = df["postText"].apply(normalize_text)

has_text = df["normalizedText"].str.len() > 0

print("Publications totales :", len(df))
print("Publications avec texte :", has_text.sum())
print("Publications sans texte :", (~has_text).sum())


# ============================================================
# 6. CHARGEMENT DU MODELE SEMANTIQUE
# ============================================================

print()
print("Chargement du modèle sémantique...")
print("Modèle :", MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)

domain_names = list(DOMAINS.keys())
domain_descriptions = list(DOMAINS.values())

domain_embeddings = model.encode(
    domain_descriptions,
    normalize_embeddings=True,
    show_progress_bar=False
)


# ============================================================
# 7. COLONNES DE RESULTAT
# ============================================================

df["semanticScore"] = np.nan
df["burkinaSignal"] = 0
df["publicActionSignal"] = 0
df["institutionSignal"] = 0
df["exclusionSignal"] = 0

df["autoDomain"] = "INDETERMINE"
df["relevanceScore"] = np.nan
df["autoLabel"] = "NON_CLASSIFIABLE"
df["classificationReason"] = ""


# ============================================================
# 8. CLASSIFICATION
# ============================================================

text_indices = df.index[has_text].tolist()
texts = df.loc[text_indices, "normalizedText"].tolist()

print("Calcul des représentations sémantiques...")

text_embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

for pos, idx in enumerate(text_indices):

    text = df.at[idx, "normalizedText"]

    # --------------------------------------------------------
    # Similarité sémantique avec les domaines
    # --------------------------------------------------------

    similarities = np.dot(domain_embeddings, text_embeddings[pos])

    best_position = int(np.argmax(similarities))
    best_domain = domain_names[best_position]
    semantic_score = float(similarities[best_position])

    # --------------------------------------------------------
    # Signaux explicables
    # --------------------------------------------------------

    burkina_signal = int(
        contains_any(text, BURKINA_TERMS)
    )

    public_action_signal = int(
        contains_any(text, PUBLIC_ACTION_TERMS)
    )

    institution_signal = int(
        contains_any(text, INSTITUTION_TERMS)
    )

    generic_greeting = contains_any(
        text, GENERIC_GREETING_TERMS
    )

    death_notice = contains_any(
        text, DEATH_NOTICE_TERMS
    )

    private_recruitment = contains_any(
        text, PRIVATE_RECRUITMENT_TERMS
    )

    exclusion_signal = int(
        generic_greeting or
        death_notice or
        private_recruitment
    )

    # --------------------------------------------------------
    # SCORE HYBRIDE
    #
    # Sémantique       : 70 %
    # Burkina          : 10 %
    # Action publique  : 12 %
    # Institution      :  8 %
    #
    # Une exclusion retire 25 points.
    # --------------------------------------------------------

    semantic_component = max(0.0, min(1.0, semantic_score))

    relevance_score = (
        0.70 * semantic_component
        + 0.10 * burkina_signal
        + 0.12 * public_action_signal
        + 0.08 * institution_signal
        - 0.25 * exclusion_signal
    )

    relevance_score = max(0.0, min(1.0, relevance_score))

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if relevance_score >= 0.55:
        label = "PERTINENT"

    elif relevance_score >= 0.40:
        label = "A_VERIFIER"

    else:
        label = "NON_PERTINENT"

    # --------------------------------------------------------
    # RAISON EXPLICABLE
    # --------------------------------------------------------

    reasons = [
        f"domaine_semantique={best_domain}",
        f"similarite={semantic_score:.3f}"
    ]

    if burkina_signal:
        reasons.append("ancrage_Burkina")

    if public_action_signal:
        reasons.append("enjeu_action_publique")

    if institution_signal:
        reasons.append("contexte_institutionnel")

    if generic_greeting:
        reasons.append("salutation_generique")

    if death_notice:
        reasons.append("avis_deces")

    if private_recruitment:
        reasons.append("recrutement_prive")

    # --------------------------------------------------------
    # ENREGISTREMENT
    # --------------------------------------------------------

    df.at[idx, "semanticScore"] = round(
        semantic_score, 4
    )

    df.at[idx, "burkinaSignal"] = burkina_signal
    df.at[idx, "publicActionSignal"] = public_action_signal
    df.at[idx, "institutionSignal"] = institution_signal
    df.at[idx, "exclusionSignal"] = exclusion_signal

    df.at[idx, "autoDomain"] = best_domain

    df.at[idx, "relevanceScore"] = round(
        relevance_score, 4
    )

    df.at[idx, "autoLabel"] = label

    df.at[idx, "classificationReason"] = "; ".join(reasons)


# ============================================================
# 9. CAS SANS TEXTE
# ============================================================

no_text_mask = ~has_text

df.loc[no_text_mask, "autoLabel"] = "NON_CLASSIFIABLE"
df.loc[no_text_mask, "autoDomain"] = "INDETERMINE"

df.loc[
    no_text_mask,
    "classificationReason"
] = "Texte de publication indisponible"


# ============================================================
# 10. EXPORT
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_excel(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. RAPPORT
# ============================================================

print()
print("=" * 72)
print("RESULTATS")
print("=" * 72)

print()
print("Labels automatiques :")
print(df["autoLabel"].value_counts())

print()
print("Domaines automatiques :")
print(df.loc[
    df["autoLabel"].isin(["PERTINENT", "A_VERIFIER"]),
    "autoDomain"
].value_counts())

print()
print("Score moyen :")
print(
    round(
        df.loc[has_text, "relevanceScore"].mean(),
        3
    )
)

print()
print("Fichier créé :")
print(OUTPUT_FILE.resolve())

print()
print("Classification automatique terminée.")