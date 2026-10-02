import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    r".\datasets\facebook\processed\Data_Posts_Comments_Deduplicated.xlsx"
)

OUTPUT_FILE = Path(
    r".\datasets\facebook\processed\Data_Posts_Auto_Classified_Final.xlsx"
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================
# PROTOTYPES DE PERTINENCE
# ============================================================

POSITIVE_PROTOTYPES = [
    "Cette publication concerne une politique publique.",
    "Cette publication concerne une action ou une responsabilité publique.",
    "Cette publication traite d'un problème collectif affectant la population.",
    "Cette publication concerne un service public ou une infrastructure publique.",
    "Cette publication concerne l'éducation, la santé ou la protection sociale.",
    "Cette publication concerne la sécurité ou la protection des populations.",
    "Cette publication concerne l'agriculture, l'élevage ou la sécurité alimentaire.",
    "Cette publication concerne l'économie, l'industrie ou l'emploi.",
    "Cette publication concerne la recherche scientifique, la technologie ou l'innovation.",
    "Cette publication concerne l'environnement ou l'énergie.",
    "Cette publication concerne l'administration ou la gouvernance.",
    "Cette publication concerne une coopération ayant une portée publique.",
]

NEGATIVE_PROTOTYPES = [
    "Cette publication est une simple salutation.",
    "Cette publication est un message personnel sans enjeu collectif.",
    "Cette publication est un avis de décès familial.",
    "Cette publication est une publicité commerciale.",
    "Cette publication est une offre d'emploi privée.",
    "Cette publication concerne un événement privé sans enjeu public.",
    "Cette publication concerne uniquement un fait divers sans enjeu public.",
]


# ============================================================
# DOMAINES
# ============================================================

DOMAINS = {

    "EDUCATION_FORMATION": [
        "éducation, enseignement, école, université, formation, alphabétisation",
        "élèves, enseignants, étudiants et rentrée scolaire",
        "infrastructures scolaires et universitaires"
    ],

    "SANTE": [
        "santé publique, maladie, prévention et soins",
        "hôpitaux, système sanitaire et accès aux soins",
        "santé reproductive et risques sanitaires"
    ],

    "SECURITE_DEFENSE": [
        "sécurité, défense et protection du territoire",
        "forces de défense et de sécurité",
        "protection des populations"
    ],

    "ADMINISTRATION_GOUVERNANCE": [
        "administration publique et gouvernance",
        "collectivités territoriales et réglementation",
        "fonctionnement des institutions publiques"
    ],

    "INFRASTRUCTURES_TRANSPORT": [
        "routes, autoroutes et transports",
        "sécurité routière et mobilité",
        "infrastructures et aménagement du territoire"
    ],

    "AGRICULTURE_ELEVAGE_ALIMENTATION": [
        "agriculture et développement rural",
        "élevage et protection du bétail",
        "sécurité alimentaire"
    ],

    "ECONOMIE_INDUSTRIE_EMPLOI": [
        "économie et développement économique",
        "industrie et production nationale",
        "emploi et travail"
    ],

    "SCIENCE_TECHNOLOGIE_INNOVATION": [
        "recherche scientifique et innovation",
        "technologie, numérique et robotique",
        "développement scientifique"
    ],

    "PROTECTION_SOCIALE": [
        "protection sociale et solidarité",
        "veuves, orphelins et populations vulnérables",
        "assistance sociale"
    ],

    "ENVIRONNEMENT_ENERGIE": [
        "environnement et climat",
        "énergie et énergie solaire",
        "développement durable"
    ],

    "COOPERATION_RELATIONS_INTERNATIONALES": [
        "coopération internationale entre États",
        "relations diplomatiques avec un pays étranger",
        "organisations internationales et diplomatie"
    ],

    "CULTURE_SPORT_COHESION": [
        "culture et patrimoine",
        "sport et jeunesse",
        "cohésion sociale et activités communautaires"
    ]
}


# ============================================================
# TERMES D'ANCRAGE
# ============================================================

BURKINA_TERMS = [
    "burkina",
    "burkinabe",
    "burkinabè",
    "burukina",
    "faso",
    "ouagadougou",
    "wagadugu",
    "bobo-dioulasso",
    "bobo dioulasso",
    "koudougou",
    "diebugou",
    "diébougou",
    "po ",
    "pô ",
]

PUBLIC_INSTITUTION_TERMS = [
    "ministere",
    "ministère",
    "gouvernement",
    "presidence",
    "présidence",
    "president du faso",
    "président du faso",
    "primature",
    "premier ministre",
    "etat",
    "état",
    "collectivite territoriale",
    "collectivité territoriale",
    "agence nationale",
]

DEATH_TERMS = [
    "deces",
    "décès",
    "faire-part",
    "faire part",
    "obseques",
    "obsèques",
]

GREETING_TERMS = [
    "excellente journee",
    "excellente journée",
    "bonne journee",
    "bonne journée",
]

PRIVATE_RECRUITMENT_TERMS = [
    "complexe scolaire",
    "emploi immediat",
    "emploi immédiat",
    "sous contrat cdd",
]


# ============================================================
# NORMALISATION
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


def normalize_terms(terms):
    return [normalize_text(x) for x in terms]


BURKINA_TERMS = normalize_terms(BURKINA_TERMS)
PUBLIC_INSTITUTION_TERMS = normalize_terms(PUBLIC_INSTITUTION_TERMS)
DEATH_TERMS = normalize_terms(DEATH_TERMS)
GREETING_TERMS = normalize_terms(GREETING_TERMS)
PRIVATE_RECRUITMENT_TERMS = normalize_terms(PRIVATE_RECRUITMENT_TERMS)


def contains_any(text, terms):
    return any(term in text for term in terms)


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 76)
print("CLASSIFICATION AUTOMATIQUE FINALE - V3")
print("=" * 76)

df = pd.read_excel(INPUT_FILE)

df["normalizedText"] = df["postText"].apply(normalize_text)

has_text = df["normalizedText"].str.len() > 0

print("Publications totales :", len(df))
print("Avec texte :", int(has_text.sum()))
print("Sans texte :", int((~has_text).sum()))


# ============================================================
# MODELE
# ============================================================

print("\nChargement du modèle...")

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# EMBEDDINGS PERTINENCE
# ============================================================

positive_embeddings = model.encode(
    POSITIVE_PROTOTYPES,
    normalize_embeddings=True,
    show_progress_bar=False
)

negative_embeddings = model.encode(
    NEGATIVE_PROTOTYPES,
    normalize_embeddings=True,
    show_progress_bar=False
)


# ============================================================
# EMBEDDINGS DOMAINES
# ============================================================

domain_names = list(DOMAINS.keys())
domain_embeddings = []

for domain in domain_names:

    examples = DOMAINS[domain]

    emb = model.encode(
        examples,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    centroid = np.mean(emb, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    domain_embeddings.append(centroid)

domain_embeddings = np.array(domain_embeddings)


# ============================================================
# TEXTES
# ============================================================

text_indices = df.index[has_text].tolist()
texts = df.loc[text_indices, "normalizedText"].tolist()

print("Analyse automatique...")

text_embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# COLONNES
# ============================================================

df["positiveScore"] = np.nan
df["negativeScore"] = np.nan
df["semanticMargin"] = np.nan

df["burkinaSignal"] = 0
df["institutionSignal"] = 0

df["autoLabel"] = "NON_CLASSIFIABLE"

df["autoDomain"] = "INDETERMINE"
df["domainScore"] = np.nan

df["classificationReason"] = ""


# ============================================================
# CLASSIFICATION
# ============================================================

for pos, idx in enumerate(text_indices):

    text = df.at[idx, "normalizedText"]
    vector = text_embeddings[pos]

    # --------------------------------------------------------
    # PERTINENCE SEMANTIQUE
    # --------------------------------------------------------

    positive_sims = np.dot(
        positive_embeddings,
        vector
    )

    negative_sims = np.dot(
        negative_embeddings,
        vector
    )

    positive_score = float(
        np.max(positive_sims)
    )

    negative_score = float(
        np.max(negative_sims)
    )

    margin = positive_score - negative_score

    # --------------------------------------------------------
    # ANCRAGE
    # --------------------------------------------------------

    burkina = int(
        contains_any(text, BURKINA_TERMS)
    )

    institution = int(
        contains_any(
            text,
            PUBLIC_INSTITUTION_TERMS
        )
    )

    # --------------------------------------------------------
    # EXCLUSIONS EXPLICITES
    # --------------------------------------------------------

    death = contains_any(
        text,
        DEATH_TERMS
    )

    greeting = contains_any(
        text,
        GREETING_TERMS
    )

    private_recruitment = contains_any(
        text,
        PRIVATE_RECRUITMENT_TERMS
    )

    # --------------------------------------------------------
    # DECISION DE PERTINENCE
    #
    # Cette décision est indépendante du domaine.
    # --------------------------------------------------------

    if death or greeting or private_recruitment:

        label = "NON_PERTINENT"

    elif not burkina and not institution:

        # Un texte sans ancrage Burkina ni institution
        # doit apporter une évidence sémantique forte.
        if margin >= 0.20:
            label = "A_VERIFIER"
        else:
            label = "NON_PERTINENT"

    else:

        if margin >= 0.08:
            label = "PERTINENT"

        elif margin >= -0.05:
            label = "A_VERIFIER"

        else:
            label = "NON_PERTINENT"

    # --------------------------------------------------------
    # CLASSIFICATION DU DOMAINE
    #
    # Totalement indépendante de la décision précédente.
    # --------------------------------------------------------

    domain_sims = np.dot(
        domain_embeddings,
        vector
    )

    domain_idx = int(
        np.argmax(domain_sims)
    )

    domain = domain_names[domain_idx]

    domain_score = float(
        domain_sims[domain_idx]
    )

    # Domaine peu fiable
    if domain_score < 0.30:
        domain = "AUTRE_OU_INDETERMINE"

    # --------------------------------------------------------
    # ENREGISTREMENT
    # --------------------------------------------------------

    df.at[idx, "positiveScore"] = round(
        positive_score, 4
    )

    df.at[idx, "negativeScore"] = round(
        negative_score, 4
    )

    df.at[idx, "semanticMargin"] = round(
        margin, 4
    )

    df.at[idx, "burkinaSignal"] = burkina
    df.at[idx, "institutionSignal"] = institution

    df.at[idx, "autoLabel"] = label

    df.at[idx, "autoDomain"] = domain

    df.at[idx, "domainScore"] = round(
        domain_score, 4
    )

    reasons = [
        f"positif={positive_score:.3f}",
        f"negatif={negative_score:.3f}",
        f"marge={margin:.3f}",
        f"domaine={domain}",
        f"score_domaine={domain_score:.3f}",
    ]

    if burkina:
        reasons.append("ancrage_Burkina")

    if institution:
        reasons.append("institution_publique")

    if death:
        reasons.append("exclusion_avis_deces")

    if greeting:
        reasons.append("exclusion_salutation")

    if private_recruitment:
        reasons.append(
            "exclusion_recrutement_prive"
        )

    df.at[idx, "classificationReason"] = "; ".join(
        reasons
    )


# ============================================================
# NO TEXT
# ============================================================

df.loc[
    ~has_text,
    "autoLabel"
] = "NON_CLASSIFIABLE"

df.loc[
    ~has_text,
    "autoDomain"
] = "INDETERMINE"

df.loc[
    ~has_text,
    "classificationReason"
] = "Texte de publication indisponible"


# ============================================================
# EXPORT
# ============================================================

df.to_excel(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RAPPORT FINAL
# ============================================================

print("\n" + "=" * 76)
print("RESULTATS V3")
print("=" * 76)

print("\nLabels :")
print(df["autoLabel"].value_counts())

print("\nDomaines :")
print(
    df.loc[
        df["autoLabel"] == "PERTINENT",
        "autoDomain"
    ].value_counts()
)

print("\nContrôles :")

print(
    "Sans texte :",
    int((df["autoLabel"] == "NON_CLASSIFIABLE").sum())
)

print(
    "Pertinents :",
    int((df["autoLabel"] == "PERTINENT").sum())
)

print(
    "À vérifier :",
    int((df["autoLabel"] == "A_VERIFIER").sum())
)

print(
    "Non pertinents :",
    int((df["autoLabel"] == "NON_PERTINENT").sum())
)

print("\nFichier créé :")
print(OUTPUT_FILE.resolve())

print("\nClassification V3 terminée.")