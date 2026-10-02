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
    r".\datasets\facebook\processed\Data_Posts_Auto_Classified_V2.xlsx"
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================
# 2. PROTOTYPES POSITIFS
#    Ils décrivent la notion d'intérêt public/gouvernemental.
# ============================================================

POSITIVE_PROTOTYPES = [
    "Cette publication traite d'une politique publique ou d'une action publique.",
    
    "Cette publication concerne un problème collectif affectant la population "
    "et pouvant relever de l'action publique.",

    "Cette publication concerne une action, une décision, un programme ou "
    "une activité d'une institution publique.",

    "Cette publication concerne l'éducation, la santé, la sécurité, "
    "l'administration ou un autre service d'intérêt public.",

    "Cette publication concerne une infrastructure publique, les transports, "
    "les routes ou l'aménagement du territoire.",

    "Cette publication concerne l'agriculture, l'élevage, la sécurité alimentaire "
    "ou le développement rural dans une perspective d'intérêt collectif.",

    "Cette publication concerne l'économie, l'industrie, l'emploi ou le "
    "développement économique dans une perspective de politique publique.",

    "Cette publication concerne la recherche scientifique, l'innovation, "
    "la technologie ou la souveraineté scientifique.",

    "Cette publication concerne la protection sociale, les populations "
    "vulnérables ou la solidarité publique.",

    "Cette publication concerne l'environnement, l'énergie ou le "
    "développement durable.",

    "Cette publication concerne la coopération internationale ou une relation "
    "entre le Burkina Faso et un partenaire extérieur.",

    "Cette publication concerne une question d'intérêt collectif au Burkina Faso.",

    "Cette publication décrit une réalisation, un investissement ou une "
    "infrastructure ayant une portée publique.",

    "Cette publication concerne la sécurité, la défense, la protection "
    "des populations ou du territoire."
]


# ============================================================
# 3. PROTOTYPES NEGATIFS
# ============================================================

NEGATIVE_PROTOTYPES = [
    "Cette publication est un message personnel sans enjeu public.",

    "Cette publication est une simple salutation ou un message générique "
    "sans information d'intérêt public.",

    "Cette publication est un avis de décès familial ou un faire-part "
    "sans enjeu de politique publique.",

    "Cette publication est une publicité commerciale ou une promotion "
    "privée sans enjeu public.",

    "Cette publication est une offre d'emploi privée sans lien avec "
    "une politique publique.",

    "Cette publication concerne uniquement un événement privé ou personnel.",

    "Cette publication concerne uniquement l'actualité d'un pays étranger "
    "sans lien explicite avec le Burkina Faso.",

    "Cette publication est principalement promotionnelle et ne présente "
    "pas de problème collectif, de service public ou de politique publique."
]


# ============================================================
# 4. DOMAINES
# ============================================================

DOMAINS = {
    "EDUCATION_FORMATION": [
        "éducation publique, école, enseignement, formation, alphabétisation",
        "élèves, enseignants, étudiants, rentrée scolaire et universitaire",
        "universités, écoles et infrastructures éducatives"
    ],

    "SANTE": [
        "santé publique, soins, hôpitaux, maladies et prévention",
        "accès aux soins, santé communautaire et politique sanitaire",
        "santé maternelle, santé reproductive et risques sanitaires"
    ],

    "SECURITE_DEFENSE": [
        "sécurité publique, défense nationale et protection des populations",
        "forces de défense et de sécurité et sécurité du territoire",
        "prévention des risques et protection des citoyens"
    ],

    "ADMINISTRATION_GOUVERNANCE": [
        "administration publique, gouvernement et services publics",
        "gouvernance et fonctionnement des institutions publiques",
        "collectivités territoriales, réglementation et administration locale"
    ],

    "INFRASTRUCTURES_TRANSPORT": [
        "routes, autoroutes, transport et mobilité",
        "sécurité routière et infrastructures de transport",
        "infrastructures publiques, bâtiments et aménagement du territoire"
    ],

    "AGRICULTURE_ELEVAGE_ALIMENTATION": [
        "agriculture, élevage et développement rural",
        "sécurité alimentaire et production agricole",
        "élevage, identification du bétail et protection des ressources animales"
    ],

    "ECONOMIE_INDUSTRIE_EMPLOI": [
        "économie, industrie et développement économique",
        "emploi, travail et politique économique",
        "industrialisation, entreprises et production nationale"
    ],

    "SCIENCE_TECHNOLOGIE_INNOVATION": [
        "recherche scientifique, innovation et technologie",
        "numérique, robotique, informatique et nouvelles technologies",
        "développement scientifique et souveraineté technologique"
    ],

    "PROTECTION_SOCIALE": [
        "protection sociale et solidarité publique",
        "aide aux veuves, orphelins et personnes vulnérables",
        "assistance sociale, inclusion et soutien aux populations"
    ],

    "ENVIRONNEMENT_ENERGIE": [
        "environnement, climat et développement durable",
        "énergie solaire, énergie et transition énergétique",
        "ressources naturelles et protection de l'environnement"
    ],

    "COOPERATION_RELATIONS_INTERNATIONALES": [
        "coopération internationale et relations diplomatiques",
        "relations entre le Burkina Faso et les pays étrangers",
        "organisations internationales et partenariats internationaux"
    ],

    "CULTURE_SPORT_COHESION": [
        "culture, sport et cohésion sociale",
        "jeunesse, activités sportives et engagement communautaire",
        "vivre ensemble, culture et activités collectives"
    ]
}


# ============================================================
# 5. NORMALISATION
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


# ============================================================
# 6. REGLES EXPLICABLES COMPLEMENTAIRES
# ============================================================

BURKINA_TERMS = [
    "burkina", "burkinabe", "burukina", "faso",
    "ouagadougou", "wagadugu", "bobo-dioulasso",
    "bobo dioulasso"
]

PUBLIC_INSTITUTION_TERMS = [
    "ministere", "gouvernement", "presidence",
    "president du faso", "primature",
    "collectivite territoriale", "commune",
    "administration publique", "etat",
    "agence nationale", "direction generale"
]

PUBLIC_ACTION_TERMS = [
    "politique publique", "programme national",
    "service public", "infrastructure",
    "autoroute", "route", "securite routiere",
    "securite alimentaire", "protection sociale",
    "sante publique", "rentrée administrative",
    "rentree administrative", "enseignement",
    "alphabetisation", "universite",
    "recherche scientifique", "raffinerie",
    "identification electronique",
    "brigades routieres", "soutien de l'etat"
]

DEATH_TERMS = [
    "avis de deces", "faire-part",
    "faire part", "funerailles",
    "inhumation", "obseques"
]

GREETING_TERMS = [
    "excellente journee",
    "bonne journee",
    "bon weekend",
    "bon week-end"
]


def contains_any(text, terms):
    return any(term in text for term in terms)


# ============================================================
# 7. CHARGEMENT
# ============================================================

print("=" * 76)
print("CLASSIFICATION AUTOMATIQUE V2")
print("=" * 76)

df = pd.read_excel(INPUT_FILE)

if "postText" not in df.columns:
    raise ValueError("Colonne 'postText' absente.")

df["normalizedText"] = df["postText"].apply(normalize_text)

has_text = df["normalizedText"].str.len() > 0

print("Publications totales :", len(df))
print("Avec texte :", int(has_text.sum()))
print("Sans texte :", int((~has_text).sum()))


# ============================================================
# 8. MODELE
# ============================================================

print("\nChargement du modèle...")
model = SentenceTransformer(MODEL_NAME)


# ============================================================
# 9. ENCODAGE DES PROTOTYPES
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


# Un prototype de domaine = moyenne des exemples du domaine.

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

    # renormalisation du centroïde
    centroid = centroid / np.linalg.norm(centroid)

    domain_embeddings.append(centroid)

domain_embeddings = np.array(domain_embeddings)


# ============================================================
# 10. ENCODAGE DES PUBLICATIONS
# ============================================================

text_indices = df.index[has_text].tolist()
texts = df.loc[text_indices, "normalizedText"].tolist()

print("Analyse sémantique des publications...")

text_embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# 11. COLONNES RESULTATS
# ============================================================

df["positiveSemanticScore"] = np.nan
df["negativeSemanticScore"] = np.nan
df["semanticMargin"] = np.nan

df["bestPositivePrototype"] = ""
df["bestNegativePrototype"] = ""

df["burkinaSignal"] = 0
df["institutionSignal"] = 0
df["publicActionSignal"] = 0

df["autoDomain"] = "INDETERMINE"
df["domainScore"] = np.nan

df["decisionScore"] = np.nan
df["autoLabel"] = "NON_CLASSIFIABLE"
df["classificationReason"] = ""


# ============================================================
# 12. CLASSIFICATION
# ============================================================

for pos, idx in enumerate(text_indices):

    text = df.at[idx, "normalizedText"]
    vector = text_embeddings[pos]

    # -------------------------
    # Pertinence positive
    # -------------------------

    positive_similarities = np.dot(
        positive_embeddings,
        vector
    )

    best_positive_idx = int(
        np.argmax(positive_similarities)
    )

    positive_score = float(
        positive_similarities[best_positive_idx]
    )

    # -------------------------
    # Pertinence négative
    # -------------------------

    negative_similarities = np.dot(
        negative_embeddings,
        vector
    )

    best_negative_idx = int(
        np.argmax(negative_similarities)
    )

    negative_score = float(
        negative_similarities[best_negative_idx]
    )

    margin = positive_score - negative_score

    # -------------------------
    # Domaine
    # -------------------------

    domain_similarities = np.dot(
        domain_embeddings,
        vector
    )

    best_domain_idx = int(
        np.argmax(domain_similarities)
    )

    best_domain = domain_names[best_domain_idx]

    domain_score = float(
        domain_similarities[best_domain_idx]
    )

    # -------------------------
    # Signaux explicables
    # -------------------------

    burkina_signal = int(
        contains_any(text, BURKINA_TERMS)
    )

    institution_signal = int(
        contains_any(text, PUBLIC_INSTITUTION_TERMS)
    )

    public_action_signal = int(
        contains_any(text, PUBLIC_ACTION_TERMS)
    )

    death_signal = contains_any(
        text, DEATH_TERMS
    )

    greeting_signal = contains_any(
        text, GREETING_TERMS
    )

    # -------------------------
    # Score de décision
    #
    # Le coeur de la décision est la
    # différence positif - négatif.
    # Les règles ne jouent qu'un rôle
    # complémentaire.
    # -------------------------

    decision_score = (
        margin
        + 0.05 * burkina_signal
        + 0.07 * institution_signal
        + 0.08 * public_action_signal
    )

    # Les deux cas ci-dessous sont
    # suffisamment spécifiques pour
    # constituer des exclusions fortes.

    if death_signal:
        decision_score -= 0.30

    if greeting_signal:
        decision_score -= 0.30

    # -------------------------
    # Décision automatique V2
    #
    # Valeurs expérimentales :
    # elles seront évaluées après
    # observation de la distribution.
    # -------------------------

    if decision_score >= 0.12:
        label = "PERTINENT"

    elif decision_score >= -0.02:
        label = "A_VERIFIER"

    else:
        label = "NON_PERTINENT"

    # -------------------------
    # Enregistrement
    # -------------------------

    df.at[idx, "positiveSemanticScore"] = round(
        positive_score, 4
    )

    df.at[idx, "negativeSemanticScore"] = round(
        negative_score, 4
    )

    df.at[idx, "semanticMargin"] = round(
        margin, 4
    )

    df.at[idx, "bestPositivePrototype"] = (
        POSITIVE_PROTOTYPES[best_positive_idx]
    )

    df.at[idx, "bestNegativePrototype"] = (
        NEGATIVE_PROTOTYPES[best_negative_idx]
    )

    df.at[idx, "burkinaSignal"] = burkina_signal
    df.at[idx, "institutionSignal"] = institution_signal
    df.at[idx, "publicActionSignal"] = public_action_signal

    df.at[idx, "autoDomain"] = best_domain
    df.at[idx, "domainScore"] = round(domain_score, 4)

    df.at[idx, "decisionScore"] = round(
        decision_score, 4
    )

    df.at[idx, "autoLabel"] = label

    reason = (
        f"positif={positive_score:.3f}; "
        f"negatif={negative_score:.3f}; "
        f"marge={margin:.3f}; "
        f"domaine={best_domain}; "
        f"score_domaine={domain_score:.3f}"
    )

    if burkina_signal:
        reason += "; ancrage_Burkina"

    if institution_signal:
        reason += "; institution_publique"

    if public_action_signal:
        reason += "; action_publique"

    if death_signal:
        reason += "; exclusion_avis_deces"

    if greeting_signal:
        reason += "; exclusion_salutation"

    df.at[idx, "classificationReason"] = reason


# ============================================================
# 13. PUBLICATIONS SANS TEXTE
# ============================================================

no_text = ~has_text

df.loc[no_text, "autoLabel"] = "NON_CLASSIFIABLE"
df.loc[no_text, "autoDomain"] = "INDETERMINE"
df.loc[
    no_text,
    "classificationReason"
] = "Texte de publication indisponible"


# ============================================================
# 14. EXPORT
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
# 15. RAPPORT
# ============================================================

print("\n" + "=" * 76)
print("RESULTATS V2")
print("=" * 76)

print("\nLabels :")
print(df["autoLabel"].value_counts())

print("\nStatistiques du score de décision :")

scores = df.loc[
    has_text,
    "decisionScore"
]

print(scores.describe())

print("\nDomaines des publications PERTINENT/A_VERIFIER :")

print(
    df.loc[
        df["autoLabel"].isin(
            ["PERTINENT", "A_VERIFIER"]
        ),
        "autoDomain"
    ].value_counts()
)

print("\nFichier créé :")
print(OUTPUT_FILE.resolve())

print("\nClassification V2 terminée.")