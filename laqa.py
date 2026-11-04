"""
LAQA: Language-Aware Query Analyzer (Optimized Version)
Module for Cancer Information System Using RAG, LLMs, LAQA and MCP.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

CANCER_TYPE_MAP = {
    # Head & Neck / Oral / Pharynx / Larynx
    "larynx": "Laryngeal Cancer",
    "laryngeal": "Laryngeal Cancer",
    "glottic": "Laryngeal Cancer",
    "glottis": "Laryngeal Cancer",
    "supraglottic": "Laryngeal Cancer",
    "subglottic": "Laryngeal Cancer",
    "head and neck": "Head and Neck Cancer",
    "oral": "Oral Cavity Cancer",
    "buccal": "Buccal Mucosa Cancer",
    "parotid": "Salivary Gland Cancer",
    "salivary": "Salivary Gland Cancer",
    "nasopharyngeal": "Nasopharyngeal Cancer",
    "nasopharynx": "Nasopharyngeal Cancer",
    "sinonasal": "Sinonasal Cancer",
    "oropharyngeal": "Oropharyngeal Cancer",
    "oropharynx": "Oropharyngeal Cancer",

    # Breast
    "breast": "Breast Cancer",
    "mammary": "Breast Cancer",

    # Skin / Melanoma
    "melanoma": "Malignant Melanoma",
    "skin": "Skin Cancer",
    "squamous cell carcinoma": "Squamous Cell Carcinoma",
    "scc": "Squamous Cell Carcinoma",
    "basal cell": "Basal Cell Carcinoma",
    "merkel cell": "Merkel Cell Carcinoma",
    "xeroderma pigmentosum": "Skin Cancer (XP-Associated)",

    # Thoracic / Lung
    "lung": "Lung Cancer",
    "nsclc": "Non-Small Cell Lung Cancer",
    "sclc": "Small Cell Lung Cancer",
    "thymoma": "Thymoma & Thymic Carcinoma",
    "mesothelioma": "Mesothelioma",
    "pancoast": "Lung Cancer (Pancoast Tumor)",

    # Gastrointestinal
    "gastrointestinal stromal": "Gastrointestinal Stromal Tumor (GIST)",
    "gist": "Gastrointestinal Stromal Tumor (GIST)",
    "esophageal": "Esophageal Cancer",
    "esophagus": "Esophageal Cancer",
    "gastric": "Gastric (Stomach) Cancer",
    "stomach": "Gastric (Stomach) Cancer",
    "colon": "Colorectal Cancer",
    "colorectal": "Colorectal Cancer",
    "rectal": "Colorectal Cancer",
    "appendix": "Appendiceal Cancer",
    "liver": "Liver Cancer (Hepatocellular Carcinoma)",
    "hepatocellular": "Liver Cancer (Hepatocellular Carcinoma)",
    "cholangiocarcinoma": "Cholangiocarcinoma (Bile Duct Cancer)",
    "pancreatic": "Pancreatic Cancer",

    # Genitourinary
    "renal cell": "Renal Cell Carcinoma",
    "kidney": "Renal Cell Carcinoma",
    "bladder": "Bladder Cancer",
    "urothelial": "Urothelial Carcinoma",
    "prostate": "Prostate Cancer",
    "testicular": "Testicular Cancer",
    "testis": "Testicular Cancer",

    # Gynecological
    "cervical": "Cervical Cancer",
    "cervix": "Cervical Cancer",
    "ovarian": "Ovarian Cancer",
    "ovary": "Ovarian Cancer",
    "uterine": "Uterine (Endometrial) Cancer",
    "endometrial": "Uterine (Endometrial) Cancer",
    "vaginal": "Vaginal Cancer",
    "vagina": "Vaginal Cancer",
    "vulvar": "Vulval (Vulvar) Cancer",
    "vulva": "Vulval (Vulvar) Cancer",
    "fallopian": "Fallopian Tube Cancer",
    "gestational trophoblastic": "Gestational Trophoblastic Disease (GTD)",
    "choriocarcinoma": "Gestational Trophoblastic Disease (GTD)",
    "hydatidiform mole": "Gestational Trophoblastic Disease (GTD)",
    "gtd": "Gestational Trophoblastic Disease (GTD)",

    # CNS / Brain / Spine
    "brain": "Central Nervous System (Brain) Tumor",
    "cns": "Central Nervous System (Brain) Tumor",
    "glioma": "Central Nervous System (Brain) Tumor",
    "spinal": "Spinal Cord / Spine Tumor",
    "pituitary": "Pituitary Tumor",

    # Endocrine
    "thyroid": "Thyroid Cancer",
    "adrenocortical": "Adrenocortical Carcinoma",

    # Sarcoma & Bone
    "sarcoma": "Sarcoma (Soft Tissue & Bone)",
    "rhabdomyosarcoma": "Rhabdomyosarcoma",
    "osteosarcoma": "Osteosarcoma",
    "ewing": "Ewing Sarcoma",
    "angiosarcoma": "Angiosarcoma",
    "lymphangiosarcoma": "Lymphangiosarcoma",

    # Pediatric Specific
    "wilms": "Wilms Tumor (Nephroblastoma)",
    "neuroblastoma": "Neuroblastoma",
    "retinoblastoma": "Retinoblastoma",

    # Hematological
    "leukemia": "Leukemia",
    "lymphoma": "Lymphoma",
    "dlbcl": "Diffuse Large B-Cell Lymphoma",
    "cml": "Chronic Myeloid Leukemia",
    "all": "Acute Lymphoblastic Leukemia"
}

CATEGORY_REGEX_PATTERNS = {
    "treatment": [
        r"\btreat(ing|ment|ed|s)?\b", r"\btherap(y|ies|eutic)\b", r"\bchemo(therapy)?\b",
        r"\bradiotherap(y|ies)\b", r"\bradiation\b", r"\bsurg(ery|ical|eon|eries)\b",
        r"\bresect(ion|ed|ing|ability)?\b", r"\bexcis(ion|ed|ing)?\b", r"\bmastectom(y|ies)\b",
        r"\bregimen(s)?\b", r"\bcure(s|d|ative)?\b", r"\bpalliat(ive|ion|e|ed|ing)?\b",
        r"\bneoadjuvant\b", r"\badjuvant\b", r"\bdrug(s)?\b", r"\bdose(s)?\b", r"\bdosage(s)?\b",
        r"\bmaintenance\b", r"\binhibitor(s)?\b", r"\btarget(ed|ing)?\b", r"\bimmunotherap(y|ies)\b",
        r"\borchiectom(y|ies)\b", r"\bflot\b", r"\bvac\b", r"\br-chop\b", r"\br-fc\b", r"\bema-co\b",
        r"\bpostpone(d|ment)?\b", r"\blocal(ized)? therapy\b", r"\bconsolidation\b"
    ],
    "diagnosis": [
        r"\bdiagnos(is|tic|ed|e|ing)?\b", r"\bsymptom(s|atic)?\b", r"\bpresent(ing|ation|s|ed)?\b",
        r"\bsign(s)?\b", r"\babcde(s)?\b", r"\btriad\b", r"\blesion(s)?\b", r"\bpain\b",
        r"\bparalysis\b", r"\bpancoast\b", r"\bparaneoplastic\b", r"\bmyasthenia\b", r"\bhematemesis\b",
        r"\bmelena\b", r"\bmass(es)?\b", r"\bplaques\b", r"\bleukoplakia\b", r"\bheadache(s)?\b",
        r"\bhematuria\b", r"\bparalysis\b", r"\bparotid\b", r"\banatom(y|ical|ic)\b", r"\bdivisions?\b",
        r"\bfeatures?\b", r"\bphysical sign\b"
    ],
    "biomarker": [
        r"\bbiomarker(s)?\b", r"\bmarker(s)?\b", r"\bgene(s)?\b", r"\bmutation(s)?\b",
        r"\breceptor(s)?\b", r"\bc-kit\b", r"\bbrca(1|2)?\b", r"\bca-?125\b", r"\bcea\b",
        r"\bpsa\b", r"\bher2(/neu)?\b", r"\bpd-l1\b", r"\begfr\b", r"\bbraf(\s*v600)?\b",
        r"\bp53\b", r"\btp53\b", r"\bret\b", r"\bn-myc\b", r"\bmycn\b", r"\bbcr/abl\b",
        r"\bchromosome(s)?\b", r"\btranslocation(s)?\b", r"\bamplif(ication|ied)?\b", r"\ballele(s)?\b",
        r"\bphiladelphia\b", r"\bhep-par-1\b", r"\bs100\b", r"\bsox10\b", r"\bca\s*19-9\b",
        r"\bca\s*27\.29\b", r"\bca\s*15-3\b"
    ],
    "staging": [
        r"\bstage(s|d|ing)?\b", r"\btnm\b", r"\bajcc\b", r"\bfigo\b", r"\bgroup\s+[i|v|x]+\b",
        r"\breese-ellsworth\b", r"\bextent\b", r"\blymph node\b", r"\bdiameter\b", r"\bsize\b",
        r"\bclassification\b"
    ],
    "prognosis": [
        r"\bprognos(is|tic)?\b", r"\bsurviv(al|e|ing|or)?\b", r"\b5-year\b", r"\brecurren(ce|t)?\b",
        r"\bdeath\b", r"\boutcome(s)?\b", r"\bworse\b", r"\bbetter\b", r"\bdisease free survival\b",
        r"\bdfs\b", r"\bdisease-free interval\b", r"\bdfi\b", r"\bweight loss\b", r"\bglasgow\b",
        r"\bgrave\b", r"\bpredictor(s)?\b"
    ],
    "epidemiology": [
        r"\brisk factor(s)?\b", r"\bcause(s|d|ing)?\b", r"\binciden(ce|t)?\b", r"\bratio\b",
        r"\bepidemiolog(y|ic|ical)?\b", r"\bdiet(ary)?\b", r"\btobacco\b", r"\balcohol\b",
        r"\bsmoking\b", r"\basbestos\b", r"\bvirus\b", r"\bebv\b", r"\bhpv\b",
        r"\bprevalen(ce|t)?\b", r"\brate(s)?\b", r"\benvironment(al)?\b", r"\bexposure(s)?\b",
        r"\bdisparit(y|ies)\b", r"\blmic(s)?\b", r"\bxeroderma\b", r"\bcongenital\b",
        r"\bwho goal\b", r"\beliminat(ing|ion)?\b"
    ],
    "investigation": [
        r"\bimag(ing|e)?\b", r"\bmri\b", r"\bct\s*(scan)?\b", r"\bultrasound\b", r"\bultrasonograph(y)?\b",
        r"\bpet(-ct)?\b", r"\bbiops(y|ies)\b", r"\bcolonoscop(y|ies)\b", r"\bradiograph\b",
        r"\bliquid biopsy\b", r"\bscreening\b", r"\btest(s|ing)?\b", r"\bendorectal\b",
        r"\bevaluation\b", r"\bplain films\b", r"\bradiographic\b"
    ],
    "pathology": [
        r"\bpatholog(y|ical|ic)?\b", r"\bhistolog(y|ic|ical)?\b", r"\bdifferentiat(ion|ed)?\b",
        r"\bgrade(s|d)?\b", r"\bfuhrman\b", r"\bsarcomatous\b", r"\bkrukenberg\b",
        r"\blcis\b", r"\bcis\b", r"\boncocytoma\b", r"\bnecrosis\b", r"\bmitotic\b"
    ],
    "mechanism": [
        r"\bmechanism(s)?\b", r"\bpathogenesis\b", r"\bmolecular\b", r"\bdna\b",
        r"\btelomerase\b", r"\bcheckpoint(s)?\b", r"\bcarcinogenesis\b", r"\bsporadic\b",
        r"\balkylating\b", r"\bapoptosis\b", r"\bt cell(s)?\b", r"\bimmortal(ity)?\b"
    ],
    "side_effects": [
        r"\bside effect(s)?\b", r"\badverse\b", r"\btoxicit(y|ies)\b", r"\btoxic\b",
        r"\bcomplication(s)?\b", r"\blate effect(s)?\b", r"\bfrey syndrome\b",
        r"\birae(s)?\b", r"\bcolitis\b", r"\bhypothyroidism\b", r"\bcritical organs?\b"
    ],
    "general": [
        r"\bgeneral\b", r"\bobjective(s)?\b", r"\bguideline(s)?\b", r"\bdefinition\b",
        r"\boverview\b", r"\bwho\b", r"\basco\b", r"\bnccn\b", r"\bicmr\b", r"\baya\b",
        r"\badvance directive\b", r"\bdistress\b", r"\bwhat is\b"
    ]
}

BOILERPLATE_PATTERNS = [
    r"^what (is|are|was|were|do|does|can|could|should|would) the\b",
    r"^what (is|are|was|were|do|does|can|could|should|would)\b",
    r"^in (what|which) percentage of\b",
    r"^which (specific|imaging|type of|threshold of|congenital|genetic|biological|fda-approved)\b",
    r"^which\b",
    r"^why (is|are|do|does)\b",
    r"^how (is|are|does|do|often|quickly|common)\b",
    r"^when (is|are|was|were)\b",
    r"^according to (the )?(indian national guidelines|csco guidelines|asco|nccn|figo|who)[,\s]*",
    r"^can you (tell me|explain)\b",
    r"\?+$"
]

@dataclass
class LAQAOutput:
    original_query: str
    language: str
    cancer_type: Optional[str]
    question_category: str
    normalized_query: str
    extracted_entities: Dict[str, List[str]]
    metadata_filters: Dict[str, Any]
    needs_clarification: bool
    clarification_reason: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not self.needs_clarification:
            d.pop("clarification_reason", None)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class LanguageAwareQueryAnalyzer:
    """
    LAQA Engine:
    Processes natural-language cancer questions into a validated, structured representation.
    """

    def __init__(self, supported_languages: Optional[List[str]] = None):
        self.supported_languages = supported_languages or ["English", "en"]

    def detect_language(self, query: str) -> str:
        if not query or not query.strip():
            return "unknown"

        if re.search(r"[\u0900-\u097F]", query):
            return "Hindi"
        if re.search(r"[\u0B80-\u0BFF]", query):
            return "Tamil"
        if re.search(r"[\u4E00-\u9FFF]", query):
            return "Chinese"
        if re.search(r"[\u0600-\u06FF]", query):
            return "Arabic"

        q_lower = query.lower()
        if any(w in q_lower.split() for w in ["qué", "cómo", "cuál", "tratamiento", "cáncer"]):
            return "Spanish"
        if any(w in q_lower.split() for w in ["quel", "quels", "traitement", "pourquoi"]):
            return "French"
        if any(w in q_lower.split() for w in ["was", "wie", "behandlung", "warum"]):
            return "German"

        return "English"

    def extract_cancer_type(self, query: str) -> Optional[str]:
        q_lower = query.lower()
        sorted_keys = sorted(CANCER_TYPE_MAP.keys(), key=len, reverse=True)
        for k in sorted_keys:
            pattern = r"\b" + re.escape(k) + r"\b"
            if re.search(pattern, q_lower):
                return CANCER_TYPE_MAP[k]
        return None

    def classify_category(self, query: str) -> str:
        q_lower = query.lower()
        scores: Dict[str, int] = {cat: 0 for cat in CATEGORY_REGEX_PATTERNS}

        for cat, patterns in CATEGORY_REGEX_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, q_lower):
                    scores[cat] += 2

        # Break ties prioritizing specific clinical categories over general
        priority_order = [
            "biomarker", "staging", "prognosis", "treatment",
            "side_effects", "pathology", "mechanism", "investigation",
            "epidemiology", "diagnosis", "general"
        ]

        max_score = max(scores.values())
        if max_score > 0:
            for cat in priority_order:
                if scores[cat] == max_score:
                    return cat

        return "general"

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {
            "drugs": [],
            "biomarkers_genes": [],
            "stages": [],
            "anatomical_sites": []
        }
        q_lower = query.lower()

        drug_vocab = [
            "cisplatin", "carboplatin", "paclitaxel", "docetaxel", "5-fluorouracil", "5-fu",
            "pembrolizumab", "nivolumab", "ipilimumab", "cetuximab", "bevacizumab", "trastuzumab",
            "olaparib", "niraparib", "rucaparib", "gefitinib", "erlotinib", "afatinib", "osimertinib",
            "vemurafenib", "dabrafenib", "methotrexate", "actinomycin d", "dactinomycin", "cyclophosphamide",
            "vincristine", "etoposide", "bleomycin", "doxorubicin", "liposomal doxorubicin", "gemcitabine", "topotecan",
            "tamoxifen", "capecitabine", "sacituzumab govitecan", "tremelimumab", "cyclosporine",
            "dostarlimab", "mirvetuximab soravtansine", "tisotumab vedotin", "ifosfamide", "mesna",
            "megestrol acetate", "medroxyprogesterone", "letrozole", "anastrozole", "fulvestrant",
            "flot", "vac", "r-chop", "r-fc", "ema-co", "bep", "tip", "l-asparaginase", "rituximab"
        ]
        for d in drug_vocab:
            if re.search(r"\b" + re.escape(d) + r"\b", q_lower):
                entities["drugs"].append(d.upper() if len(d) <= 6 else d.title())

        biomarker_vocab = [
            "c-kit", "brca", "brca1", "brca2", "her2", "her2/neu", "egfr", "braf", "braf v600",
            "p53", "tp53", "ret", "n-myc", "mycn", "bcr/abl", "ca-125", "ca 125", "cea",
            "ca 19-9", "ca 27.29", "ca 15-3", "psa", "afp", "beta-hcg", "ldh", "pd-l1", "ctla-4",
            "hep-par-1", "s100", "sox10", "pole", "mmr", "dmmr", "msi-h", "hrd", "homologous recombination deficiency",
            "folr1", "fr-alpha", "he4", "p16", "ki-67", "wt1", "er", "pr", "estrogen receptor", "progesterone receptor",
            "philadelphia chromosome", "chromosome 18q", "t(x;18)", "t(9;22)"
        ]
        for b in biomarker_vocab:
            if re.search(r"\b" + re.escape(b) + r"\b", q_lower):
                entities["biomarkers_genes"].append(b.upper() if len(b) <= 6 else b.title())

        stage_matches = re.findall(
            r"\b(?:FIGO\s+Stage\s+[I|V|X]+[A-C]?[1-2]?|Stage\s+[I|V|X]+[A-C]?[1-2]?|Stage\s+[1-4][A-C]?[1-2]?|T[0-4]N[0-3]M[0-1]|Group\s+[I|V|X]+|Early\s+Stage|Locally\s+Advanced|Metastatic|Recurrent)\b",
            query,
            re.IGNORECASE
        )
        if stage_matches:
            entities["stages"] = list(dict.fromkeys([s.title() for s in stage_matches]))

        anatomy_vocab = [
            "larynx", "supraglottis", "glottis", "subglottis", "parotid", "cavernous sinus",
            "orbit", "parietal lobe", "mediastinal", "axillary", "spinal", "colon", "rectum",
            "ovary", "ovaries", "cervix", "uterus", "endometrium", "myometrium", "breast", "thyroid",
            "adrenal", "testis", "appendix", "fimbria", "fallopian", "peritoneum", "vulva", "vagina",
            "esophagus", "stomach", "pelvic", "para-aortic", "groin", "inguinal"
        ]
        for a in anatomy_vocab:
            if re.search(r"\b" + re.escape(a) + r"\b", q_lower):
                entities["anatomical_sites"].append(a.title())

        return entities

    def normalize_query(self, query: str) -> str:
        normalized = query.strip()
        for pattern in BOILERPLATE_PATTERNS:
            normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE).strip()

        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized:
            normalized = normalized[0].upper() + normalized[1:]
        else:
            normalized = query.strip()
        return normalized

    def build_metadata_filters(self, cancer_type: Optional[str], category: str) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if cancer_type:
            filters["cancer_type"] = cancer_type
        return filters

    def analyze(self, query: str) -> LAQAOutput:
        clean_q = query.strip() if query else ""
        lang = self.detect_language(clean_q)

        if not clean_q or len(clean_q.split()) < 2:
            return LAQAOutput(
                original_query=clean_q,
                language=lang,
                cancer_type=None,
                question_category="general",
                normalized_query=clean_q,
                extracted_entities={"drugs": [], "biomarkers_genes": [], "stages": [], "anatomical_sites": []},
                metadata_filters={},
                needs_clarification=True,
                clarification_reason="Query is too short or lacks sufficient medical context to analyze.",
                confidence=0.0
            )

        if lang != "English":
            return LAQAOutput(
                original_query=clean_q,
                language=lang,
                cancer_type=None,
                question_category="general",
                normalized_query=clean_q,
                extracted_entities={"drugs": [], "biomarkers_genes": [], "stages": [], "anatomical_sites": []},
                metadata_filters={},
                needs_clarification=True,
                clarification_reason=f"Language '{lang}' detected. Currently, English is the primary verified language for retrieval.",
                confidence=0.4
            )

        cancer_type = self.extract_cancer_type(clean_q)
        category = self.classify_category(clean_q)
        entities = self.extract_entities(clean_q)
        normalized_q = self.normalize_query(clean_q)
        metadata_filters = self.build_metadata_filters(cancer_type, category)

        has_entities = any(bool(v) for v in entities.values())
        confidence = 0.95 if (cancer_type and has_entities) else (0.85 if cancer_type or has_entities else 0.70)

        return LAQAOutput(
            original_query=clean_q,
            language=lang,
            cancer_type=cancer_type,
            question_category=category,
            normalized_query=normalized_q,
            extracted_entities=entities,
            metadata_filters=metadata_filters,
            needs_clarification=False,
            confidence=confidence
        )

laqa = LanguageAwareQueryAnalyzer()
