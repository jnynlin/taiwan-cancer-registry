"""ICD-O topography C-code to cancer name mapping for CMUH registry."""

SITE_LABELS = {
    "C00": "Lip", "C01": "Base of tongue", "C02": "Other tongue",
    "C03": "Gum", "C04": "Floor of mouth", "C05": "Palate",
    "C06": "Other mouth", "C07": "Parotid gland", "C08": "Other salivary glands",
    "C09": "Tonsil", "C10": "Oropharynx", "C11": "Nasopharynx",
    "C12": "Pyriform sinus", "C13": "Hypopharynx", "C14": "Pharynx NOS",
    "C15": "Esophagus", "C16": "Stomach", "C17": "Small intestine",
    "C18": "Colon", "C19": "Rectosigmoid", "C20": "Rectum",
    "C21": "Anus", "C22": "Liver", "C24": "Other biliary",
    "C25": "Pancreas", "C30": "Nasal cavity", "C31": "Accessory sinuses",
    "C32": "Larynx", "C33": "Trachea", "C34": "Lung",
    "C37": "Thymus", "C38": "Mediastinum", "C40": "Bone (limb)",
    "C41": "Other bone", "C42": "Hematopoietic", "C44": "Skin",
    "C48": "Retroperitoneum", "C49": "Soft tissue", "C50": "Breast",
    "C53": "Cervix uteri", "C54": "Corpus uteri", "C55": "Uterus NOS",
    "C56": "Ovary", "C57": "Other female genital", "C61": "Prostate",
    "C62": "Testis", "C64": "Kidney", "C66": "Ureter",
    "C67": "Bladder", "C69": "Eye", "C71": "Brain",
    "C72": "Other CNS", "C73": "Thyroid", "C74": "Adrenal gland",
    "C75": "Other endocrine", "C76": "Other/ill-defined",
    "C77": "Lymph node", "C80": "Unknown primary",
}

# Reverse lookup: common name → C-code
SITE_ALIASES: dict[str, str] = {
    "esophageal": "C15", "esophagus": "C15", "oesophageal": "C15",
    "stomach": "C16", "gastric": "C16",
    "colon": "C18", "colorectal": "C18", "crc": "C18",
    "rectal": "C20", "rectum": "C20",
    "liver": "C22", "hepatic": "C22", "hcc": "C22", "hepatocellular": "C22",
    "lung": "C34", "pulmonary": "C34", "nsclc": "C34",
    "breast": "C50",
    "cervix": "C53", "cervical": "C53",
    "uterus": "C54", "endometrial": "C54", "uterine": "C54",
    "ovary": "C56", "ovarian": "C56",
    "prostate": "C61", "prostatic": "C61",
    "bladder": "C67",
    "brain": "C71", "glioma": "C71",
    "thyroid": "C73", "ptc": "C73",
    "nasopharynx": "C11", "npc": "C11",
    "hypopharynx": "C13",
    "larynx": "C32",
    "oral": "C06", "mouth": "C06",
    "pancreas": "C25", "pancreatic": "C25",
    "colon": "C18", "colonic": "C18",
    "skin": "C44",
    "lymph": "C77", "lymphoma": "C77",
    "pharynx": "C14",
    "oropharynx": "C10",
    "pyriform": "C12",
    "parotid": "C07",
}

QUICK_SITES = [
    ("C15", "Esophagus"),
    ("C22", "Liver"),
    ("C34", "Lung"),
    ("C50", "Breast"),
    ("C11", "Nasopharynx"),
    ("C73", "Thyroid"),
    ("C61", "Prostate"),
    ("C54", "Uterus"),
    ("C16", "Stomach"),
    ("C18", "Colon"),
]


def label(code: str) -> str:
    """Return human-readable cancer name for a C-code."""
    return SITE_LABELS.get(code.upper(), code)


def extract_site(text: str) -> str | None:
    """Extract first cancer site code from free text. Returns C-code or None."""
    import re
    text_lower = text.lower()

    # Direct C-code match (e.g. "C15", "c22")
    m = re.search(r"\bc(\d{2})\b", text_lower)
    if m:
        return f"C{m.group(1)}"

    # Name alias match — use word boundaries to avoid false substring hits
    # (e.g. "oral" in "temporal", "rectal" in "colorectal")
    for alias, code in sorted(SITE_ALIASES.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
            return code

    return None
