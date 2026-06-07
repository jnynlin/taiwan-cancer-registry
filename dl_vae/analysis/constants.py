"""
Canonical axis set definitions for CMUH cancer registry analysis.

Single source of truth — all scripts import from here.
Biological basis: VAE latent space analysis (Scripts 02-03), locked 2026-06-07.

  UADT     — Upper aero-digestive tract; tobacco/betel/alcohol mucosal field
  GI_SYS   — GI/systemic; HBV/metabolic/alcohol hepatic-enteric axis
  HORMONAL — Hormonal/reproductive; androgen/oestrogen-driven

C34 lung and C11 nasopharynx confirmed UADT (tobacco co-exposure, z4/z7 VAE loadings).
C61 prostate confirmed HORMONAL (androgen-driven, not GI/metabolic).
C32 larynx confirmed UADT (tobacco/alcohol UADT anatomy).
"""

UADT: frozenset = frozenset({
    "C02", "C03", "C04", "C05", "C06",
    "C09", "C10", "C11", "C12", "C13",
    "C15", "C32", "C34",
})

GI_SYS: frozenset = frozenset({
    "C16", "C17", "C18", "C19", "C20",
    "C22", "C25", "C67",
})

HORMONAL: frozenset = frozenset({
    "C50", "C53", "C54", "C56", "C61",
})

# Aliases used by some scripts
UADT_SITES     = UADT
HORMONAL_SITES = HORMONAL

# Subset of GI_SYS used by Script 03 axis_name() heuristic
# (liver + lower GI; tighter than full GI_SYS for VAE dimension labelling)
LIVER_GI_SITES: frozenset = frozenset({"C18", "C20", "C22", "C25"})


def axis_label(site: str, target: str | None = None) -> str:
    """Return canonical axis label for a site code.

    Args:
        site:   ICD-O-3 C-code (e.g. 'C34').
        target: Optional hub site treated as its own category (used in Script 16
                where C22 is the analysis TARGET, not classified as GI/systemic).

    Returns one of: 'UADT', 'GI/systemic', 'Hormonal', '<target> hub', 'Other'.
    """
    if target and site == target:
        return f"{target} hub"
    if site in UADT:
        return "UADT"
    if site in GI_SYS:
        return "GI/systemic"
    if site in HORMONAL:
        return "Hormonal"
    return "Other"
