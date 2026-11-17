"""Static content: lesion information for the analysis page and the fictional
hospital used to demonstrate how the tool could fit into a clinical workflow."""
from urllib.parse import quote_plus


def _links(search, wiki):
    return {
        "google": "https://www.google.com/search?q=" + quote_plus(search + " skin lesion"),
        "youtube": "https://www.youtube.com/results?search_query=" + quote_plus(search + " dermoscopy"),
        "wikipedia": "https://en.wikipedia.org/wiki/" + wiki,
        "scholar": "https://scholar.google.com/scholar?q=" + quote_plus(search + " dermoscopy deep learning"),
    }


LESIONS = {
    "MEL": {
        "name": "Melanoma", "category": "malignant",
        "summary": "A malignant tumour of melanocytes, the pigment-producing cells of the skin. It is the "
                   "most dangerous common skin cancer because it can spread early, but it is highly "
                   "curable when removed at a thin stage.",
        "features": ["Atypical pigment network", "Irregular dots and globules", "Blue-white veil",
                     "Regression structures", "Irregular streaks at the periphery"],
        "management": "Suspected melanoma is referred for excision biopsy and histopathology. Treatment "
                      "and follow-up depend on tumour thickness and stage.",
        "links": _links("melanoma", "Melanoma"),
    },
    "NV": {
        "name": "Melanocytic nevus", "category": "benign",
        "summary": "A common mole: a benign collection of melanocytes. Most adults have several, and "
                   "the vast majority never become malignant.",
        "features": ["Regular pigment network", "Symmetric colour and structure", "Uniform globules"],
        "management": "Usually no treatment. Changing, symptomatic or atypical-looking naevi are "
                      "reviewed and sometimes removed for histology.",
        "links": _links("melanocytic nevus", "Melanocytic_nevus"),
    },
    "BCC": {
        "name": "Basal cell carcinoma", "category": "malignant",
        "summary": "The most common skin cancer. It grows slowly and very rarely spreads, but can "
                   "destroy surrounding tissue if left untreated.",
        "features": ["Arborizing (branching) vessels", "Blue-grey ovoid nests", "Leaf-like areas",
                     "Ulceration", "Spoke-wheel structures"],
        "management": "Confirmed by biopsy and usually treated by surgical excision; superficial types "
                      "may be treated with topical therapy or other local methods.",
        "links": _links("basal cell carcinoma", "Basal-cell_carcinoma"),
    },
    "AKIEC": {
        "name": "Actinic keratosis / intraepithelial carcinoma", "category": "pre-malignant",
        "summary": "Rough, scaly sun-damage lesions (actinic keratosis) and Bowen's disease. They can "
                   "progress to invasive squamous cell carcinoma.",
        "features": ["Strawberry pattern (red pseudo-network)", "Surface scale", "Glomerular vessels",
                     "Rosettes"],
        "management": "Treated with cryotherapy, topical creams or photodynamic therapy; thick or "
                      "atypical lesions are biopsied.",
        "links": _links("actinic keratosis", "Actinic_keratosis"),
    },
    "BKL": {
        "name": "Benign keratosis", "category": "benign",
        "summary": "Seborrhoeic keratosis, solar lentigo and lichen-planus-like keratosis: harmless "
                   "growths that become more common with age.",
        "features": ["Milia-like cysts", "Comedo-like openings", "Sharp demarcation",
                     "Fingerprint-like structures"],
        "management": "No treatment is needed; removal is cosmetic or for irritation.",
        "links": _links("seborrheic keratosis", "Seborrheic_keratosis"),
    },
    "DF": {
        "name": "Dermatofibroma", "category": "benign",
        "summary": "A small, firm, harmless nodule of fibrous tissue, often on the legs and sometimes "
                   "after a minor injury or insect bite.",
        "features": ["Central white scar-like patch", "Delicate peripheral pigment network",
                     "Dimples inward when pinched"],
        "management": "Usually left alone; excision if symptomatic or the diagnosis is uncertain.",
        "links": _links("dermatofibroma", "Dermatofibroma"),
    },
    "VASC": {
        "name": "Vascular lesion", "category": "benign",
        "summary": "Lesions of blood vessels such as cherry angiomas, angiokeratomas and pyogenic "
                   "granulomas.",
        "features": ["Red, purple or blue lacunae", "Well-defined borders", "Little or no pigment network"],
        "management": "Usually benign and left alone; treated with laser or removal if they bleed or "
                      "for cosmetic reasons.",
        "links": _links("cherry angioma", "Cherry_hemangioma"),
    },
}

REFER_CLASSES = {"MEL", "BCC", "AKIEC"}

# Fictional demonstration site. No real institution, staff or contact details.
HOSPITAL = {
    "name": "Himalaya District Hospital",
    "note": "Fictional hospital used to demonstrate the workflow. It is not a real deployment.",
    "location": "A district hospital in the hills of Nepal",
    "beds": 100,
    "services": [
        {"name": "Outpatient dermatology clinic",
         "text": "Health assistants see skin complaints daily and capture dermoscopic images of lesions."},
        {"name": "Visiting dermatologist",
         "text": "A consultant dermatologist visits on fixed days and reviews referred cases first."},
        {"name": "Minor procedures theatre",
         "text": "Excision biopsies are performed on visiting days; samples go to the regional pathology lab."},
        {"name": "Tele-dermatology link",
         "text": "Difficult cases can be shared with the referral centre in Kathmandu between visits."},
    ],
    "schedule": [
        {"role": "Consultant dermatologist (visiting)", "days": [1, 3], "hours": "09:00 – 13:00"},
        {"role": "Dermatology medical officer", "days": [0, 1, 2, 3, 4, 5], "hours": "10:00 – 16:00"},
        {"role": "Dermatopathology review (remote)", "days": [4], "hours": "By referral"},
    ],
    "appointment_days": [1, 3],            # Monday = 0; visiting-dermatologist days
    "slots": ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30"],
}
