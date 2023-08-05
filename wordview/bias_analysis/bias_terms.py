# TODO Currently does not support multiword terms (e.g. African American, Middle Eastern, Puerto Rican).
# TODO Change the intersection mechanism to add support for multiword terms.

# TODO Currently stemming is not applied to make the matches. For instance, Asian --> Asians etc.
# TODO For some of the words, keywords are added, but this has to be more systematically fixed.

gender_terms_en = {
    "male": [
        "man",
        "men",
        "male",
        "boy",
        "boys",
        "he",
        "his",
        "him",
        "himself",
        "gentleman",
        "guy",
        "dude",
        "lad",
    ],
    "female": [
        "woman",
        "women",
        "female",
        "girl",
        "girls",
        "she",
        "her",
        "hers",
        "herself",
        "lady",
        "gal",
    ],
}

gender_terms_de = {
    "male": [
        "Mann",
        "männlich",
        "Junge",
        "er",
        "sein",
        "sich (m)",
        "Herr",
        "Typ",
        "Kerl",
    ],
    "female": [
        "Frau",
        "weiblich",
        "Mädchen",
        "sie",
        "ihr",
        "sich (f)",
        "Dame",
        "Mädel",
    ],
}

racial_terms_en = {
    "white": ["white", "caucasian", "european", "anglo-saxon"],
    "black": ["black", "african", "afro", "afro-american", "negro"],
    "asian": [
        "asian",
        "asians",
        "oriental",
        "chinese",
        "japanese",
        "korean",
        "vietnamese",
    ],
    "latino": ["latino", "latinos", "hispanic", "latin", "mexican"],
    "indian": ["indian", "indians", "hindustani", "bharati"],
    "middle_eastern": [
        "arab",
        "arabs",
        "persian",
        "iranian",
        "iranians",
        "persians",
        "saudi",
        "emirati",
        "iraqi",
    ],
}

racial_terms_de = {
    "white": ["weiß", "kaukasisch", "europäisch"],
    "black": ["schwarz", "afrikanisch"],
    "asian": ["asiatisch", "chinesisch", "japanisch", "koreanisch", "vietnamesisch"],
    "latino": ["lateinamerikanisch", "hispanisch", "mexikanisch"],
    "indian": ["indisch"],
    "middle_eastern": ["arabisch", "persisch", "iranisch", "iraner"],
}

religion_terms_en = {
    "christian": [
        "christian",
        "christians",
        "catholic",
        "protestant",
        "baptist",
        "methodist",
        "evangelical",
    ],
    "muslim": ["muslim", "muslims", "islamic", "sunnah", "shiite", "sufi"],
    "jew": ["jew", "jews", "jewish", "hebrew", "yiddish", "zionist"],
    "hindu": ["hindu", "hinduism"],
    "buddhist": ["buddhist", "buddha", "dharma"],
    "atheist": ["atheist", "atheists", "agnostic", "non-believer", "secular"],
}

religion_terms_de = {
    "christian": ["Christ", "katholisch", "protestantisch", "evangelisch"],
    "muslim": ["Muslim", "Muslims", "islamisch", "Sunni", "Shiite"],
    "jew": ["Jude", "jüdisch", "Hebräer"],
    "hindu": ["Hindu"],
    "buddhist": ["Buddhist", "Buddha"],
    "atheist": ["Atheist", "Agnostiker", "säkular"],
}


def get_terms(language, category):
    """
    Fetches the set of terms for a given language and category.

    Parameters:
    - language (str): 'en' or 'de'
    - category (str): 'gender', 'racial', or 'religion'

    Returns:
    - dict: Dictionary of terms for the given language and category
    """
    if language == "en":
        if category == "gender":
            return gender_terms_en
        elif category == "racial":
            return racial_terms_en
        elif category == "religion":
            return religion_terms_en
    elif language == "de":
        if category == "gender":
            return gender_terms_de
        elif category == "racial":
            return racial_terms_de
        elif category == "religion":
            return religion_terms_de
    else:
        raise ValueError(f"Unsupported language: {language}")
