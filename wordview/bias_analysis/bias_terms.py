gender_terms_en = {
    "male": [
        "man",
        "male",
        "boy",
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
        "female",
        "girl",
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
    "asian": ["asian", "oriental", "chinese", "japanese", "korean", "vietnamese"],
    "latino": ["latino", "hispanic", "latin", "mexican", "puerto rican"],
    "middle_eastern": [
        "arab",
        "persian",
        "middle eastern",
        "saudi",
        "emirati",
        "iraqi",
    ],
    "native_american": ["native american", "indigenous", "navajo", "cherokee"],
    "indian": ["indian", "hindustani", "bharati"],
}

racial_terms_de = {
    "white": ["weiß", "kaukasisch", "europäisch"],
    "black": ["schwarz", "afrikanisch"],
    "asian": ["asiatisch", "chinesisch", "japanisch", "koreanisch", "vietnamesisch"],
    "latino": ["lateinamerikanisch", "hispanisch", "mexikanisch"],
    "middle_eastern": ["arabisch", "persisch"],
    "native_american": ["Ureinwohner", "indigen"],
    "indian": ["indisch"],
}

religion_terms_en = {
    "christian": [
        "christian",
        "catholic",
        "protestant",
        "baptist",
        "methodist",
        "evangelical",
    ],
    "muslim": ["muslim", "islamic", "sunnah", "shiite", "sufi"],
    "jew": ["jew", "jewish", "hebrew", "yiddish", "zionist"],
    "hindu": ["hindu", "hinduism"],
    "buddhist": ["buddhist", "buddha", "dharma"],
    "atheist": ["atheist", "agnostic", "non-believer", "secular"],
    "sikh": ["sikh", "sikhism", "guru"],
    "bahai": ["bahai", "bahaullah"],
    "shinto": ["shinto", "kami"],
}

religion_terms_de = {
    "christian": ["Christ", "katholisch", "protestantisch", "evangelisch"],
    "muslim": ["Muslim", "islamisch", "Sunni", "Shiite"],
    "jew": ["Jude", "jüdisch", "Hebräer"],
    "hindu": ["Hindu"],
    "buddhist": ["Buddhist", "Buddha"],
    "atheist": ["Atheist", "Agnostiker", "säkular"],
    "sikh": ["Sikh"],
    "bahai": ["Bahai"],
    "shinto": ["Shinto"],
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
