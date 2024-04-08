from typing import Dict, List


class EnMWEPatterns:
    patterns: Dict[str, List[str]] = {}

    def __init__(
        self,
        mwe_types=[
            "Light Verb Constructions",
            "Noun Noun Compounds",
            "Noun Noun Noun Compounds",
            "Adjective Noun Compounds",
            "Adjective Adjective Noun Compounds",
            "Verb Particle Constructions",
        ],
    ):
        if "Light Verb Constructions" in mwe_types:
            self.patterns["Light Verb Constructions"] = [
                "Light Verb Constructions: {<VB*><DT><\\w+>}",
            ]
        if "Noun Noun Compounds" in mwe_types:
            self.patterns["Noun Noun Compounds"] = [
                "Noun Noun Compounds: {<NN|NNS><NN|NNS>}",
            ]
        if "Noun Noun Noun Compounds" in mwe_types:
            self.patterns["Noun Noun Noun Compounds"] = [
                "Noun Noun Noun Compounds: {<NN|NNS><NN|NNS><NN|NNS>}",
            ]
        if "Adjective Noun Compounds" in mwe_types:
            self.patterns["Adjective Noun Compounds"] = [
                "Adjective Noun Compounds: {<JJ><NN|NNS>}",
            ]
        if "Adjective Adjective Noun Compounds" in mwe_types:
            self.patterns["Adjective Adjective Noun Compounds"] = [
                "Adjective Adjective Noun Compounds: {<JJ><JJ><NN|NNS>}"
            ]
        if "Verb Particle Constructions" in mwe_types:
            self.patterns["Verb Particle Constructions"] = [
                "Verb Particle Constructions: {<VB|VBP><RP>}",
            ]


class DeMWEPatterns:
    patterns: Dict[str, List[str]] = {}

    def __init__(
        self,
        mwe_types=[
            "Light Verb Constructions",
            "Noun Noun Compounds",
            "Noun Noun Noun Compounds",
            "Adjective Noun Compounds",
            "Adjective Adjective Noun Compounds",
            "Verb Particle Constructions",
        ],
    ):
        if "Light Verb Constructions" in mwe_types:
            self.patterns["Light Verb Constructions"] = [
                "Light Verb Constructions: {<VB*><DT><\\w+>}",
            ]
        # Define the patterns for 2 and 3-word noun compounds (e.g., "Hausaufgaben", "Fußballplatz")
        if "Noun Noun Compounds" in mwe_types:
            self.patterns["Noun Noun Compounds"] = [
                "Noun Noun Compounds: {<NN|NNS><NN|NNS>}",
            ]
        if "Noun Noun Noun Compounds" in mwe_types:
            self.patterns["Noun Noun Noun Compounds"] = [
                "Noun Noun Noun Compounds: {<NN|NNS><NN|NNS><NN|NNS>}",
            ]
        if "Adjective Noun Compounds" in mwe_types:
            self.patterns["Adjective Noun Compounds"] = [
                "Adjective Noun Compounds: {<JJ><NN|NNS>}",
            ]
        if "Adjective Adjective Noun Compounds" in mwe_types:
            self.patterns["Adjective Adjective Noun Compounds"] = [
                "Adjective Adjective Noun Compounds: {<JJ><JJ><NN|NNS>}"
            ]
        # Define the patterns for verb particle constructions (e.g., "aufstehen", "zurückkommen")
        if "Verb Particle Constructions" in mwe_types:
            self.patterns["Verb Particle Constructions"] = [
                "Verb Particle Constructions: {<VB|VBP><RP>}",
            ]
