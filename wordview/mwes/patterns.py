import nltk
from typing import List, Dict


class EnMWEPatterns:
    patterns: Dict[str, List[str]] = {}
    def __init__(self,
                 mwe_types=['LVC', 'NC2', 'NC3', 'ANC2', 'ANC3', 'VPC']):
        if 'LVC' in mwe_types:
            self.patterns['LVC'] = [
                    'LVC: {<VB*><DT><\\w+>}',
                ]
        if 'NC2' in mwe_types:
            self.patterns['NC2'] = [
                    'NC2: {<NN|NNS><NN|NNS>}',
                ]
        if 'NC3' in mwe_types:
            self.patterns['NC3'] = [
                    'NC3: {<NN|NNS><NN|NNS><NN|NNS>}',
                ]
        if 'ANC2' in mwe_types:
            self.patterns['ANC2'] = [
                    'ANC2: {<JJ><NN|NNS>}',
                ]
        if 'ANC3' in mwe_types:
            self.patterns['ANC3'] = [
                    'ANC3: {<JJ><JJ><NN|NNS>}'
                ]
        if 'VPC' in mwe_types:
            self.patterns['VPC'] = [
                    'VPC: {<VB|VBP><RP>}',
                ]


class DeMWEPatterns:
    def __init__(self) -> None:
        light_verb_patterns = [
            'LV: {<VB|VBP><DT><\\w+>}',
            'LV: {<VB|VBP><RB><\\w+>}',  # Additional pattern for adverb before noun
        ]
        # Define the patterns for 2 and 3-word noun compounds (e.g., "Hausaufgaben", "Fußballplatz")
        noun_compound_patterns = [
            'NC2: {<NN|NNS><NN|NNS>}',
            'NC3: {<NN|NNS><NN|NNS>}',
        ]
        # Define the patterns for 2 and 3-word adjective-noun compounds (e.g., "blauer Himmel", "großer Baum")
        adj_noun_compound_patterns = [
            'ANC2: {<JJ><NN|NNS>}',
            'ANC3: {<JJ><JJ><NN|NNS>}',
        ]
        # Define the patterns for verb particle constructions (e.g., "aufstehen", "zurückkommen")
        verb_particle_patterns = [
            'VPC: {<VB|VBP><RP>}',
        ]
