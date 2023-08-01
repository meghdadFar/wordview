import nltk
from typing import List, Dict


class ENMWEPatterns:
    patterns: Dict[str, List[str]] = {}
    def __init__(self,
                 mwe_types=['LVC', 'NC2', 'NC3', 'ANC2', 'ANC3', 'VPC']):
        if 'LVC' in mwe_types:
            self.patterns['LVC'] = [
                    'LVC: {<VB|VBP><DT><\\w+>}',
                    'LVC: {<VB|VBP><RB><\\w+>}',
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


class DEMWEPatterns:
    def __init__(self,
                    mwe_types=['LVC', 'NC2', 'NC3', 'ANC2', 'ANC3', 'VPC']):
        self.patterns: Dict[str, List[str]] = {}
        if 'LVC' in mwe_types:
            self.patterns['LVC'] = [
                    'LVC: {<VB|VBP><DT><\\w+>}',
                    'LVC: {<VB|VBP><RB><\\w+>}',
                ]
        # Hausaufgaben"
        if 'NC2' in mwe_types:
            self.patterns['NC2'] = [
                    'NC2: {<NN|NNS><NN|NNS>}',
                ]
        # Fußballplatz
        if 'NC3' in mwe_types:
            self.patterns['NC3'] = [
                    'NC3: {<NN|NNS><NN|NNS>}',
                ]
        # blauer Himmel
        if 'ANC2' in mwe_types:
            self.patterns['ANC2'] = [
                    'ANC2: {<JJ><NN|NNS>}',
                ]
        if 'ANC3' in mwe_types:
            self.patterns['ANC3'] = [
                    'ANC3: {<JJ><JJ><NN|NNS>}',
                ]
        # aufstehen zurückkommen
        if 'VPC' in mwe_types:
            self.patterns['VPC'] = [
                    'VPC: {<VB|VBP><RP>}',
                ]
        

        # Combine all patterns
        all_patterns = '\n'.join(light_verb_patterns + noun_compound_patterns +
                                 adj_noun_compound_patterns + verb_particle_patterns)

        # Create the RegexpParser with all patterns
        self.chunk_parser = nltk.RegexpParser(all_patterns)

    def parse(self, sentence):
        # Tokenize the sentence
        tokens = nltk.word_tokenize(sentence)

        # Part-of-speech tagging
        pos_tags = nltk.pos_tag(tokens)

        # Use the tagged sentence for parsing
        parsed_tree = self.chunk_parser.parse(pos_tags)

        return parsed_tree


class DEMWEPatterns:
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
    


# Example usage
if __name__ == "__main__":
    custom_chunker_en = Patterns(language='EN')
    custom_chunker_de = Patterns(language='DE')

    sentence_en = "I will take a walk and give a speech. The coffee shop near the swimming pool sells red apples."
    sentence_de = "Ich werde einen Spaziergang machen und eine Rede halten. Der Coffeeshop in der Nähe des Schwimmbads verkauft rote Äpfel."

    parsed_tree_en = custom_chunker_en.parse(sentence_en)
    parsed_tree_de = custom_chunker_de.parse(sentence_de)

    # Print the parsed trees
    print("English:", parsed_tree_en)
    print("German:", parsed_tree_de)
